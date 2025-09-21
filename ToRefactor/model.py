# src/models/graph_jepa.py
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Batch


# -----------------------------
# 1) Утилиты/лоссы
# -----------------------------
def cosine_loss(h_pred, h_tgt):
    hp = F.normalize(h_pred, dim=-1, eps=1e-8)
    ht = F.normalize(h_tgt, dim=-1, eps=1e-8)
    cos = (hp * ht).sum(-1)          # per vector cosine
    return 1.0 - cos.mean()


def variance_loss(H: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    # VICReg-style стабилизатор против коллапса
    # H: [N, d]
    std = H.std(dim=0) + eps
    return torch.relu(target_std - std).mean()


# -----------------------------
# 2) Простой GIN-энкодер патча
# -----------------------------

class PatchGINEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, depth: int = 3, out_dim: int = 128,
                 dropout: float = 0.1, use_bn: bool = True):
        super().__init__()
        layers = []
        last = in_dim
        for _ in range(depth):
            mlp = nn.Sequential(
                nn.Linear(last, hidden),
                nn.BatchNorm1d(hidden) if use_bn else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, hidden),
            )
            layers.append(GINConv(mlp))
            last = hidden
        self.gins = nn.ModuleList(layers)
        self.proj = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor,
                return_node_repr: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = x
        for conv in self.gins:
            h = conv(h, edge_index)
            h = F.relu(h, inplace=True)
            h = self.dropout(h)

        h_patch = global_mean_pool(h, batch)   
        h_patch = self.proj(h_patch)          
        if return_node_repr:
            return h_patch, h
        return h_patch


# -----------------------------
# 3) Узкий MLP-предиктор
# -----------------------------
class Predictor(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        dim = in_dim
        for i in range(depth - 1):
            layers += [nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            dim = hidden
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# -----------------------------
# 4) Конфиг и EMA расписание
# -----------------------------
@dataclass
class JEPAConfig:
    in_dim: int                  # размерность node-фич в Data.x
    d: int = 128                 # латент энкодера
    enc_hidden: int = 128
    enc_depth: int = 3
    enc_dropout: float = 0.1
    use_bn: bool = True

    pred_hidden: int = 64        # узкий предиктор (<= d)
    pred_depth: int = 2
    pred_dropout: float = 0.1

    z_dim: int = 2               # размер Z (из датасета)
    m_targets: int = 4           # m из датасета (для sanity-чеков)

    lr: float = 1e-3
    weight_decay: float = 1e-4
    ema_tau_base: float = 0.99
    ema_tau_final: float = 0.9995
    ema_warmup_steps: int = 10_000

    var_loss_beta: float = 1e-3  # коэффициент variance loss (0 чтобы выключить)

def ema_momentum(step: int, cfg: JEPAConfig) -> float:
    # плавно растём от tau_base к tau_final по косинус-расписанию в первые ema_warmup_steps
    if cfg.ema_warmup_steps <= 0:
        return cfg.ema_tau_final
    t = min(1.0, step / float(cfg.ema_warmup_steps))
    cos_scale = 0.5 * (1 + math.cos(math.pi * (1 - t)))  # 0->1
    return cfg.ema_tau_base * (1 - cos_scale) + cfg.ema_tau_final * cos_scale


# -----------------------------
# 5) Lightning-модель JEPA
# -----------------------------
class GraphJEPA(pl.LightningModule):
    """
    Ожидает batch из твоего jepa_collate:
      batch = {
        "context_batch": DataBatch,
        "target_batch":  DataBatch,
        "Z": Tensor[B*m, p],
        "m": int,
        "B": int
      }
    """
    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg

        self.encoder = PatchGINEncoder(
            in_dim=cfg.in_dim, hidden=cfg.enc_hidden, depth=cfg.enc_depth,
            out_dim=cfg.d, dropout=cfg.enc_dropout, use_bn=cfg.use_bn,
        )
        # таргет-энкодер = EMA-копия
        self.target_encoder = PatchGINEncoder(
            in_dim=cfg.in_dim, hidden=cfg.enc_hidden, depth=cfg.enc_depth,
            out_dim=cfg.d, dropout=cfg.enc_dropout, use_bn=cfg.use_bn,
        )
        self._init_target_encoder()

        # предиктор принимает [h_ctx_rep || Z]
        self.predictor = Predictor(
            in_dim=cfg.d + cfg.z_dim, hidden=cfg.pred_hidden,
            out_dim=cfg.d, depth=cfg.pred_depth, dropout=cfg.pred_dropout,
        )

        # счётчик шагов для EMA
        self.register_buffer("_ema_step", torch.zeros((), dtype=torch.long), persistent=False)

    # --- init EMA weights ---
    def _init_target_encoder(self):
        for p_t, p_e in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            p_t.data.copy_(p_e.data)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    # --- основной лосс JEPA ---
    def jepa_loss(self, h_ctx: torch.Tensor, Z: torch.Tensor, h_tgt: torch.Tensor, m: int) -> torch.Tensor:
        """
        h_ctx: [B, d]
        Z:     [B*m, p]
        h_tgt: [B*m, d]
        """
        # повторим контекст m раз
        h_ctx_rep = h_ctx.repeat_interleave(m, dim=0)  # [B*m, d]
        # предиктор
        h_pred = self.predictor(torch.cat([h_ctx_rep, Z], dim=-1))  # [B*m, d]

        loss = cosine_loss(h_pred, h_tgt)
        if self.cfg.var_loss_beta > 0:
            # variance рег: и на предсказаниях, и на таргетах
            loss = loss + self.cfg.var_loss_beta * (variance_loss(h_pred) + variance_loss(h_tgt))
        return loss

    # --- EMA апдейт ---
    @torch.no_grad()
    def _ema_update(self):
        self._ema_step += 1
        tau = ema_momentum(int(self._ema_step.item()), self.cfg)
        for p_t, p_e in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            p_t.data.mul_(tau).add_(p_e.data, alpha=(1.0 - tau))

    # --- Lightning hooks ---
    def training_step(self, batch, batch_idx = None):
        C: Batch = batch["context"]
        T: Batch = batch["targets"]
        Z: torch.Tensor = batch["z"]  # [B*m, p]
        m: int = batch["m"]
       

        # 1) контекст-латенты
        h_ctx: torch.Tensor = self.encoder(C.x, C.edge_index, C.batch)  # [B, d]

        # 2) таргет-латенты (EMA-энкодер, без градиента)
        with torch.no_grad():
            h_tgt: torch.Tensor = self.target_encoder(T.x, T.edge_index, T.batch)  # [B*m, d]

        # 3) JEPA лосс
        loss = self.jepa_loss(h_ctx, Z, h_tgt, m)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        # 4) шаг оптимайзера
        return loss

    def on_after_backward(self):
        # после градиента обновляем EMA-энкодер
        self._ema_update()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        C: Batch = batch["context"]
        T: Batch = batch["target_batch"]
        Z: torch.Tensor = batch["Z"]
        m: int = batch["m"]

        h_ctx = self.encoder(C.x, C.edge_index, C.batch)
        h_tgt = self.target_encoder(T.x, T.edge_index, T.batch)
        val_loss = self.jepa_loss(h_ctx, Z, h_tgt, m)
        self.log("val/loss", val_loss, prog_bar=True, on_epoch=True, sync_dist=False)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        # можно добавить CosineAnnealing или т.п.
        return opt
