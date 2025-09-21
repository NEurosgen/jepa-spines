import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from model.Predictor import Predictor
from model.GNN import GNN
from model.GIN import GIN



import torch
import torch.nn as nn

class Predictor(nn.Module):
    def __init__(self, latent_dim: int, pe_dim: int,
                 hidden_dim: int, out_dim: int = 2,
                 num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        # отдельные проекции для контекста и подсказки
        self.proj_ctx = nn.Linear(latent_dim, hidden_dim)
        self.proj_pe  = nn.Linear(pe_dim,     hidden_dim)

        layers = []
        dim = 2 * hidden_dim  # concat[ctx_proj, pe_proj]
        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            dim = hidden_dim
        layers += [nn.Linear(dim, out_dim)]  # -> (cosh, sinh)
        self.net = nn.Sequential(*layers)

    def forward(self, context: torch.Tensor, hint_P: torch.Tensor):
        # context: (M, latent_dim), hint_P: (M, pe_dim)
        
        c = self.proj_ctx(context)   # (M, hidden_dim)
        p = self.proj_pe(hint_P)     # (M, hidden_dim)
        x = torch.cat([c, p], dim=-1)
        return self.net(x)           # (M, 2)


import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class GNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        assert layers >= 1
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # стек свёрток
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # (опц.) нормализации после каждого слоя, кроме последнего
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layers)])

    def forward(self, data):
        """
        data: torch_geometric.data.Data или Batch
              должен иметь .x, .edge_index и (если это Batch) .batch
        Возврат: (B, hidden_dim) — графовые эмбеддинги после mean-pool.
                 Если это одиночный граф без .batch, вернёт (1, hidden_dim).
        """
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)  # тензор размера [num_nodes] или None

        for conv, ln in zip(self.convs, self.norms):
            x = conv(x, edge_index)           # (num_nodes, hidden_dim)
            x = ln(x)
            x = self.act(x)
            x = self.dropout(x)

        if batch is None:
            # одиночный граф (нет .batch): усредняем по узлам
            return x.mean(dim=0, keepdim=True)         # (1, hidden_dim)
        else:
            # батч из нескольких графов: пулим по индексу графа в .batch
            return global_mean_pool(x, batch)          # (B, hidden_dim)



class JEPAModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=["cfg"])
        self.student_encoder = GNN(in_dim=cfg.model.num_features, hidden_dim=cfg.model.latent_dim)
        self.teacher_encoder = GNN(in_dim=cfg.model.num_features, hidden_dim=cfg.model.latent_dim)


        self.predictor = Predictor(
            latent_dim=self.cfg.model.latent_dim,
            pe_dim=self.cfg.model.pe_dim,                 # = rwse_k из датасета (16)
            hidden_dim=self.cfg.model.predictor_hidden_dim,
            out_dim=2,
            num_layers=getattr(self.cfg, "pred_layers", 2),
            dropout=getattr(self.cfg, "dropout", 0.0),
        )


        self.teacher_encoder.load_state_dict(self.student_encoder.state_dict())
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False
        self.teacher_encoder.eval()

    
        self.ema_tau = float(getattr(cfg, "ema_tau", 0.996))
        self.smoothl1_beta = float(getattr(cfg, "smoothl1_beta", 1.0))

    @torch.no_grad()
    def _ema_update(self):
        """EMA-обновление весов учителя из студента."""
        tau = self.ema_tau
        for ps, pt in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            pt.data.mul_(tau).add_(ps.data, alpha=1.0 - tau)

    @torch.no_grad()
    def psi_from_target_latent(self, Zy: torch.Tensor) -> torch.Tensor:

        alpha = Zy.mean(dim=-1, keepdim=True)            
        return torch.cat([torch.cosh(alpha), torch.sinh(alpha)], dim=-1)  

    def cos_loss(self, context_lt, target_lt):
        context_lt = F.normalize(context_lt, dim=-1)
        target_lt  = F.normalize(target_lt,  dim=-1)
        return 1.0 - (target_lt * context_lt).sum(dim=-1).mean()

    def forward(self, x, edge_index):
        z = self.student_encoder(x, edge_index)
        return z

    def training_step(self, batch, batch_idx):
        ctx_batch = batch["context_batch"]
        tgt_batch = batch["target_batch"]
        P = batch["encoded"]           # [sum_T, k]
        sizes = batch["sizes"]         # [B]

        # If there are no target patches in this batch, skip (return 0 loss)
        if tgt_batch is None or P.numel() == 0 or sizes.sum().item() == 0:
            loss = P.new_zeros(())
            print("FAIL")
            self.log("train_loss", loss, prog_bar=True, on_epoch=True)
            return loss

        # Move to device
        device = self.device
        ctx_batch = ctx_batch.to(device)
        tgt_batch = tgt_batch.to(device)
        P = P.to(device)

        z_ctx = self.student_encoder(ctx_batch)        # expect [B, Dx]
        assert z_ctx.dim() == 2, "student_encoder must return [B, D] for context_batch"
        z_ctx_rep = torch.repeat_interleave(z_ctx, sizes.to(device), dim=0)  # [sum_T, Dx]

        hat_psi = self.predictor(z_ctx_rep, P)
        with torch.no_grad():
            Z_y = self.teacher_encoder(tgt_batch)      # [sum_T, Dy]
            psi = self.psi_from_target_latent(Z_y)     # [sum_T, *] to match hat_psi

        # Smooth L1 loss
        loss_main = F.smooth_l1_loss(hat_psi, psi, beta=getattr(self, "smoothl1_beta", 1.0), reduction="mean")
        lambda_cos = getattr(self.cfg.model, "lambda_cos", 0.1)  # добавь в конфиг при желании
        loss_aux = self.cos_loss(z_ctx_rep, Z_y.detach())

        loss = loss_main + lambda_cos * loss_aux
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)

        # EMA update for teacher (if you use BYOL/MoCo-style teacher)
        self._ema_update()
        return loss


    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        ctx_batch = batch["context_batch"]     # PyG Batch of B context graphs
        tgt_batch = batch["target_batch"]      # PyG Batch of sum_T target patches
        P        = batch["encoded"]            # [sum_T, k] target embeddings
        sizes    = batch["sizes"]              # [B] targets per context

        if tgt_batch is None or P.numel() == 0 or sizes.sum().item() == 0:
            loss = torch.tensor(0.0, device=self.device)
            print("FAIL")
            self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
            return loss

        device = self.device
        ctx_batch = ctx_batch.to(device)
        tgt_batch = tgt_batch.to(device)
        P = P.to(device)

        with torch.no_grad():
            z_ctx = self.student_encoder(ctx_batch)          

            z_ctx_rep = torch.repeat_interleave(z_ctx, sizes.to(device), dim=0)

          
            hat_psi = self.predictor(z_ctx_rep, P)           # [sum_T, *]

            # Teacher encodes target patches -> derive target signal
            Z_y = self.teacher_encoder(tgt_batch)            # [sum_T, Dy]
            psi = self.psi_from_target_latent(Z_y)           # [sum_T, *]

            loss = F.smooth_l1_loss(
                hat_psi, psi,
                beta=getattr(self, "smoothl1_beta", 1.0),
                reduction="mean"
            )

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss


    def configure_optimizers(self):
        optim = torch.optim.SGD(
            list(self.student_encoder.parameters()) + list(self.predictor.parameters()),
            lr=float(self.cfg.model.lr),
            momentum=0.9,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.cfg.trainer.max_epochs)
        return {"optimizer": optim, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

    def on_train_epoch_start(self):
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", lr, prog_bar=True, on_epoch=True)