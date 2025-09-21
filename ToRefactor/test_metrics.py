# latent_worker.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch

# --- твои импорты (исправил опечатки в названиях)
# from omegaconf import DictConfig
# from graph import GetDatasetGraph   # <- если у тебя есть готовая функция
from rep_eval import evaluate_representations  # , RepEvaluator, RepEvalConfig
from run_umap_vis import plot_umap_2d, plot_umap_3d

from KkopPatchDataset import KHopPatchDataset,collate_ctx_targets_single,collate_ctx_targets_many,collate_ctx_with_targets


from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
from torch import Tensor
from torch_geometric.data import Batch

from graph_cut_datset import make_loader  # your collate returns a dict


@dataclass
class EmbedConfig:
    per_graph: bool = False                 # True: one vector per base-graph (pool over targets, optionally include context)
    include_context_in_pool: bool = True    # pool context together with targets in per_graph mode
    use_encoder: str = "student"            # "teacher" | "student"
    batch_size: int = 1
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory: bool = False


@torch.no_grad()
def encode_batch_with(model, batch: Batch, which: str) -> Tensor:
    # Expect encoder to return [num_graphs, D] for a PyG Batch
    enc = model.teacher_encoder if which == "teacher" else model.student_encoder
    return enc(batch)


def _get_graph_labels_safe(batch: Optional[Batch]) -> Optional[Tensor]:
    """Return graph-level labels if present; None otherwise."""
    if batch is None:
        return None
    y = getattr(batch, "y", None)
    if y is None:
        return None
    if not isinstance(y, torch.Tensor):
        return None
    # We assume graph-level labels: shape [num_graphs] or [num_graphs, ...]
    # If someone packed node-level labels, this will likely be longer; we
    # still slice to the number of graphs to be safe.
    n = int(batch.num_graphs)
    y = y.view(n, -1) if y.dim() > 1 else y.view(n)
    return y


from typing import Tuple, Optional, List
import torch
from torch import Tensor
from torch_geometric.data import Batch

# предполагаю, что у тебя уже есть:
# - make_loader(dataset, batch_size, shuffle, num_workers, pin_memory)
# - encode_batch_with(model, batch: Batch, which: str) -> Tensor
# - _get_graph_labels_safe(batch: Batch) -> Optional[Tensor]

@torch.no_grad()
def get_embeddings_ctx_targets(
    model,
    dataset,
    cfg,   # DictConfig нового формата (cfg.model.*, cfg.trainer.*)
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Возвращает:
      X:
        - если cfg.model.per_graph=True: [N_graphs, D] (пуллинг контекста + таргетов по каждому графу)
        - иначе: сначала все контексты, затем все таргеты: [N_ctx + N_tgt, D]
      y:
        - лейблы (если есть у графов): выровнены с X (контексты, затем таргеты) или только по графам при per_graph=True
    """

    # --- читаем нужные поля из нового конфига с безопасными дефолтами ---
    per_graph = bool(getattr(cfg.model, "per_graph", False))
    include_ctx = bool(getattr(cfg.model, "include_context_in_pool", True))
    which_enc = str(getattr(cfg.model, "use_encoder", "student"))

    bs          = int(getattr(cfg.trainer, "batch_size", 1))
    num_workers = int(getattr(cfg.trainer, "num_workers", 0))
    pin_memory  = bool(getattr(cfg.trainer, "pin_memory", False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    loader = make_loader(
        dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    X_list: List[Tensor] = []
    y_list: List[Tensor] = []

    for batch in loader:
        ctx_b: Batch = batch["context_batch"]                 # [B] контекстные графы
        tgt_b: Optional[Batch] = batch.get("target_batch")    # [sum_T] таргеты или None
        sizes: torch.Tensor = batch["sizes"]                  # [B] число таргетов на каждый контекст

        if ctx_b is None or ctx_b.num_graphs == 0:
            continue

        ctx_b = ctx_b.to(device)
        if tgt_b is not None and tgt_b.num_graphs > 0:
            tgt_b = tgt_b.to(device)
        else:
            tgt_b = None  # нормализуем к None

        # эмбеддинги контекстов: [B, D]
        z_ctxB = encode_batch_with(model, ctx_b, which_enc)

        if per_graph:
            # заранее (по возможности) посчитаем эмбеддинги таргетов
            z_tgt_all = encode_batch_with(model, tgt_b, which_enc) if (tgt_b is not None) else None

            parts = []
            start = 0
            for i, t in enumerate(sizes.tolist()):
                vecs = []
                if include_ctx:
                    vecs.append(z_ctxB[i:i+1])  # [1, D]
                if z_tgt_all is not None and t > 0:
                    end = start + t
                    vecs.append(z_tgt_all[start:end].mean(0, keepdim=True))  # [1, D]
                pooled = torch.cat(vecs, dim=0).mean(0, keepdim=True) if len(vecs) > 0 else z_ctxB[i:i+1]
                parts.append(pooled)
                start += t

            X_graph = torch.cat(parts, dim=0)  # [B, D]
            X_list.append(X_graph.detach().cpu())

            # лейблы только по графам-контекстам (если есть)
            y_ctx = _get_graph_labels_safe(ctx_b)
            if y_ctx is not None:
                y_list.append(y_ctx.detach().cpu())

        else:
            # без пуллинга: сначала все контексты, потом все таргеты
            X_list.append(z_ctxB.detach().cpu())
            y_ctx = _get_graph_labels_safe(ctx_b)
            if y_ctx is not None:
                y_list.append(y_ctx.detach().cpu())

            if tgt_b is not None and tgt_b.num_graphs > 0:
                z_tgt = encode_batch_with(model, tgt_b, which_enc)  # [sum_T, D]
                X_list.append(z_tgt.detach().cpu())

                y_tgt = _get_graph_labels_safe(tgt_b)
                if y_tgt is not None:
                    y_list.append(y_tgt.detach().cpu())

    X = torch.cat(X_list, dim=0) if len(X_list) > 0 else torch.empty((0, 0))
    y = torch.cat(y_list, dim=0) if len(y_list) > 0 else None
    return X, y


# =========================
# Загрузка чекпойнта (Lightning)
# =========================
from model.JEPA import JEPAModel
def get_last_model(ckpt_root: str | Path, cfg ,version: Optional[int] = None):
    """
    Ищет чекпойнты в структуре Lightning:
      lightning_logs/version_{K}/checkpoints/*.ckpt
    Если version=None — берёт наибольший K. Из чекпойнтов берём
    приоритетно 'last.ckpt', иначе самый свежий по времени.
    Вернёт torch.nn.Module (ожидается, что у тебя есть класс модели и способ загрузки state_dict).
    """
    ckpt_root = Path(ckpt_root)
    logs = ckpt_root / "lightning_logs"

    if version is None:
        versions = sorted([p for p in logs.glob("version_*") if p.is_dir()],
                          key=lambda p: int(p.name.split("_")[-1]))
        if not versions:
            raise FileNotFoundError(f"No versions in {logs}")
        vdir = versions[-1]
    else:
        vdir = logs / f"version_{version}"
        if not vdir.exists():
            raise FileNotFoundError(f"{vdir} not found")

    cdir = vdir / "checkpoints"
    ckpts = list(cdir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {cdir}")

    # приоритет last.ckpt
    last = [c for c in ckpts if c.name == "last.ckpt"]
    if last:
        ckpt_path = last[0]
    else:
        ckpt_path = max(ckpts, key=lambda p: p.stat().st_mtime)

    # ======== ВАЖНО: здесь подставь код инициализации твоей модели ========

    
    model = JEPAModel.load_from_checkpoint(str(ckpt_path),cfg=cfg,                       # <<— главное!
    strict=False )  # Lightning
    model.eval()
    return model



# =========================
# Загрузка датасета
# =========================
from ExperimentalDataset import SpineGraphDataset


from GraphCut.graphcutter import (
    GraphCutter, KHop, BFSBudget, RandomWalk,
    NodeBudget, EnsureConnected, Unique,
    InduceEdges, Relabel, save_patch_npz
)

def get_strategy(mode: str):
    mode = (mode or "").lower()
    if mode == "bfs":
        return BFSBudget
    if mode == "khop":
        return KHop
def build_cutters(cfg: DictConfig):
    # контекст — по желанию можно добавить стратегию из конфига
    ctx_strats = []
    if "ctx_mode" in cfg.cutter.strategy and cfg.cutter.strategy.ctx_mode is not None:
        ctx_cls = get_strategy(cfg.cutter.strategy.ctx_mode)
        ctx_strats.append(ctx_cls(cfg.cutter.strategy.ctx_param))

    ctx_cutter = GraphCutter(
        strategies=ctx_strats,
        constraints=[EnsureConnected(), Unique()],
        post=[InduceEdges(), Relabel()],
        rng_seed=getattr(cfg, "seed", 0)
    )

    tgt_cls = get_strategy(cfg.cutter.strategy.tgt_mode)
    tgt_cutter = GraphCutter(
        strategies=[tgt_cls(cfg.cutter.strategy.tgt_param)],
        constraints=[EnsureConnected(), Unique()],
        post=[InduceEdges(), Relabel()],
        rng_seed=getattr(cfg, "seed", 0)
    )
    return ctx_cutter, tgt_cutter


from normalize_graph import _apply_norm,compute_feature_stats
def apply_norm_inplace(dataset, params, method="zscore", attr="x", eps=1e-8, clip=None):
    for data in dataset:
        if hasattr(data, attr) and getattr(data, attr) is not None:
            x = getattr(data, attr)
            x_norm = _apply_norm(x, method=method, params=params,
                                 eps=eps, per_graph=False, clip=clip)
            setattr(data, attr, x_norm)
def get_dataset(cfg):
    base_ds =SpineGraphDataset(
        root='Test',
        knn=cfg.data.knn,
        count_per_item=cfg.data.count_per_item,
        mode=cfg.data.mode,
        gamma=cfg.data.gamma
    )
    print("Size dataset", len(base_ds))
    ctx_cutter,tgtx_cutter = build_cutters(cfg)
    # 2. Разделение на train/val (80/20)
  


    method = "zscore"  # или "minmax" / "robust"
    params = compute_feature_stats(base_ds, method=method, attr="x")
    apply_norm_inplace(base_ds, params, method=method, attr="x")
    patches_set = CutPatchesDataset(base_graphs=base_ds,context_cutter=ctx_cutter,targets_cutter=tgtx_cutter,rwse_k=cfg.model.pe_dim,max_targets=cfg.data.max_patches)
    return patches_set
# =========================
# Пример main (пайплайн)
# =========================
import hydra
from omegaconf import DictConfig
from graph_cut_datset import CutPatchesDataset
# --- NEW: универсальный загрузчик patches под любой root ---
def build_patches_set(root_path: str | Path, cfg: DictConfig):
    if root_path == 'Test':
        base_ds = SpineGraphDataset(
            root=str(root_path),
            knn=cfg.data.knn,
            count_per_item=0,
            mode=cfg.data.mode,
            gamma=cfg.data.gamma,
        ) 
    else:   
        base_ds = SpineGraphDataset(
            root=str(root_path),
            knn=cfg.data.knn,
            count_per_item=cfg.data.count_per_item,
            mode=cfg.data.mode,
            gamma=cfg.data.gamma,
        )
    print(f"[build_patches_set] root={root_path} size={len(base_ds)}")
    ctx_cutter, tgt_cutter = build_cutters(cfg)

    # нормализация по ТОМУ ЖЕ root (без утечки между train/test)
    method = "zscore"
    params = compute_feature_stats(base_ds, method=method, attr="x")
    apply_norm_inplace(base_ds, params, method=method, attr="x")

    return CutPatchesDataset(
        base_graphs=base_ds,
        context_cutter=ctx_cutter,
        targets_cutter=tgt_cutter,
        rwse_k=cfg.model.pe_dim,
        max_targets=cfg.data.max_patches,
    )

# --- NEW: обучение и оценка линейной модели на эмбеддингах ---
def fit_and_eval_linear(X_tr: Tensor, y_tr: Tensor, X_te: Tensor, y_te: Tensor):
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score

    # to numpy
    X_tr = X_tr.cpu().numpy()
    X_te = X_te.cpu().numpy()
    y_tr = y_tr.view(-1).cpu().numpy()
    y_te = y_te.view(-1).cpu().numpy()

    # классификация или регрессия?
    is_classification = (np.issubdtype(y_tr.dtype, np.integer) or
                         (np.unique(y_tr).size <= 20 and y_tr.dtype.kind in "ifu"))

    if is_classification:
        # логистическая регрессия + стандартизация
        clf = make_pipeline(
        
            LogisticRegression(max_iter=500, n_jobs=None)
        )
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        report = {
            "acc": float(accuracy_score(y_te, y_pred)),
            "f1_macro": float(f1_score(y_te, y_pred, average="macro")),
        }
        # AUC если бинарная
        if np.unique(y_te).size == 2:
            try:
                proba = clf.predict_proba(X_te)[:, 1]
                report["roc_auc"] = float(roc_auc_score(y_te, proba))
            except Exception:
                pass
        return report
    else:
        # регрессия
        reg = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LinearRegression()
        )
        reg.fit(X_tr, y_tr)
        y_hat = reg.predict(X_te)
        report = {
            "mse": float(mean_squared_error(y_te, y_hat)),
            "r2": float(r2_score(y_te, y_hat)),
        }
        return report

# =========================
# main: учим линейку на data/, тестим на Test/
# =========================
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 1) грузим обученную JEPA-модель (энкодер), чекпоинт как раньше
    model = get_last_model(ckpt_root=".", cfg=cfg)

    # 2) строим два набора патчей: train=data/, test=Test/
    train_set = build_patches_set(root_path="data", cfg=cfg)
    test_set  = build_patches_set(root_path="Test", cfg=cfg)

    # 3) считаем эмбеддинги
    X_tr, y_tr = get_embeddings_ctx_targets(model, train_set, cfg)
    X_te, y_te = get_embeddings_ctx_targets(model, test_set, cfg)

    if y_tr is None or y_te is None:
        raise RuntimeError("В датасетах не найдено graph-level 'y'. Нужны метки для линейной оценки.")

    print(f"[embeddings] train: X={tuple(X_tr.shape)}, y={tuple(y_tr.shape)}; "
          f"test: X={tuple(X_te.shape)}, y={tuple(y_te.shape)}")

    # 4) обучаем и оцениваем линейную модель
    lin_report = fit_and_eval_linear(X_tr, y_tr, X_te, y_te)
    print("=== Linear head report (train=data/, test=Test/) ===")
    for k, v in lin_report.items():
        print(f"{k:10s}: {v:.4f}")

    
    # 5) UMAP на тесте (для визуального контроля)
    plot_umap_2d(X_te, y_te, title="JEPA embeddings (Test) — UMAP 2D")

if __name__ == "__main__":
    main()
