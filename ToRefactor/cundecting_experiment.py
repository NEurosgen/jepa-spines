import os, json, csv, random, itertools
from pathlib import Path
from typing import List, Dict, Any

import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor
)
from torch.utils.data import random_split

from torch_geometric.data import Data, Batch

# ---- твои модули
from model.JEPA import JEPAModel
from ExperimentalDataset import SpineGraphDataset,WightlessDS
from GraphCut.graphcutter import (
    GraphCutter, KHop, BFSBudget,
    NodeBudget, EnsureConnected, Unique,
    InduceEdges, Relabel
)
from graph_cut_datset import CutPatchesDataset, make_loader
from normalize_graph import _apply_norm, compute_feature_stats
from GetEmbedingInformation import get_embeddings_ctx_targets
from rep_eval import evaluate_representations


# -------------------- utils --------------------
def set_seed(seed: int, deterministic_torch: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def flatten_cfg(cfg: DictConfig) -> Dict[str, Any]:
    """В плоский dict для логирования гиперпараметров."""
    return OmegaConf.to_container(cfg, resolve=True)


def save_jsonl_line(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def append_metrics_csv(path: Path, header: List[str], row: List[Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(header)
        w.writerow(row)


# -------------------- graph cutters --------------------
def get_strategy(mode: str):
    mode = (mode or "").lower()
    if mode == "bfs":
        return BFSBudget
    if mode == "khop":
        return KHop
    raise ValueError(f"Unknown strategy mode: {mode}")


def build_cutters(cfg: DictConfig,seed = None):
    if seed is None:
        seed = cfg.seed        

    # контекст — по желанию можно добавить стратегию из конфига
    ctx_strats = []
    if "ctx_mode" in cfg.cutter.strategy and cfg.cutter.strategy.ctx_mode is not None:
        ctx_cls = get_strategy(cfg.cutter.strategy.ctx_mode)
        ctx_strats.append(ctx_cls(cfg.cutter.strategy.ctx_param))

    ctx_cutter = GraphCutter(
        strategies=ctx_strats,
        constraints=[EnsureConnected(), Unique()],
        post=[InduceEdges(), Relabel()],
        rng_seed=seed
    )

    tgt_cls = get_strategy(cfg.cutter.strategy.tgt_mode)
    tgt_cutter = GraphCutter(
        strategies=[tgt_cls(cfg.cutter.strategy.tgt_param)],
        constraints=[EnsureConnected(), Unique()],
        post=[InduceEdges(), Relabel()],
        rng_seed=seed
    )
    return ctx_cutter, tgt_cutter


# -------------------- data --------------------
def apply_norm_inplace(dataset, params, method="zscore", attr="x", eps=1e-8, clip=None):
    for data in dataset:
        if hasattr(data, attr) and getattr(data, attr) is not None:
            x = getattr(data, attr)
            x_norm = _apply_norm(x, method=method, params=params,
                                 eps=eps, per_graph=False, clip=clip)
            setattr(data, attr, x_norm)


def get_datasets(cfg: DictConfig,seed = None):
    if seed is None:
        seed = cfg.seed
    
    base_ds = WightlessDS(cfg.data.root)

    n_total = len(base_ds)
    n_val = int(cfg.data.val_ratio * n_total)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(base_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(seed))
    train_syn = SpineGraphDataset(
        ds=train_ds,
        knn=cfg.data.knn,
        count_per_item=cfg.data.count_per_item,
        mode=cfg.data.mode,
        gamma=cfg.data.gamma
    )
    val_syn = SpineGraphDataset(
        ds=val_ds,
        knn=cfg.data.knn,
        count_per_item=0,
        mode=cfg.data.mode,
        gamma=cfg.data.gamma
    )
    # нормализация по train
    method = cfg.data.norm.method
    params = compute_feature_stats(train_syn, method=method, attr=cfg.data.norm.attr)
    apply_norm_inplace(train_syn, params, method=method, attr=cfg.data.norm.attr)
    apply_norm_inplace(val_syn,   params, method=method, attr=cfg.data.norm.attr)

    # патчи
    ctx_cutter, tgt_cutter = build_cutters(cfg)
    train_patch_ds = CutPatchesDataset(
        base_graphs=train_syn,
        context_cutter=ctx_cutter,
        targets_cutter=tgt_cutter,
        rwse_k=cfg.model.pe_dim,
        max_targets=cfg.data.max_patches
    )
    val_patch_ds = CutPatchesDataset(
        base_graphs=val_syn,
        context_cutter=ctx_cutter,
        targets_cutter=tgt_cutter,
        rwse_k=cfg.model.pe_dim,
        max_targets=cfg.data.max_patches
    )
    return train_patch_ds, val_patch_ds

from torch import Tensor
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

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, n_jobs=None)
    )
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    y_pred = clf.predict(X_te)
    report = {
        "acc": float(accuracy_score(y_te, y_pred)),
        "f1_macro": float(f1_score(y_te, y_pred, average="macro")),
        "roc_auc": float(roc_auc_score(y_te, proba))
    }
    
        
       
    return report
    
    

def evaluate_val(model: torch.nn.Module, val_ds, train_ds,cfg: DictConfig) -> Dict[str, float]:
    model.eval()
    torch.set_grad_enabled(False)
    X_tr, y_tr = get_embeddings_ctx_targets(model, train_ds, cfg)
    X_te, y_te = get_embeddings_ctx_targets(model, val_ds, cfg)
    lin_report = fit_and_eval_linear(X_tr, y_tr, X_te, y_te)
    return lin_report


# -------------------- one experiment run --------------------
def run_experiment(cfg: DictConfig, workdir: Path,seed = 42) -> Dict[str, Any]:
    set_seed(seed)
    pl.seed_everything(seed, workers=True)

    train_ds, val_ds = get_datasets(cfg,seed)
    train_loader = make_loader(
        train_ds,
        batch_size=cfg.trainer.batch_size,
        shuffle=True,
        num_workers=cfg.trainer.num_workers,
        pin_memory=cfg.trainer.pin_memory,
    )
    val_loader = make_loader(
        val_ds,
        batch_size=cfg.trainer.batch_size,
        shuffle=False,
        num_workers=cfg.trainer.num_workers,
        pin_memory=cfg.trainer.pin_memory,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = JEPAModel(cfg).to(device)

    ckpt_dir = workdir / "checkpoints"
    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            monitor=cfg.trainer.monitor_metric,
            mode=cfg.trainer.monitor_mode,
            save_top_k=1,
            save_last=True,
            filename="{epoch}-{"+cfg.trainer.monitor_metric+":.4f}",
        ),
    ]
    if cfg.trainer.early_stopping.enabled:
        callbacks.append(
            EarlyStopping(
                monitor=cfg.trainer.monitor_metric,
                mode=cfg.trainer.monitor_mode,
                patience=cfg.trainer.early_stopping.patience,
                min_delta=cfg.trainer.early_stopping.min_delta,
            )
        )

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
       # precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        enable_checkpointing=True,
        default_root_dir=str(workdir),
        #callbacks=callbacks,
        detect_anomaly=cfg.trainer.detect_anomaly,
    )

    model.eval(); torch.set_grad_enabled(False)
    X0,_ = get_embeddings_ctx_targets(model, val_ds, cfg)

    trainer.fit(model, train_loader, val_loader)

    model.eval(); torch.set_grad_enabled(False)
    X1,_ = get_embeddings_ctx_targets(model, val_ds, cfg)


    print("[DEBUG] mean |Δembedding| =", (X1 - X0).abs().mean().item())
    # оценка на валидейшне (репрезентации)
    metrics = evaluate_val(model, val_ds, train_ds,cfg)

    # собираем всё для логов
    result = {
        "metrics": metrics,
        "hparams": flatten_cfg(cfg),
        "best_ckpt": callbacks[1].best_model_path if isinstance(callbacks[1], ModelCheckpoint) else "",
    }
    return result




import math
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

# --- хелпер: обычный грид в список комбинаций ---
def grid_dict_product(grid: dict):
    import itertools
    if not grid:
        return [{}]
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    runs = []
    for combo in itertools.product(*vals):
        runs.append({k: v for k, v in zip(keys, combo)})
    return runs

def resolve_expr_fields(runs):
    for combo in runs:
        for k, v in list(combo.items()):
            if isinstance(v, str) and v.strip().startswith("@expr:"):
                expr = v.split(":", 1)[1].strip()
                # безопасное окружение: только math и текущие значения гиперов
                env = {"math": math}
                # ключи с точками -> идентификаторы через __
                env.update({kk.replace(".", "__"): vv for kk, vv in combo.items()})
                val = eval(expr.replace(".", "__"), {"__builtins__": {}}, env)
                combo[k] = int(val) if isinstance(val, (int, float)) else val

import copy
import math

# Регистр готовых функций по имени
def f_linear2x(h: int) -> int:  # пример 1
    return  2*h

def f_shift6(h: int) -> int:    # пример 2
    return h + 6
def f_shift4(h: int) -> int:    # пример 2
    return h + 4

def f_shift2(h: int) -> int:    # пример 2
    return h + 2


def f_pow2(h: int) -> int:      # пример 3 — ближайшая степень двойки не меньше 8 от цели 2*h
    target = int(h**2)
    return target
def f_xsqrt(h: int)->int:
    target = h* int(h**0.5)
    return target
FUNC_REGISTRY = {
    "linear2x": f_linear2x,
    "shift+6": f_shift6,
    "shift+4": f_shift4,
    "shift+2": f_shift2,
    "sqrtx":f_xsqrt,
    "pow2": f_pow2,
}

def resolve_expr(expr: str, combo: dict):
    """Поддержка строк-формул вида '@expr: max(8, 2*model.predictor_hidden_dim)'."""
    expr = expr.split(":", 1)[1].strip()
    env = {"math": math}
    env.update({k.replace(".", "__"): v for k, v in combo.items()})
    return eval(expr.replace(".", "__"), {"__builtins__": {}}, env)


import numpy as np

def mean_by_seed_records(seeds, cfg, workdir):
    records = []

    for seed in seeds:
        result = run_experiment(cfg, workdir=workdir, seed=seed)
        records.append(result['metrics'])

    # Список словарей -> словарь массивов
    metrics = {k: [rec[k] for rec in records] for k in records[0].keys()}

    # Считаем среднее и std
    stats = {}
    for k, vals in metrics.items():
        arr = np.array(vals)
        stats[k + "_mean"] = float(arr.mean())
        stats[k + "_std"] = float(arr.std(ddof=1))  # выборочное std
    result['metrics'] = stats
    return result


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    outdir = Path(os.getcwd()) / Path('experiments')
    results_path = outdir / "results_2.jsonl"
    results_csv = outdir / "metrics_2.csv"

    grid = OmegaConf.to_container(cfg.experiments.grid, resolve=True) or {}
    base_runs = grid_dict_product(grid)


    runs = []
    for base in base_runs:
        
        combo = copy.deepcopy(base)
            # если пользователь не задал latent_dim явно/через @expr, — применяем правило   

         
        runs.append(combo)




    # заголовок CSV
    csv_header = None

    for i, overrides in enumerate(runs):
        exp_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        for k, v in overrides.items():
            if k.startswith("_"):  # служебные поля не в конфиг
                continue
            OmegaConf.update(exp_cfg, k, v, merge=True)

        exp_dir = outdir / f"exp_{i:03d}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Run {i+1}/{len(runs)} overrides={overrides} ===")
        result = mean_by_seed_records(cfg=exp_cfg,workdir=exp_dir,seeds = [2,13,24,33,23,43,12,23])
        #result = run_experiment(exp_cfg, workdir=exp_dir,seed=cfg.seed)
       
        record = {
            "exp_id": i,
            "overrides": {k:v for k,v in overrides.items() if not k.startswith("_")},
            "metrics": result["metrics"],
            "hparams": result["hparams"],
            "best_ckpt": result["best_ckpt"],
            "notes": {
                "latent_rule": overrides.get("_latent_rule_note", "n/a"),
                "predictor_hidden_dim": overrides.get("model.predictor_hidden_dim"),
                "latent_dim": overrides.get("model.latent_dim"),
            },
        }
        save_jsonl_line(results_path, record)

        metrics_flat = result["metrics"]
        if csv_header is None:
            csv_header = ["exp_id", "latent_rule", "predictor_hidden_dim", "latent_dim"] + list(metrics_flat.keys())
        csv_row = [
            i,
            record["notes"]["latent_rule"],
            record["notes"]["predictor_hidden_dim"],
            record["notes"]["latent_dim"],
        ] + [metrics_flat.get(k, "") for k in csv_header[4:]]
        append_metrics_csv(results_csv, csv_header, csv_row)

    print(f"\nSaved JSONL to: {results_path}")
    print(f"Saved CSV    to: {results_csv}")


if __name__ == "__main__":
    main()
