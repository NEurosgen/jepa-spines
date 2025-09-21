
from itertools import product
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, List, Iterable, Tuple, Optional
import json, random
import numpy as np
from omegaconf import OmegaConf, DictConfig




def _set_by_dot(cfg: DictConfig, key: str, value: Any):
    """
    Устанавливает cfg['a']['b']['c'] по ключу 'a.b.c' в OmegaConf.
    """
    parts = key.split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node:
            node[p] = {}
        node = node[p]
    node[parts[-1]] = value



def _clone_cfg(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def _flatten_trial_record(trial_out: dict):
    rec = {}

    for k, v in trial_out.get("params", {}).items():
        rec[k] = v

    rec["tag"] = trial_out.get("tag", "")

    for k, v in (trial_out.get("linear_agg") or {}).items():
        rec[k] = v
    return rec



def _load_results_from_dir(dirpath):
    import json
    p = Path(dirpath) / "linear_eval_agg.json"
    if p.exists():
        return json.loads(p.read_text())
    return {"linear_agg": {}, "linear": {}}




from pathlib import Path
from typing import Dict, Any, Optional
import json

def _ensure_native(v):

    try:
        import numpy as np
        if isinstance(v, (np.generic,)):
            return v.item()
    except Exception:
        pass
    try:
        import torch
        if isinstance(v, (torch.Tensor,)) and v.ndim == 0:
            return v.item()
    except Exception:
        pass
    return v

def _extract_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Берём метрики из results["linear_agg"] (acc_mean, f1_macro_mean, roc_auc_mean),
    и маппим их на короткие ключи: acc/f1_macro/roc_auc.
    Если linear_agg отсутствует — пробуем взять из первого сида results["linear"].
    Если и этого нет — пустой dict.
    """
    metrics = {}
    la = (results or {}).get("linear_agg") or {}
    # приоритет: *_mean
    mapping = {
        "acc": ("acc_mean", "acc"),
        "f1_macro": ("f1_macro_mean", "f1_macro"),
        "roc_auc": ("roc_auc_mean", "roc_auc"),
    }
    for short, keys in mapping.items():
        val = None
        for k in keys:
            if k in la:
                val = la[k]
                break
   
        if val is None:
            by_seed = (results or {}).get("linear") or {}
            if by_seed:
                first_seed = sorted(by_seed.keys())[0]
                val = by_seed[first_seed].get(short)
        if val is not None:
            metrics[short] = float(_ensure_native(val))
    return metrics

def _next_exp_id(jsonl_path: Path) -> int:
    if not jsonl_path.exists():
        return 0

    with jsonl_path.open("r") as f:
        return sum(1 for _ in f)

def write_jsonl_record(
    jsonl_path: Path,
    overrides: Dict[str, Any],
    results: Dict[str, Any],
    hparams_min: Optional[Dict[str, Any]] = None,
    notes: Optional[Dict[str, Any]] = None,
    exp_id: Optional[int] = None,
) -> int:
    """
    Записывает одну строку JSON:
      {"exp_id": N, "overrides": {...}, "metrics": {...}, "hparams": {...}, "notes": {...}}
    Возвращает exp_id, под которым записали.
    """
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    if exp_id is None:
        exp_id = _next_exp_id(jsonl_path)

    record = {
        "exp_id": exp_id,
        "overrides": overrides or {},
        "metrics": _extract_metrics(results),
    }
    if hparams_min:

        record["hparams"] = {k: _ensure_native(v) for k, v in hparams_min.items()}
    if notes:
        record["notes"] = {k: _ensure_native(v) for k, v in notes.items()}

    with jsonl_path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return exp_id

