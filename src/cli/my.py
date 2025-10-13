import torch

from logger.log import config_logger
from model_utils.asam import ASAM
from torch_geometric.loader import DataLoader
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error

from data_utils.get_data import create_dataset
from omegaconf import OmegaConf
from model_utils.get_model import create_model
from itertools import product
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, List, Iterable, Tuple, Optional
import json, random
import numpy as np
from omegaconf import OmegaConf, DictConfig

from src.experiments.sweeps import sweep_experiments
from src.experiments.run_experiment import run_experiment,train,test
from estimate_representation.report_options import _clone_cfg,_set_by_dot


from pathlib import Path
from estimate_representation.report_options import write_jsonl_record


from pathlib import Path
import json

def _next_exp_id(jsonl_path: Path) -> int:
    if not jsonl_path.exists():
        return 0
    with jsonl_path.open("r") as f:
        return sum(1 for _ in f)

def run_single_trial(
    base_cfg,
    params: Dict[str, Any],
    trial_dir: Path,
    seeds,
    tag: str = "",
    out_jsonl: Path = None,
):

    cfg = _clone_cfg(base_cfg)
    for k, v in params.items():
        _set_by_dot(cfg, k, v)
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))


    results = run_experiment(
        cfg=cfg,
        create_dataset=create_dataset,
        create_model=create_model,
        train=train,
        test=test,
        seeds=seeds,
    )

    linear_agg  = (results or {}).get("linear_agg", {})   
    linear_seed = (results or {}).get("linear", {})


    trial_payload = {
        "tag": tag,
        "params": params,
        "seeds": seeds,
        "linear_agg": linear_agg,
        "linear_by_seed": linear_seed,
    }
    (trial_dir / "trial.json").write_text(json.dumps(trial_payload, indent=2))

   
    if out_jsonl is not None:
        exp_id = _next_exp_id(out_jsonl)
        # минимальные hparams — по желанию (оставлю seed первого прогона)
        hparams_min = {"seed": seeds[0]} if seeds else {}

        jsonl_record = {
            "exp_id": exp_id,
            "overrides": params,                
            "metrics": {k: float(v) for k, v in linear_agg.items()},  
            "hparams": hparams_min,
            # "notes": {...}  # при желании добавишь
        }
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_jsonl.open("a") as f:
            f.write(json.dumps(jsonl_record, ensure_ascii=False) + "\n")

    return trial_payload


from omegaconf import OmegaConf

cfg = OmegaConf.load("train/configs/zinc.yaml")

param_grid = {
   
         'model.nlayer_gnn':[1, 2,4],
        'model.nlayer_mlpmixer': [1,2,4],
        'model.hidden_size': [i for i in range(64,512,32)],


}
print(create_dataset(cfg))
seeds = [cfg.seed + i for i in range(5)]  






sweep_experiments(
    base_cfg=cfg,
    param_grid=param_grid,
    run_single_trial=run_single_trial,
    seeds=seeds,
    sweep_name="lr_hidden_dist_grid",
    rank_metric="acc_mean",     # или "roc_auc_mean" / "f1_macro_mean" / что у тебя от linear_agg
    maximize=True,              # для accuracy/AUC/F1 — True
    result_file='result_1'
)