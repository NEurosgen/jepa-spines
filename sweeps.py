
from itertools import product
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, List, Iterable, Tuple, Optional
import json, random
import numpy as np
from omegaconf import OmegaConf, DictConfig


from report_options import _flatten_trial_record
def _assert_override_keys_exist(cfg: DictConfig, param_grid: Dict[str, Iterable[Any]]):
    """
    Проверяем, что все ключи из param_grid реально есть в cfg.
    """
    for key in param_grid.keys():
        parts = key.split(".")
        node = cfg
        for p in parts[:-1]:
            assert p in node, f"Invalid override '{key}': section '{p}' not found in config"
            node = node[p]
        last = parts[-1]
        assert last in node, f"Invalid override '{key}': key '{last}' not found in config section {'.'.join(parts[:-1])}"

def _param_grid_iter(grid: Dict[str, Iterable[Any]]) -> Iterable[Tuple[Dict[str, Any], str]]:
    """
    Принимает сетку вида {'train.lr':[...], 'model.hidden_size':[...]}
    Возвращает пары (params_dict, short_tag).
    """
    keys = list(grid.keys())
    values = [list(grid[k]) for k in keys]
    for combo in product(*values):
        params = dict(zip(keys, combo))
        # короткий тег для имени папки
        tag = "_".join([f"{k.replace('.','-')}={v}" for k, v in params.items()])
        yield params, tag

def _param_grid_random_samples(grid: Dict[str, Iterable[Any]], n_samples: int, seed: int = 42):
    """
    Случайный сэмпл комбинаций из полной сетки (без повторов).
    """
    all_combos = list(_param_grid_iter(grid))
    rng = random.Random(seed)
    if n_samples >= len(all_combos):
        return all_combos
    return rng.sample(all_combos, n_samples)













def sweep_experiments(
    base_cfg: DictConfig,
    param_grid: Dict[str, Iterable[Any]],
    run_single_trial ,
    seeds: Optional[List[int]] = None,
    out_dir: str = None,
    sweep_name: str = "sweep",
    random_samples: Optional[int] = None,  
    random_seed: int = 123,
    rank_metric: Optional[str] = None,      
    maximize: bool = True,    
    result_file = 'result'             
) -> Dict[str, Any]:
    """
    Запускает серию трейлов (grid или random search) и собирает summary.json.
    - base_cfg: базовый OmegaConf cfg
    - param_grid: словарь { "train.lr": [1e-3, 5e-4], "model.hidden_size":[128,256], ... }
    - seeds: список сидов; если None — возьмём [base_cfg.seed + i for i in range(base_cfg.train.runs)]
    - out_dir: базовая папка вывода (по умолчанию runs/{dataset}_jepa/sweeps/{sweep_name})
    - random_samples: если задано — случайно выбираем N комбинаций из сетки
    - rank_metric: ключ из linear_agg, по которому выбрать лучший конфиг (например 'acc_mean')
    - maximize: если True — чем больше, тем лучше (accuracy/AUC); если False — меньше лучше (loss)
    """
    _assert_override_keys_exist(base_cfg, param_grid)
    if seeds is None:
        seeds = [int(base_cfg.seed) + i for i in range(int(base_cfg.train.runs))]

    if out_dir is None:
        out_dir = f"runs/{base_cfg.dataset}_jepa/sweeps/{sweep_name}"


    base = Path(out_dir or f"runs/{base_cfg.dataset}_jepa/sweeps/{sweep_name}")
    base.mkdir(parents=True, exist_ok=True)
    results_jsonl = base /  (result_file + ".jsonl")

    (base / "_base_config.yaml").write_text(OmegaConf.to_yaml(base_cfg))
    (base / "_param_grid.json").write_text(json.dumps({k:list(v) for k,v in param_grid.items()}, indent=2))

    
    if random_samples is not None:
        trials = _param_grid_random_samples(param_grid, random_samples, seed=random_seed)
    else:
        trials = list(_param_grid_iter(param_grid))

    all_results: List[Dict[str, Any]] = []


   
    for idx, (params, tag) in enumerate(trials, start=1):
        trial_dir = base / f"trial_{idx:03d}__{tag}"
        print(f"\n=== Trial {idx}/{len(trials)} ===\nParams: {params}\nDir: {trial_dir}")

        out = run_single_trial(
            base_cfg=base_cfg,
            params=params,
            trial_dir=trial_dir,
            seeds=seeds,
            tag=tag,
            out_jsonl=results_jsonl,     
        )
        all_results.append(out)

    # агрегация и ранжирование
    summary = {
        "sweep_name": sweep_name,
        "seeds": seeds,
        "n_trials": len(all_results),
        "results": all_results,
        "rank_metric": rank_metric,
        "maximize": maximize,
    }

    # выбрать лучший конфиг по rank_metric, если доступна
    best = None
    if rank_metric is not None:
        pairs = []
        for r in all_results:
            val = r.get("linear_agg", {}).get(rank_metric, None)
            if isinstance(val, (int, float)):
                pairs.append((val, r))
        if pairs:
            if maximize:
                best = max(pairs, key=lambda x: x[0])[1]
            else:
                best = min(pairs, key=lambda x: x[0])[1]
    summary["best"] = best

    (base / "summary.json").write_text(json.dumps(summary, indent=2))

    # красивый вывод топа
    if best is not None:
        print("\n=== BEST TRIAL ===")
        print("tag:", best["tag"])
        print("params:", best["params"])
        print(f"{rank_metric}: {best['linear_agg'].get(rank_metric)}")
    else:
        print("\n(no rank_metric provided or not found in results — summary saved)")

    return summary






# # запускаем GRID:
# sweep_experiments(
#     base_cfg=cfg,
#     param_grid=param_grid,
#     seeds=seeds,
#     sweep_name="lr_hidden_dist_grid",
#     rank_metric="acc_mean",     # или "roc_auc_mean" / "f1_macro_mean" / что у тебя от linear_agg
#     maximize=True,              # для accuracy/AUC/F1 — True
# )

# # или RANDOM SEARCH на 6 конфигов:
# sweep_experiments(
#     base_cfg=cfg,
#     param_grid=param_grid,
#     seeds=seeds,
#     sweep_name="random6",
#     random_samples=6,
#     random_seed=123,
#     rank_metric="roc_auc_mean",
#     maximize=True,
# )
