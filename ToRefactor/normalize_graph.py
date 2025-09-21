from typing import Iterable, Literal, Optional, Tuple, Dict, Any, Union
import math
import torch
from torch import Tensor
from torch_geometric.data import Data, InMemoryDataset

NormMethod = Literal["zscore", "minmax", "robust", "l2"]

@torch.no_grad()
def _stack_all_features(
    dataset: Iterable[Data],
    attr: str = "x",
) -> Tensor:
    xs = []
    for data in dataset:
        if not hasattr(data, attr):
            continue
        x = getattr(data, attr)
        if x is None or x.numel() == 0:
            continue
        xs.append(x.detach().float().reshape(-1, x.size(-1)))
    return torch.cat(xs, dim=0) if xs else torch.empty(0)

def _finite_mask(t: Tensor) -> Tensor:
    return torch.isfinite(t).all(dim=-1) if t.ndim == 2 else torch.isfinite(t)

@torch.no_grad()
def compute_feature_stats(
    dataset: Iterable[Data],
    method: NormMethod = "zscore",
    attr: str = "x",
    eps: float = 1e-8,
) -> Dict[str, Tensor]:
    """
    Считает по-столбцово статистику на всем датасете для выбранного метода.
    Возвращает словарь параметров нормализации.
    """
    X = _stack_all_features(dataset, attr=attr)
    if X.numel() == 0:
        return {}

    mask = _finite_mask(X)
    X = X[mask]

    if X.numel() == 0:
        return {}

    if method == "zscore":
        mean = X.mean(dim=0)
        std = X.std(dim=0, unbiased=False).clamp_min(eps)
        return {"mean": mean, "std": std}
    elif method == "minmax":
        xmin = X.min(dim=0).values
        xmax = X.max(dim=0).values
        scale = (xmax - xmin).clamp_min(eps)
        return {"min": xmin, "scale": scale}
    elif method == "robust":
        q1 = X.quantile(0.25, dim=0)
        q3 = X.quantile(0.75, dim=0)
        med = X.median(dim=0).values
        iqr = (q3 - q1).clamp_min(eps)
        return {"median": med, "iqr": iqr}
    elif method == "l2":
        # для l2 глобальные параметры не нужны
        return {}
    else:
        raise ValueError(f"Unknown method: {method}")

@torch.no_grad()
def _apply_norm(
    x: Tensor,
    method: NormMethod,
    params: Dict[str, Tensor],
    eps: float,
    per_graph: bool,
    clip: Optional[Tuple[float, float]],
) -> Tensor:
    x = x.float()
    mask = torch.isfinite(x)
    # временно заменим нечисла на нули, вернем после
    x_safe = x.clone()
    x_safe[~mask] = 0.0

    if method == "zscore":
        mean = params["mean"]
        std = params["std"]
        out = (x_safe - mean) / std
    elif method == "minmax":
        xmin = params["min"]
        scale = params["scale"]
        out = (x_safe - xmin) / scale
    elif method == "robust":
        med = params["median"]
        iqr = params["iqr"]
        out = (x_safe - med) / iqr
    elif method == "l2":
        # по-строчно: ||x_i||_2 = 1 (если строка не нулевая)
        norms = x_safe.norm(dim=-1, keepdim=True).clamp_min(eps)
        out = x_safe / norms
    else:
        raise ValueError(method)

    # Вернём NaN/Inf туда, где исходно были нечисла
    out[~mask] = float("nan")

    if clip is not None:
        lo, hi = clip
        out = out.clamp(min=lo, max=hi)

    return out.to(x.dtype)

@torch.no_grad()
def normalize_dataset_node_features(
    dataset: Union[InMemoryDataset, Iterable[Data]],
    method: NormMethod = "zscore",
    *,
    attr: str = "x",
    per_graph: bool = False,
    eps: float = 1e-8,
    clip: Optional[Tuple[float, float]] = None,
    inplace: bool = True,
    return_stats: bool = False,
) -> Union[None, Tuple[Union[InMemoryDataset, Iterable[Data]], Dict[str, Any]]]:
    """
    Нормализует признаки узлов в датасете PyG.

    Параметры:
      - dataset: InMemoryDataset или любой итерируемый набор Data
      - method: "zscore" | "minmax" | "robust" | "l2"
      - attr: имя атрибута с признаками (по умолчанию "x")
      - per_graph: если True — считать статистику отдельно для каждого графа
      - eps: числ. стабилизация
      - clip: (low, high) — опциональное обрезание
      - inplace: менять объекты на месте или вернуть копии
      - return_stats: вернуть использованные параметры нормировки

    Возвращает:
      - если not inplace: (нормализованный_набор, stats_or_list)
      - если inplace: None (и опционально stats)
    """
    # Приведём к простому списку для одноразового прохода при not inplace
    is_inmemory = isinstance(dataset, InMemoryDataset)
    data_iter = dataset if not is_inmemory else list(dataset)

    # Глобальные параметры (если нужны)
    global_params = None if per_graph else compute_feature_stats(
        data_iter, method=method, attr=attr, eps=eps
    )

    if not inplace:
        new_list = []

    collected_stats = [] if (per_graph and return_stats) else None

    for data in (dataset if not is_inmemory else data_iter):
        if not hasattr(data, attr) or getattr(data, attr) is None:
            if not inplace:
                new_list.append(data)
            continue

        x: Tensor = getattr(data, attr)
        if x.numel() == 0:
            if not inplace:
                new_list.append(data)
            continue

        if per_graph:
            params = compute_feature_stats([data], method=method, attr=attr, eps=eps)
            if collected_stats is not None:
                collected_stats.append(params)
        else:
            params = global_params or {}

        x_norm = _apply_norm(x, method=method, params=params, eps=eps, per_graph=per_graph, clip=clip)

        if inplace:
            setattr(data, attr, x_norm)
        else:
            d = data.clone()
            setattr(d, attr, x_norm)
            new_list.append(d)

    if inplace:
        if return_stats:
            return None if not per_graph else (None, collected_stats)  # тип-совместимость
        return None

    # Если dataset был InMemoryDataset, вернём просто список Data
    stats_out: Any
    if return_stats:
        stats_out = collected_stats if per_graph else global_params
        return new_list, stats_out
    else:
        return new_list, {}

# -------- Пример использования --------
# 1) По всему датасету (z-score), меняем на месте:
# normalize_dataset_node_features(train_dataset, method="zscore", inplace=True)

# 2) Пер-графово (min-max), получить новый список + статистики на каждый граф:
# new_graphs, per_graph_stats = normalize_dataset_node_features(
#     train_dataset, method="minmax", per_graph=True, inplace=False, return_stats=True
# )

# 3) Робастная нормализация с обрезанием экстремумов:
# normalize_dataset_node_features(train_dataset, method="robust", clip=(-5, 5))

# 4) L2-нормализация строк (каждый узел -> единичная норма), обычно как последний шаг:
# normalize_dataset_node_features(train_dataset, method="l2", inplace=True)


from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data

class FeatureNormTransform(BaseTransform):
    def __init__(self, params, method="zscore", attr="x", eps=1e-8, clip=None):
        self.params, self.method, self.attr, self.eps, self.clip = params, method, attr, eps, clip

    def __call__(self, data: Data) -> Data:
        if hasattr(data, self.attr) and getattr(data, self.attr) is not None:
            x = getattr(data, self.attr)
            x_norm = _apply_norm(x, method=self.method, params=self.params,
                                 eps=self.eps, per_graph=False, clip=self.clip)
            setattr(data, self.attr, x_norm)
        return data