# normalize_nodes.py
import math
import torch
from torch_geometric.data import Dataset, Data

class NodeFeatureNormalizer:
    """
    Нормализация узловых фич по формуле (x - mean) / (std + eps).
    Хранит mean/std и умеет применять к объектам Data.
    """
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, attr: str = "x", eps: float = 1e-8):
        assert mean.dim() == 1 and std.dim() == 1, "mean/std должны быть shape=[F]"
        self.mean = mean.detach().clone()
        self.std = std.detach().clone()
        self.attr = attr
        self.eps = eps

        # защитимся от нулевой дисперсии по признакам
        self.std = torch.where(self.std < eps, torch.ones_like(self.std), self.std)

    @torch.no_grad()
    def __call__(self, data: Data) -> Data:
        if hasattr(data, self.attr) and getattr(data, self.attr) is not None:
            x = getattr(data, self.attr)
            # ожидаем x: [N, F]
            if x.dim() != 2 or x.size(-1) != self.mean.numel():
                raise ValueError(f"{self.attr} должен иметь форму [N, {self.mean.numel()}], а не {tuple(x.shape)}")
            x = (x - self.mean.to(x.device)) / (self.std.to(x.device) + self.eps)
            setattr(data, self.attr, x)
        return data

    def invert(self, data: Data) -> Data:
        """Обратно вернуть из нормализованных величин (редко нужно, но иногда удобно для отладки/визуализации)."""
        if hasattr(data, self.attr) and getattr(data, self.attr) is not None:
            x = getattr(data, self.attr)
            x = x * (self.std.to(x.device) + self.eps) + self.mean.to(x.device)
            setattr(data, self.attr, x)
        return data

    def state_dict(self):
        return {"mean": self.mean, "std": self.std, "attr": self.attr, "eps": self.eps}

    @staticmethod
    def from_state_dict(state):
        return NodeFeatureNormalizer(state["mean"], state["std"], state.get("attr", "x"), state.get("eps", 1e-8))


@torch.no_grad()
def compute_node_feature_stats(dataset: Dataset, attr: str = "x", device: str = "cpu"):
    """
    Считает mean/std по признакам узлов поверх ВЕГО датасета (все графы, все узлы).
    Использует одно-проходный Welford для численной устойчивости.
    Возвращает (mean[F], std[F]).
    """
    count = 0
    mean = None  # shape [F]
    M2 = None    # накопитель суммы квадратов отклонений

    for i in range(len(dataset)):
        data = dataset.get(i)
        x = getattr(data, attr, None)
        if x is None:
            continue
        if not torch.is_floating_point(x):
            x = x.float()

        x = x.to(device)  # можно оставить cpu, если не влазит на gpu
        if mean is None:
            mean = torch.zeros(x.size(1), device=device)
            M2 = torch.zeros_like(mean)

        # обновляем по батчу узлов текущего графа
        # сначала считаем среднее по узлам текущего графа
        batch_n = x.size(0)
        if batch_n == 0:
            continue
        batch_mean = x.mean(dim=0)

        # Welford: обновление глобального среднего/дисперсии
        count_new = count + batch_n
        delta = batch_mean - mean
        mean = mean + delta * (batch_n / max(count_new, 1))

        # сумма квадратов отклонений на батче:
        # Var_total = Var_old + Var_batch + delta^2 * (n_old * n_batch / n_new)
        # Var_batch = sum((x - batch_mean)^2) /? — для M2 нам нужна суммы без деления:
        batch_M2 = ((x - batch_mean).pow(2)).sum(dim=0)
        M2 = M2 + batch_M2 + (delta.pow(2)) * (count * batch_n / max(count_new, 1))
        count = count_new

    if count <= 1:
        raise ValueError("Недостаточно узлов, чтобы посчитать статистику.")

    var = M2 / (count - 1)  # несмещённая оценка
    std = torch.sqrt(torch.clamp(var, min=0.0))
    return mean.cpu(), std.cpu()


@torch.no_grad()
def normalize_dataset_inplace(dataset: Dataset, normalizer: NodeFeatureNormalizer):
    """
    Применяет нормализацию к каждому графу датасета (in-place).
    """
    for i in range(len(dataset)):
        data = dataset.get(i)
        dataset[i] = normalizer(data)  # PyG Dataset поддерживает присваивание


# ---------- пример использования ----------


