import torch

from torch import nn


class GraphAugment(nn.Module):
    def __init__(self, edge_drop=0.1, node_drop=0.0, feat_noise_std=0.0, feat_mask_prob=0.0):
        super().__init__()
        self.edge_drop = float(edge_drop)
        self.node_drop = float(node_drop)
        self.feat_noise_std = float(feat_noise_std)
        self.feat_mask_prob = float(feat_mask_prob)

    @torch.no_grad()
    def forward(self, batch):
        # Клонируем, чтобы не портить исходник
        x = batch.x.clone()
        edge_index = batch.edge_index.clone()
        device = x.device

        # (1) Шум к фичам
        if self.feat_noise_std > 0:
            x.add_(torch.randn_like(x) * self.feat_noise_std)

        # (2) Маскирование фич
        if self.feat_mask_prob > 0:
            mask = torch.rand_like(x) < self.feat_mask_prob
            x = x.masked_fill(mask, 0.0)

        # (3) Дроп рёбер
        if self.edge_drop > 0 and edge_index.numel() > 0:
            E = edge_index.size(1)
            keep = torch.rand(E, device=device) > self.edge_drop
            edge_index = edge_index[:, keep]

        # (4) (Опц.) Дроп узлов — аккуратно, требует пересчёта batch/edge_index; по умолчанию выключен
        # Можно пропустить: node_drop = 0.0 по дефолту

        new = batch.__class__()
        for k, v in batch:
            setattr(new, k, v)
        new.x = x
        new.edge_index = edge_index
        return new
