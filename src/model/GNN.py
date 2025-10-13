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
