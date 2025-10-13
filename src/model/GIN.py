# src/models/gin_classifier.py
import torch
import hydra
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from pytorch_lightning import LightningModule
from torchmetrics.classification import BinaryAccuracy       
from torchmetrics.classification import (
    BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC
)

def activations(name_func:str):
    if name_func == 'relu':
        return nn.ReLU
    if name_func == 'silu':
        return nn.SiLU
    if name_func == 'tanh':
        return nn.Tanh
    if name_func == 'selu':
        return nn.SELU    

#--------0.Блок свертки-------
class GINBlock(nn.Module):
    def __init__(self,in_channels: int, hidden: int,p :float = 0,activation:callable = nn.ReLU,ResCon=True):
        super().__init__()
        self.ResCon = ResCon
        self.conv = GINConv(nn.Sequential(
            nn.Linear(in_channels,hidden),
            activation(),
            nn.Linear(hidden,hidden)
        ))
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(p=p)
        self.res = nn.Linear(in_channels,hidden)
    def forward(self,x,edge_index):
        out = self.conv(x,edge_index)
        out = self.norm(out)
        out = self.dropout(out)
        if self.ResCon:
            out = out + self.res(x)
        return out
    
class GIN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, layers: int = 2, dropout: float = 0.2,activation:callable = nn.GELU):
        super().__init__()
        self.convs = nn.ModuleList([
            GINBlock(in_dim if i == 0 else hidden_dim, hidden_dim,activation=activation,p=dropout,ResCon = i>0) for i in range(layers)
        ])
    def forward(self,data):
        x = data.x
        edge_index = data.edge_index
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        batch = getattr(data, "batch", None) 
        if batch is None:
            # одиночный граф (нет .batch): усредняем по узлам
            return x.mean(dim=0, keepdim=True)         # (1, hidden_dim)
        else:
            # батч из нескольких графов: пулим по индексу графа в .batch
            return global_mean_pool(x, batch)          # (B, hidden_dim)

