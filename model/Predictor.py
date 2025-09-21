import torch
import torch.nn as nn

class Predictor(nn.Module):
    def __init__(self, latent_dim: int, pe_dim: int,
                 hidden_dim: int, out_dim: int = 2,
                 num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        # отдельные проекции для контекста и подсказки
        self.proj_ctx = nn.Linear(latent_dim, hidden_dim)
        self.proj_pe  = nn.Linear(pe_dim,     hidden_dim)

        layers = []
        dim = 2 * hidden_dim  # concat[ctx_proj, pe_proj]
        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            dim = hidden_dim
        layers += [nn.Linear(dim, out_dim)]  # -> (cosh, sinh)
        self.net = nn.Sequential(*layers)

    def forward(self, context: torch.Tensor, hint_P: torch.Tensor):
        # context: (M, latent_dim), hint_P: (M, pe_dim)
        
        c = self.proj_ctx(context)   # (M, hidden_dim)
        p = self.proj_pe(hint_P)     # (M, hidden_dim)
        x = torch.cat([c, p], dim=-1)
        return self.net(x)           # (M, 2)
