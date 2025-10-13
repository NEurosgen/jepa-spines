"""Loss helpers used by the JEPA training loops."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from model_utils.hyperbolic_dist import hyperbolic_dist


def resolve_loss(criterion_type: int) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return the appropriate loss callable based on configuration."""
    if criterion_type == 0:
        return torch.nn.SmoothL1Loss(beta=0.5)
    if criterion_type == 1:
        return F.mse_loss
    if criterion_type == 2:
        return hyperbolic_dist
    raise ValueError(f"Unsupported criterion type: {criterion_type}")
