"""Training and evaluation loops for JEPA models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Protocol

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data

from .losses import resolve_loss


class SupportsStep(Protocol):
    def zero_grad(self) -> None: ...
    def step(self) -> None: ...


@dataclass
class LoopResult:
    loss: float


def _flip_laplacian_signs(batch: Data) -> None:
    if not hasattr(batch, "lap_pos_enc"):
        return
    lap_pos_enc = batch.lap_pos_enc
    if lap_pos_enc is None:
        return
    sign_flip = torch.rand(lap_pos_enc.size(1), device=lap_pos_enc.device)
    sign_flip[sign_flip >= 0.5] = 1.0
    sign_flip[sign_flip < 0.5] = -1.0
    batch.lap_pos_enc = lap_pos_enc * sign_flip.unsqueeze(0)


def _update_target_encoder(model: nn.Module, momentum_weight: float) -> None:
    if not hasattr(model, "context_encoder") or not hasattr(model, "target_encoder"):
        return
    with torch.no_grad():
        for param_q, param_k in zip(model.context_encoder.parameters(), model.target_encoder.parameters()):
            param_k.data.mul_(momentum_weight).add_(
                (1.0 - momentum_weight) * param_q.detach().data
            )


def train_epoch(
    loader: Iterable[Data],
    model: nn.Module,
    optimizer: SupportsStep,
    device: torch.device,
    momentum_weight: float,
    *,
    loss_type: int,
    sharpness_optimizer: Optional[object] = None,
) -> LoopResult:
    criterion = resolve_loss(loss_type)
    step_losses: list[float] = []
    num_targets: list[int] = []

    model.train()
    for batch in loader:
        if getattr(model, "use_lap", False):
            _flip_laplacian_signs(batch)
        batch = batch.to(device)
        optimizer.zero_grad()
        target_x, target_y = model(batch)
        loss = criterion(target_x, target_y)
        loss.backward()
        if sharpness_optimizer is None:
            optimizer.step()
        else:
            sharpness_optimizer.ascent_step()
            sharpness_optimizer.descent_step()
        _update_target_encoder(model, momentum_weight)
        step_losses.append(float(loss.item()))
        num_targets.append(len(target_y))

    epoch_loss = float(np.average(step_losses, weights=num_targets))
    return LoopResult(loss=epoch_loss)


def evaluate(
    loader: Iterable[Data],
    model: nn.Module,
    device: torch.device,
    *,
    loss_type: int,
) -> LoopResult:
    criterion = resolve_loss(loss_type)
    step_losses: list[float] = []
    num_targets: list[int] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            target_x, target_y = model(batch)
            loss = criterion(target_x, target_y)
            step_losses.append(float(loss.item()))
            num_targets.append(len(target_y))

    epoch_loss = float(np.average(step_losses, weights=num_targets))
    return LoopResult(loss=epoch_loss)
