"""High level orchestration for running JEPA training experiments."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from asam import ASAM
from representation_metrtic import encode_repr, fit_and_eval_linear

from .loops import LoopResult, evaluate, train_epoch
from .utils import count_parameters
from ..utils.seed import set_seed


DatasetFactory = Callable[[object], Tuple[Iterable, Iterable, Iterable]]
ModelFactory = Callable[[object], torch.nn.Module]


@dataclass
class TrainingArtifacts:
    train_loss: float
    val_loss: float
    test_loss: float
    per_epoch_seconds: float
    total_hours: float


class TrainingRunner:
    """Execute one or multiple training runs according to a configuration."""

    def __init__(
        self,
        cfg: object,
        create_dataset: DatasetFactory,
        create_model: ModelFactory,
    ) -> None:
        self.cfg = cfg
        self._create_dataset = create_dataset
        self._create_model = create_model
        self.device = torch.device(cfg.device)

    def _build_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_dataset, val_dataset, test_dataset = self._create_dataset(self.cfg)
        loader_kwargs = dict(batch_size=self.cfg.train.batch_size, num_workers=self.cfg.num_workers)
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
        return train_loader, val_loader, test_loader

    def run(
        self,
        train_loop: Callable[..., LoopResult] = train_epoch,
        eval_loop: Callable[..., LoopResult] = evaluate,
        evaluator: Optional[object] = None,
    ) -> Sequence[TrainingArtifacts]:
        cfg = self.cfg
        if cfg.seed is not None:
            seeds = [cfg.seed]
            cfg.train.runs = 1
        else:
            default_seeds = [21, 42, 41, 95, 12, 35, 66, 85, 3, 1234]
            requested_runs = getattr(cfg.train, "runs", len(default_seeds))
            seeds = default_seeds[:requested_runs]
            cfg.train.runs = len(seeds)

        train_loader, val_loader, test_loader = self._build_loaders()
        results: list[TrainingArtifacts] = []

        for run_idx, seed in enumerate(seeds):
            set_seed(seed)
            model = self._create_model(cfg).to(self.device)
            print(f"\nNumber of parameters: {count_parameters(model)}")

            if cfg.train.optimizer == "ASAM":
                base_optimizer = torch.optim.SGD(
                    model.parameters(), lr=cfg.train.lr, momentum=0.9, weight_decay=cfg.train.wd
                )
                sharpness_optimizer: Optional[ASAM] = ASAM(base_optimizer, model, rho=0.5)
                optimizer = base_optimizer
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
                sharpness_optimizer = None

            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=cfg.train.lr_decay,
                patience=cfg.train.lr_patience,
                verbose=True,
            )

            start_outer = time.time()
            per_epoch_durations: list[float] = []
            momentum_values = self._momentum_schedule(len(train_loader) * cfg.train.epochs)

            for epoch in range(cfg.train.epochs):
                epoch_start = time.time()
                train_result = train_loop(
                    train_loader,
                    model,
                    optimizer,
                    self.device,
                    momentum_weight=next(momentum_values),
                    loss_type=cfg.jepa.dist,
                    sharpness_optimizer=sharpness_optimizer,
                )
                val_result = eval_loop(val_loader, model, self.device, loss_type=cfg.jepa.dist)
                test_result = eval_loop(test_loader, model, self.device, loss_type=cfg.jepa.dist)

                epoch_duration = time.time() - epoch_start
                per_epoch_durations.append(epoch_duration)
                print(
                    f"Epoch: {epoch:03d}, Train Loss: {train_result.loss:.4f}, "
                    f"Val: {val_result.loss:.4f}, Test: {test_result.loss:.4f} Seconds: {epoch_duration:.4f}"
                )

                scheduler.step(val_result.loss)
                if sharpness_optimizer is None and optimizer.param_groups[0]["lr"] < cfg.train.min_lr:
                    print("!! LR EQUAL TO MIN LR SET.")
                    break

            average_epoch_time = float(np.mean(per_epoch_durations)) if per_epoch_durations else 0.0
            total_time_hours = float((time.time() - start_outer) / 3600)

            if cfg.checkpoint:
                torch.save({"model_state_dict": model.state_dict()}, "checkpoints/checkpoint.ckpt")

            self._maybe_log_linear_evaluation(model, train_loader, val_loader)

            results.append(
                TrainingArtifacts(
                    train_loss=train_result.loss,
                    val_loss=val_result.loss,
                    test_loss=test_result.loss,
                    per_epoch_seconds=average_epoch_time,
                    total_hours=total_time_hours,
                )
            )

            print("\nRun: ", run_idx)
            print(f"Train Loss: {train_result.loss:.4f}")
            print(f"Convergence Time (Epochs): {epoch + 1}")
            print(f"AVG TIME PER EPOCH: {average_epoch_time:.4f} s")
            print(f"TOTAL TIME TAKEN: {total_time_hours:.4f} h")

        if cfg.train.runs > 1:
            self._summarize_runs(results)
        return results

    def _momentum_schedule(self, total_steps: int, start: float = 0.996, end: float = 1.0) -> Iterable[float]:
        total_steps = max(1, total_steps)
        for i in range(total_steps + 1):
            yield start + i * (end - start) / total_steps

    def _maybe_log_linear_evaluation(self, model: torch.nn.Module, train_loader: DataLoader, val_loader: DataLoader) -> None:
        X_train, y_train = encode_repr(loader=train_loader, model=model, device=self.device)
        X_val, y_val = encode_repr(loader=val_loader, model=model, device=self.device)
        X_train = torch.as_tensor(X_train, dtype=torch.float32)
        y_train = torch.as_tensor(y_train, dtype=torch.long)
        X_val = torch.as_tensor(X_val, dtype=torch.float32)
        y_val = torch.as_tensor(y_val, dtype=torch.long)
        fit_and_eval_linear(X_tr=X_train, y_tr=y_train, X_te=X_val, y_te=y_val)

    def _summarize_runs(self, results: Sequence[TrainingArtifacts]) -> None:
        train_losses = torch.tensor([r.train_loss for r in results])
        epoch_times = torch.tensor([r.per_epoch_seconds for r in results])
        total_times = torch.tensor([r.total_hours for r in results])
        print(
            f"\nFinal Train Loss: {train_losses.mean():.4f} ± {train_losses.std():.4f}"
            f"\nSeconds/epoch: {epoch_times.mean():.4f}"
            f"\nHours/total: {total_times.mean():.4f}"
        )
