"""Compatibility layer around the refactored training pipeline."""

from __future__ import annotations

from jepa_spines.training.loops import evaluate, train_epoch
from jepa_spines.training.runner import TrainingRunner
from jepa_spines.training.utils import count_parameters
from jepa_spines.utils.seed import set_seed

__all__ = ["run", "count_parameters", "set_seed", "train_epoch", "evaluate"]


def run(cfg, create_dataset, create_model, train=None, test=None, evaluator=None):
    runner = TrainingRunner(cfg, create_dataset=create_dataset, create_model=create_model)
    train_loop = train if train is not None else train_epoch
    eval_loop = test if test is not None else evaluate
    return runner.run(train_loop=train_loop, eval_loop=eval_loop, evaluator=evaluator)
