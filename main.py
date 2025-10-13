"""Entry point for running JEPA experiments via Hydra."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from jepa_spines.data.dataset_factory import create_dataset
from jepa_spines.models.factory import create_model
from jepa_spines.training.loops import evaluate, train_epoch
from jepa_spines.training.runner import TrainingRunner


@hydra.main(version_base=None, config_path="train/configs", config_name="zinc")
def main(cfg: DictConfig) -> None:
    runner = TrainingRunner(cfg, create_dataset=create_dataset, create_model=create_model)
    runner.run(train_loop=train_epoch, eval_loop=evaluate)


if __name__ == "__main__":
    main()
