"""Model factory helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from logger.logging_utils import setup_logger
from model.GraphJepa import GraphJepa


@dataclass(frozen=True)
class ModelSpec:
    node_features: int
    edge_features: int
    node_type: str
    edge_type: str
    output_dim: int


_MODEL_SPECS: Dict[str, ModelSpec] = {
    "ZINC": ModelSpec(28, 4, "Discrete", "Discrete", 1),
    "exp-classify": ModelSpec(2, 1, "Discrete", "Linear", 2),
    "MUTAG": ModelSpec(7, 4, "Linear", "Linear", 2),
    "PROTEINS": ModelSpec(3, 1, "Linear", "Linear", 2),
    "DD": ModelSpec(89, 1, "Linear", "Linear", 2),
    "REDDIT-BINARY": ModelSpec(1, 1, "Linear", "Linear", 2),
    "REDDIT-MULTI-5K": ModelSpec(1, 1, "Linear", "Linear", 5),
    "IMDB-BINARY": ModelSpec(1, 1, "Linear", "Linear", 2),
    "IMDB-MULTI": ModelSpec(1, 1, "Linear", "Linear", 3),
    "labid": ModelSpec(13, 0, "Linear", "Linear", 3),
    "labid_spheric": ModelSpec(25, 0, "Linear", "Linear", 3),
    "microns_classic_feat": ModelSpec(13, 0, "Linear", "Linear", 3),
    "microns_data": ModelSpec(25, 0, "Linear", "Linear", 3),
}


def create_model(cfg) -> GraphJepa:
    spec = _MODEL_SPECS.get(cfg.dataset)
    if spec is None:
        raise ValueError(f"Dataset '{cfg.dataset}' is not supported.")
    if cfg.metis.n_patches <= 0:
        raise ValueError("Graph partitioning (metis.n_patches) must be positive.")
    if not cfg.jepa.enable:
        raise ValueError("JEPA is disabled in the configuration.")

    logger = setup_logger(name="graphjepa", log_dir="logs", filename="train.log")
    return GraphJepa(
        nfeat_node=spec.node_features,
        nfeat_edge=spec.edge_features,
        nhid=cfg.model.hidden_size,
        nout=spec.output_dim,
        nlayer_gnn=cfg.model.nlayer_gnn,
        node_type=spec.node_type,
        edge_type=spec.edge_type,
        nlayer_mlpmixer=cfg.model.nlayer_mlpmixer,
        gMHA_type=cfg.model.gMHA_type,
        gnn_type=cfg.model.gnn_type,
        rw_dim=cfg.pos_enc.rw_dim,
        lap_dim=cfg.pos_enc.lap_dim,
        pooling=cfg.model.pool,
        dropout=cfg.train.dropout,
        mlpmixer_dropout=cfg.train.mlpmixer_dropout,
        n_patches=cfg.metis.n_patches,
        patch_rw_dim=cfg.pos_enc.patch_rw_dim,
        num_context_patches=cfg.jepa.num_context,
        num_target_patches=cfg.jepa.num_targets,
        debug=False,
        logger=logger,
    )
