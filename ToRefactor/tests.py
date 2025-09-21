

import hydra
from omegaconf import DictConfig
from graph import GetDatasetGrap
from PathcesDataset import PatchGroupDataset
from model.JEPA import JEPAModel
import pytorch_lightning as pl
from torch_geometric.data import Batch
import torch
from run_umap_vis import collect_graph_embeddings,plot_umap_2d,plot_umap_3d,collect_graph_embeddings_simple
from KkopPatchDataset import KHopPatchDataset,collate_ctx_targets_single,collate_ctx_with_targets

import os, random
import numpy as np
import torch

def set_seed(seed: int, deterministic_torch: bool = True):
    # 1) базовые генераторы
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # 2) pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


import numpy as np
from rep_eval import evaluate_representations, RepEvaluator, RepEvalConfig

import torch
from torch_geometric.data import Data
from GraphCut.graphcutter import (
    GraphCutter, KHop, BFSBudget, RandomWalk,
    NodeBudget, EnsureConnected, Unique,
    InduceEdges, Relabel, save_patch_npz
)
from ExperimentalDataset import SpineGraphDataset,WightlessDS
from graph_cut_datset import make_loader
from normalize_graph import _apply_norm,compute_feature_stats
def get_strategy(mode):
    if mode == 'BFS':
        return BFSBudget
    if mode == 'khop':
        return KHop


def GetCutters(cfg):
    ctx_cutter = GraphCutter(
        strategies=[
           # get_strategy(cfg.cutter.strategy.ctx_mode)(cfg.cutter.strategy.ctx_param)
                          

        ],
        constraints=[
         #   NodeBudget(10),             # hard cap to avoid runaway growth
            EnsureConnected(),   
            Unique()        # keep only largest connected component
        # drop duplicates by node set
        ],
        post=[
            InduceEdges(),             # induce edges for the chosen node set
            Relabel()                  # reindex nodes to 0..m-1 and slice x/y
        ],
        rng_seed=0
    )


    tgtx_cutter = GraphCutter(
        strategies=[
           get_strategy(cfg.cutter.strategy.tgt_mode)(cfg.cutter.strategy.tgt_param)            

        ],
        constraints=[
            #NodeBudget(10),             # hard cap to avoid runaway growth
            EnsureConnected(),         # keep only largest connected component
            Unique()                   # drop duplicates by node set
        ],
        post=[
            InduceEdges(),             # induce edges for the chosen node set
            Relabel()                  # reindex nodes to 0..m-1 and slice x/y
        ],
        rng_seed=0
    )
    return ctx_cutter,tgtx_cutter


def build_cutters(cfg: DictConfig):
    # контекст — по желанию можно добавить стратегию из конфига
    ctx_strats = []
    if "ctx_mode" in cfg.cutter.strategy and cfg.cutter.strategy.ctx_mode is not None:
        ctx_cls = get_strategy(cfg.cutter.strategy.ctx_mode)
        ctx_strats.append(ctx_cls(cfg.cutter.strategy.ctx_param))

    ctx_cutter = GraphCutter(
        strategies=ctx_strats,
        constraints=[EnsureConnected(), Unique()],
        post=[InduceEdges(), Relabel()],
        rng_seed=getattr(cfg, "seed", 0)
    )

    tgt_cls = get_strategy(cfg.cutter.strategy.tgt_mode)
    tgt_cutter = GraphCutter(
        strategies=[tgt_cls(cfg.cutter.strategy.tgt_param)],
        constraints=[EnsureConnected(), Unique()],
        post=[InduceEdges(), Relabel()],
        rng_seed=getattr(cfg, "seed", 0)
    )
    return ctx_cutter, tgt_cutter


def get_datasets(cfg: DictConfig):
    base_ds = WightlessDS(cfg.data.root)


    n_total = len(base_ds)
    n_val = int(cfg.data.val_ratio * n_total)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(base_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(cfg.seed))
    train_syn = SpineGraphDataset(
        ds=train_ds,
        knn=cfg.data.knn,
        count_per_item=cfg.data.count_per_item,
        mode=cfg.data.mode,
        gamma=cfg.data.gamma
    )
    val_syn = SpineGraphDataset(
        ds=val_ds,
        knn=cfg.data.knn,
        count_per_item=0,
        mode=cfg.data.mode,
        gamma=cfg.data.gamma
    )
    # нормализация по train
    method = cfg.data.norm.method
    params = compute_feature_stats(train_syn, method=method, attr=cfg.data.norm.attr)
    apply_norm_inplace(train_syn, params, method=method, attr=cfg.data.norm.attr)
    apply_norm_inplace(val_syn,   params, method=method, attr=cfg.data.norm.attr)

    # патчи
    ctx_cutter, tgt_cutter = build_cutters(cfg)
    train_patch_ds = CutPatchesDataset(
        base_graphs=train_syn,
        context_cutter=ctx_cutter,
        targets_cutter=tgt_cutter,
        rwse_k=cfg.model.pe_dim,
        max_targets=cfg.data.max_patches
    )
    val_patch_ds = CutPatchesDataset(
        base_graphs=val_syn,
        context_cutter=ctx_cutter,
        targets_cutter=tgt_cutter,
        rwse_k=cfg.model.pe_dim,
        max_targets=cfg.data.max_patches
    )
    return train_patch_ds, val_patch_ds
def apply_norm_inplace(dataset, params, method="zscore", attr="x", eps=1e-8, clip=None):
    for data in dataset:
        if hasattr(data, attr) and getattr(data, attr) is not None:
            x = getattr(data, attr)
            x_norm = _apply_norm(x, method=method, params=params,
                                 eps=eps, per_graph=False, clip=clip)
            setattr(data, attr, x_norm)

from graph_cut_datset import CutPatchesDataset
from torch.utils.data import DataLoader,random_split
from ExperimentalDataset import SpineGraphDataset
from pytorch_lightning.callbacks import LearningRateMonitor
@hydra.main(version_base=None,config_path="conf", config_name="config")
def main(cfg: DictConfig):
    set_seed(42)
    train_patch_ds,val_patch_ds = get_datasets(cfg)
    train_loader = make_loader(train_patch_ds, batch_size=cfg.trainer.batch_size, shuffle=True)
    val_loader = make_loader(val_patch_ds, batch_size=cfg.trainer.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 5. Модель
    model = JEPAModel(cfg).to(device)

    # 6. Trainer
   
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        precision="16-mixed",
        gradient_clip_val=1.0,
        log_every_n_steps=20,
        enable_checkpointing=True,
        callbacks=[LearningRateMonitor(logging_interval='epoch')] 
    )

    # 7. fit с train и val
    trainer.fit(model, train_loader, val_loader)








if __name__ == "__main__":
    main()
