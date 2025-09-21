#!/usr/bin/env python3
"""
GraphCutter — a focused yet flexible graph slicing toolkit (PyTorch Geometric)

Key ideas
---------
• OOP API: a single `GraphCutter` orchestrates strategies, constraints and post‑processing.
• Strategies (plug‑and‑play): KHop, BFSBudget, RandomWalk, MetisPartition.
• Constraints: NodeBudget, EdgeBudget, EnsureConnected, MinDegree, Unique.
• Post‑processing: InduceEdges, Relabel, KeepLargestCC, SliceFeatures/Labels.
• Deterministic RNG for reproducibility.
• Works with torch_geometric `Data`. Optional `pymetis` for METIS.
• CLI with JSON config for batch runs.

Examples (Python)
-----------------
from graphcutter import GraphCutter, KHop, BFSBudget, RandomWalk, Relabel, InduceEdges
from graphcutter import NodeBudget, EnsureConnected

cutter = GraphCutter(
    strategies=[
        KHop(k=2),
        BFSBudget(budget_nodes=64),
        RandomWalk(walk_len=32, repeats=2),
    ],
    constraints=[NodeBudget(128), EnsureConnected()],
    post=[InduceEdges(), Relabel(),],
    rng_seed=0,
)
patches = cutter.cut(data, seeds=[0, 3, 7])

CLI
---
python graphcutter.py --in graph.pt --out outdir --cfg config.json

Config example
--------------
{
  "rng_seed": 0,
  "seeds": [0, 3, 7],
  "strategies": [
    {"name": "khop", "k": 2},
    {"name": "bfs", "budget_nodes": 64},
    {"name": "rwalk", "walk_len": 32, "repeats": 2}
  ],
  "constraints": [
    {"name": "node_budget", "max_nodes": 128},
    {"name": "ensure_connected"}
  ],
  "post": [
    {"name": "induce"},
    {"name": "relabel"}
  ]
}

License: MIT
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, k_hop_subgraph
from torch_geometric.utils import subgraph as pyg_subgraph
from torch_geometric.utils import degree, is_undirected
from torch_geometric.utils import coalesce

try:
    import pymetis  # type: ignore
    HAS_METIS = True
except Exception:
    HAS_METIS = False

# ------------------------------- Patch -------------------------------

@dataclass
class Patch:
    nodes_orig: Tensor        # shape [m] original node ids
    edge_index: Tensor        # shape [2, E] (may be empty prior to induce)
    num_nodes: int            # m
    x: Optional[Tensor] = None
    y: Optional[Tensor] = None
    node_mapping: Optional[Dict[int, int]] = None  # original -> new (after Relabel)

    def to_data(self) -> Data:
        d = Data(edge_index=self.edge_index, num_nodes=self.num_nodes)
        if self.x is not None:
            d.x = self.x
        if self.y is not None:
            d.y = self.y
        return d

# ----------------------------- Utilities -----------------------------

def _infer_num_nodes(data: Data) -> int:
    """Robustly infer num_nodes from data.num_nodes, edge_index, x, y."""
    n_decl = int(getattr(data, 'num_nodes', 0) or 0)
    m_edge = 0
    if getattr(data, 'edge_index', None) is not None and data.edge_index.numel() > 0:
        m_edge = int(data.edge_index.max().item()) + 1
    xlen = int(data.x.size(0)) if getattr(data, 'x', None) is not None and data.x.dim() > 0 else 0
    ylen = int(data.y.size(0)) if getattr(data, 'y', None) is not None and data.y.dim() > 0 else 0
    return max(n_decl, m_edge, xlen, ylen)

def _effective_num_nodes(data: Data, seeds: Sequence[int]) -> int:
    base = _infer_num_nodes(data)
    if seeds:
        return max(base, int(max(seeds)) + 1)
    return base

def _ensure_undirected(edge_index: Tensor, num_nodes: int) -> Tensor:
    if not is_undirected(edge_index, num_nodes=num_nodes):
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    return coalesce(edge_index, num_nodes=num_nodes)

# --- Compatibility: connected components (no PyG dependency) ---
# Returns (num_components, labels) where labels[i] in [0..num_components-1]
# for node i in [0..num_nodes-1]. Assumes nodes are 0..num_nodes-1 in the
# given edge index; relabel before if needed.

def _connected_components(ei: Tensor, num_nodes: int) -> Tuple[int, Tensor]:
    if ei.numel() == 0 or num_nodes <= 1:
        labels = torch.zeros(num_nodes, dtype=torch.long)
        return (1 if num_nodes > 0 else 0), labels
    # Build adjacency lists
    src, dst = ei
    adj = [[] for _ in range(num_nodes)]
    for u, v in zip(src.tolist(), dst.tolist()):
        if u == v:
            continue
        adj[u].append(v)
        adj[v].append(u)
    labels = torch.full((num_nodes,), -1, dtype=torch.long)
    comp_id = 0
    from collections import deque
    for s in range(num_nodes):
        if labels[s] != -1:
            continue
        # BFS
        dq = deque([s])
        labels[s] = comp_id
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if labels[v] == -1:
                    labels[v] = comp_id
                    dq.append(v)
        comp_id += 1
    return comp_id, labels



def _unique_undirected(ei: Tensor) -> Tensor:
    """Return each undirected edge once (no (u,v)/(v,u) duplicates)."""
    if ei.numel() == 0:
        return ei
    u, v = ei
    a = torch.minimum(u, v)
    b = torch.maximum(u, v)
    pairs = torch.stack([a, b], dim=1)         # [E, 2] canonical (min,max)
    uniq = torch.unique(pairs, dim=0)          # dedupe
    return uniq.t().contiguous()               # back to [2, E]


# ---------------------------- Strategies -----------------------------

class Strategy:
    def cut(self, data: Data, seeds: Sequence[int], rng: np.random.Generator) -> List[Patch]:
        raise NotImplementedError


class KHop(Strategy):
    def __init__(self, k: int = 2):
        self.k = int(k)

    def cut(self, data: Data, seeds: Sequence[int], rng: np.random.Generator) -> List[Patch]:
        num_nodes = _effective_num_nodes(data, seeds)
        base_n = _infer_num_nodes(data)
        ei = _ensure_undirected(data.edge_index, num_nodes)
        out: List[Patch] = []
        for s in seeds:
            s = int(s)
            # skip seeds not present in the real graph
            if s < 0 or s >= base_n:
                continue
            nodes, _, _, _ = k_hop_subgraph(torch.tensor([s]), self.k, ei,
                                            num_nodes=num_nodes, relabel_nodes=False)
            nodes = nodes.to(torch.long)
            # clamp to the real graph and skip empty
            nodes = nodes[(nodes >= 0) & (nodes < base_n)]
            if nodes.numel() == 0:
                continue
            out.append(Patch(nodes_orig=nodes,
                             edge_index=torch.empty((2, 0), dtype=torch.long),
                             num_nodes=int(nodes.numel())))
        return out



class BFSBudget(Strategy):
    def __init__(self, budget_nodes: int = 64):
        self.budget_nodes = int(budget_nodes)

    def cut(self, data: Data, seeds: Sequence[int], rng: np.random.Generator) -> List[Patch]:
        num_nodes = _effective_num_nodes(data, seeds)
        ei = _ensure_undirected(data.edge_index, num_nodes)
        src, dst = ei
        adj = [[] for _ in range(num_nodes)]
        for u, v in zip(src.tolist(), dst.tolist()):
            if u != v:
                adj[u].append(v)
                adj[v].append(u)
        patches: List[Patch] = []
        from collections import deque
        for s in seeds:
            s = int(s)
            if s < 0 or s >= num_nodes:
                continue
            visited = {s}
            queue = deque([s])
            while queue and len(visited) < self.budget_nodes:
                u = queue.popleft()
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        if len(visited) >= self.budget_nodes:
                            break
                        queue.append(v)
            nodes = torch.tensor(sorted(visited), dtype=torch.long)
            patches.append(Patch(nodes_orig=nodes, edge_index=torch.empty((2,0),dtype=torch.long), num_nodes=nodes.numel()))
        return patches


class RandomWalk(Strategy):
    def __init__(self, walk_len: int = 32, repeats: int = 1, unique: bool = True):
        self.walk_len = int(walk_len)
        self.repeats = int(repeats)
        self.unique = bool(unique)

    def cut(self, data: Data, seeds: Sequence[int], rng: np.random.Generator) -> List[Patch]:
        # Try to import either PyG's CSR-based RW or torch_cluster's COO-based RW
        rw_impl = None
        impl_kind = None  # 'csr' or 'coo'
        try:
            from torch_geometric.utils import random_walk as rw_impl  # csr variant
            impl_kind = 'csr'
        except ImportError:
            try:
                from torch_cluster import random_walk as rw_impl  # coo variant
                impl_kind = 'coo'
            except ImportError as e:
                raise ImportError("random_walk not available: install torch-cluster or newer torch-geometric") from e

        num_nodes = _effective_num_nodes(data, seeds)
        ei = _ensure_undirected(data.edge_index, num_nodes)
        row, col = ei

        if impl_kind == 'csr':
            deg = torch.bincount(row, minlength=num_nodes)
            rowptr = torch.zeros(num_nodes + 1, dtype=torch.long)
            rowptr[1:] = torch.cumsum(deg, dim=0)
            order = torch.argsort(row)
            col_sorted = col[order]
        torch.manual_seed(int(rng.integers(0, 2**31-1)))
        patches: List[Patch] = []
        for s in seeds:
            s = int(s)
            if s < 0 or s >= num_nodes:
                continue
            seen: List[int] = []
            for _ in range(self.repeats):
                if impl_kind == 'csr':
                    path = rw_impl(rowptr, col_sorted, start=torch.tensor([s]), walk_length=self.walk_len)[0]
                else:
                    path = rw_impl(row, col, torch.tensor([s]), self.walk_len, num_nodes=num_nodes)[0]
                seen.extend(path.tolist())
            if self.unique:
                nodes = torch.tensor(sorted(set(int(v) for v in seen if 0 <= int(v) < num_nodes)), dtype=torch.long)
            else:
                nodes = torch.tensor(sorted([int(v) for v in seen if 0 <= int(v) < num_nodes]), dtype=torch.long).unique()
            patches.append(Patch(nodes_orig=nodes, edge_index=torch.empty((2,0),dtype=torch.long), num_nodes=nodes.numel()))
        return patches


class MetisPartition(Strategy):
    def __init__(self, parts: int = 8):
        self.parts = int(parts)
        if not HAS_METIS:
            raise RuntimeError("pymetis is not installed; `pip install pymetis`")

    def cut(self, data: Data, seeds: Sequence[int], rng: np.random.Generator) -> List[Patch]:
        num_nodes = _infer_num_nodes(data)
        ei = _ensure_undirected(data.edge_index, num_nodes)
        src, dst = ei
        adj = [[] for _ in range(num_nodes)]
        for u, v in zip(src.tolist(), dst.tolist()):
            if u != v:
                adj[u].append(v)
                adj[v].append(u)
        _, labels = pymetis.part_graph(nparts=self.parts, adjacency=adj)
        by_part: Dict[int, List[int]] = {}
        for nid, lab in enumerate(labels):
            by_part.setdefault(lab, []).append(nid)
        patches: List[Patch] = []
        for lab, nodes_list in by_part.items():
            nodes = torch.tensor(sorted(nodes_list), dtype=torch.long)
            patches.append(Patch(nodes_orig=nodes, edge_index=torch.empty((2,0),dtype=torch.long), num_nodes=nodes.numel()))
        return patches





# ---------------------------- Constraints ----------------------------

class Constraint:
    def apply(self, data, patch):
        return patch
    def end_cut(self):
        pass  # called after each cut()

class NodeBudget(Constraint):
    def __init__(self, max_nodes: int):
        self.max_nodes = int(max_nodes)
    def apply(self, data: Data, patch: Patch) -> Optional[Patch]:
        if patch.num_nodes <= self.max_nodes:
            return patch
        # Trim by degree‑descending to keep central nodes
        degs = degree(data.edge_index[0], num_nodes=_infer_num_nodes(data))[patch.nodes_orig]
        order = torch.argsort(degs, descending=True)[: self.max_nodes]
        nodes = patch.nodes_orig[order]
        return Patch(nodes_orig=nodes, edge_index=torch.empty((2,0),dtype=torch.long), num_nodes=nodes.numel())

class EdgeBudget(Constraint):
    def __init__(self, max_edges: int):
        self.max_edges = int(max_edges)
    def apply(self, data: Data, patch: Patch) -> Optional[Patch]:
        if patch.edge_index.numel() == 0:
            return patch  # will be induced later
        E = patch.edge_index.size(1)
        if E <= self.max_edges:
            return patch
        keep = torch.arange(self.max_edges, dtype=torch.long)
        ei = patch.edge_index[:, keep]
        return Patch(nodes_orig=patch.nodes_orig, edge_index=ei, num_nodes=patch.num_nodes)

class MinDegree(Constraint):
    def __init__(self, k: int):
        self.k = int(k)
    def apply(self, data: Data, patch: Patch) -> Optional[Patch]:
        sub_nodes = patch.nodes_orig
        sub_ei, _ = pyg_subgraph(sub_nodes, data.edge_index, relabel_nodes=False)
        degs = torch.bincount(sub_ei[0], minlength=_infer_num_nodes(data))[sub_nodes]
        mask = degs >= self.k
        if mask.sum() == 0:
            return None
        nodes = sub_nodes[mask]
        return Patch(nodes_orig=nodes, edge_index=torch.empty((2,0),dtype=torch.long), num_nodes=nodes.numel())

class EnsureConnected(Constraint):
    def apply(self, data: Data, patch: Patch) -> Optional[Patch]:
        if patch.num_nodes <= 1:
            return patch
        # Build local mapping orig->local 0..m-1
        nodes = patch.nodes_orig.to(torch.long)
        m = int(nodes.numel())
        lut = torch.full((_infer_num_nodes(data),), -1, dtype=torch.long)
        lut[nodes] = torch.arange(m, dtype=torch.long)
        # Subgraph on original ids, then relabel to local for CC computation
        sub_ei_global, _ = pyg_subgraph(nodes, _ensure_undirected(data.edge_index, _infer_num_nodes(data)), relabel_nodes=False)
        sub_ei_local = lut[sub_ei_global]
        _, labels = _connected_components(sub_ei_local, m)
        counts = torch.bincount(labels, minlength=labels.max().item()+1 if labels.numel()>0 else 0)
        if counts.numel() == 0:
            return patch
        keep_label = int(torch.argmax(counts))
        keep_mask = labels == keep_label
        keep_nodes_orig = nodes[keep_mask]
        return Patch(nodes_orig=keep_nodes_orig, edge_index=torch.empty((2,0),dtype=torch.long), num_nodes=int(keep_mask.sum()))

class Unique(Constraint):
    def __init__(self):
        self._seen: set = set()

    def apply(self, data, patch):
        if patch.num_nodes == 0:
            return None
        key = tuple(patch.nodes_orig.tolist())
        if key in self._seen:
            return None
        self._seen.add(key)
        return patch

    def end_cut(self):
        # clear after a single cut() finishes
        self._seen.clear()




# -------------------------- Post‑processing --------------------------

class Post:
    def apply(self, data: Data, patch: Patch) -> Patch:
        return patch

class InduceEdges(Post):
    def apply(self, data: Data, patch: Patch) -> Patch:
        base_n = _infer_num_nodes(data)
        safe_nodes = patch.nodes_orig[(patch.nodes_orig >= 0) & (patch.nodes_orig < base_n)]
        if safe_nodes.numel() == 0:
            return Patch(nodes_orig=safe_nodes,
                         edge_index=torch.empty((2,0), dtype=torch.long),
                         num_nodes=0, x=None, y=None)
        ei, _ = pyg_subgraph(safe_nodes, _ensure_undirected(data.edge_index, base_n),
                             relabel_nodes=False)
        ei = _unique_undirected(ei)  # <<< make simple-undirected
        return Patch(nodes_orig=safe_nodes, edge_index=ei,
                     num_nodes=int(safe_nodes.numel()), x=patch.x, y=patch.y)





class Relabel(Post):
    def apply(self, data: Data, patch: Patch) -> Patch:
        nodes = patch.nodes_orig.to(torch.long)
        m = nodes.numel()
        # LUT: original id -> local 0..m-1
        lut = torch.full((_infer_num_nodes(data),), -1, dtype=torch.long)
        lut[nodes] = torch.arange(m, dtype=torch.long)
        # Remap edges if present
        if patch.edge_index.numel() > 0:
            ei = lut[patch.edge_index]
        else:
            ei = patch.edge_index
        mapping = {int(nodes[i]): int(i) for i in range(m)}
        # Slice features if node-level features exist
        x = data.x[nodes] if getattr(data, 'x', None) is not None else None
        # Handle labels robustly: node-level vs graph-level
        y = None
        if getattr(data, 'y', None) is not None:
            y_t = data.y
            try:
                # Node-level if first dim equals num_nodes
                if y_t.dim() > 0 and y_t.size(0) == _infer_num_nodes(data):
                    y = y_t[nodes]
                else:
                    # Graph-level or other shape -> keep as-is
                    y = y_t
            except Exception:
                y = y_t
        return Patch(nodes_orig=nodes, edge_index=ei, num_nodes=m, x=x, y=y, node_mapping=mapping)

class KeepLargestCC(Post):
    def apply(self, data: Data, patch: Patch) -> Patch:
        # If there are no edges or single node, nothing to do
        if patch.edge_index.numel() == 0 or patch.num_nodes <= 1:
            return patch
        # Ensure edges are labeled locally 0..m-1; if not, relabel via nodes_orig
        ei = patch.edge_index
        m = int(patch.num_nodes)
        if int(ei.max()) >= m:
            nodes = patch.nodes_orig.to(torch.long)
            lut = torch.full((_infer_num_nodes(data),), -1, dtype=torch.long)
            lut[nodes] = torch.arange(m, dtype=torch.long)
            ei = lut[ei]
        _, labels = _connected_components(ei, m)
        counts = torch.bincount(labels, minlength=labels.max().item()+1 if labels.numel()>0 else 0)
        if counts.numel() == 0:
            return patch
        keep_label = int(torch.argmax(counts))
        keep_mask = labels == keep_label
        # Map kept local indices back to original ids
        nodes = patch.nodes_orig.to(torch.long)
        keep_nodes_orig = nodes[keep_mask]
        return Patch(nodes_orig=keep_nodes_orig, edge_index=torch.empty((2,0),dtype=torch.long), num_nodes=int(keep_mask.sum()), x=None, y=None)

# ----------------------------- Orchestrator -----------------------------

@dataclass
class GraphCutter:
    strategies: List[Strategy]
    constraints: List[Constraint] = None
    post: List[Post] = None
    rng_seed: int = 0

    def __post_init__(self):
        if self.constraints is None:
            self.constraints = []
        if self.post is None:
            self.post = []
        self.rng = np.random.default_rng(self.rng_seed)

    def cut(self, data: Data, seeds: Sequence[int]) -> List[Patch]:
        patches: List[Patch] = []
        try:
            for strat in self.strategies:
                raw = strat.cut(data, seeds, self.rng)
                for p in raw:
                    keep = p
                    for con in self.constraints:
                        if keep is None:
                            break
                        keep = con.apply(data, keep)
                    if keep is None:
                        continue
                    out = keep
                    for post in self.post:
                        out = post.apply(data, out)
                    if getattr(out, "num_nodes", 0) <= 0:
                        continue
                    patches.append(out)
            return patches
        finally:
            # ← clear per-call memory in constraints
            for con in self.constraints:
                if hasattr(con, "end_cut"):
                    con.end_cut()


# ------------------------------- I/O --------------------------------

def load_graph_pt(path: str) -> Data:
    try:
        # PT < 2.6 or envs where object load is allowed:
        return torch.load(path)
    except Exception:
        # PT 2.6+: allow full object unpickling for trusted local test file
        return torch.load(path, weights_only=False)



def load_graph_npz(path: str) -> Data:
    z = np.load(path)
    ei = torch.from_numpy(z["edge_index"]).long()
    n = int(z["num_nodes"][0])
    d = Data(edge_index=ei, num_nodes=n)
    if "x" in z:
        d.x = torch.from_numpy(z["x"]).float()
    if "y" in z:
        d.y = torch.from_numpy(z["y"]).float()
    return d


def save_patch_npz(path: str, patch: Patch) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = {
        "edge_index": patch.edge_index.cpu().numpy(),
        "num_nodes": np.array([patch.num_nodes], dtype=np.int64),
        "nodes_orig": patch.nodes_orig.cpu().numpy(),
    }
    if patch.x is not None:
        out["x"] = patch.x.cpu().numpy()
    if patch.y is not None:
        out["y"] = patch.y.cpu().numpy()
    if patch.node_mapping is not None:
        keys = np.array(list(patch.node_mapping.keys()), dtype=np.int64)
        vals = np.array(list(patch.node_mapping.values()), dtype=np.int64)
        out["node_mapping_keys"] = keys
        out["node_mapping_vals"] = vals
    np.savez_compressed(path, **out)


def _load_any(path: str) -> Data:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pt":
        return load_graph_pt(path)
    if ext == ".npz":
        return load_graph_npz(path)
    raise ValueError("Unsupported input format; use .pt or .npz")

# -------------------------------- CLI -------------------------------

_STRAT_REG = {
    "khop": KHop,
    "bfs": BFSBudget,
    "rwalk": RandomWalk,
    "metis": MetisPartition,
}
_CON_REG = {
    "node_budget": NodeBudget,
    "edge_budget": EdgeBudget,
    "min_degree": MinDegree,
    "ensure_connected": EnsureConnected,
    "unique": Unique,
}
_POST_REG = {
    "induce": InduceEdges,
    "relabel": Relabel,
    "keep_largest_cc": KeepLargestCC,
}


def _from_cfg(cfg: dict) -> GraphCutter:
    rng_seed = int(cfg.get("rng_seed", 0))
    strategies = []
    for s in cfg.get("strategies", []):
        name = s.get("name")
        if name not in _STRAT_REG:
            raise ValueError(f"Unknown strategy: {name}")
        kwargs = {k: v for k, v in s.items() if k != "name"}
        strategies.append(_STRAT_REG[name](**kwargs))
    constraints = []
    for c in cfg.get("constraints", []):
        name = c.get("name")
        if name not in _CON_REG:
            raise ValueError(f"Unknown constraint: {name}")
        kwargs = {k: v for k, v in c.items() if k != "name"}
        constraints.append(_CON_REG[name](**kwargs))
    post = []
    for p in cfg.get("post", []):
        name = p.get("name")
        if name not in _POST_REG:
            raise ValueError(f"Unknown post: {name}")
        kwargs = {k: v for k, v in p.items() if k != "name"}
        post.append(_POST_REG[name](**kwargs))
    return GraphCutter(strategies=strategies, constraints=constraints, post=post, rng_seed=rng_seed)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GraphCutter — flexible graph slicing")
    p.add_argument("--in", dest="inp", required=True, help="Input graph (.pt or .npz)")
    p.add_argument("--out", dest="outdir", required=True, help="Output directory")
    p.add_argument("--cfg", dest="cfg", required=True, help="JSON config path")
    p.add_argument("--seeds", nargs="*", type=int, default=None, help="Override seeds list from cfg (optional)")
    return p.parse_args()


def _main() -> None:
    args = _parse_args()
    with open(args.cfg, "r") as f:
        cfg = json.load(f)
    seeds = args.seeds if args.seeds is not None and len(args.seeds)>0 else cfg.get("seeds", [])
    if len(seeds) == 0:
        raise SystemExit("Seeds are required (cfg.seeds or --seeds)")

    data = _load_any(args.inp)
    cutter = _from_cfg(cfg)
    patches = cutter.cut(data, seeds=seeds)

    os.makedirs(args.outdir, exist_ok=True)
    for i, p in enumerate(patches):
        save_patch_npz(os.path.join(args.outdir, f"patch_{i:04d}.npz"), p)
    print(f"Saved {len(patches)} patches to {args.outdir}")


if __name__ == "__main__":
    _main()
