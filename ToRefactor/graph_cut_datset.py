from typing import List, Dict, Any, Optional, Tuple
import random
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import AddRandomWalkPE
from torch_geometric.utils import to_undirected

from GraphCut.graphcutter import GraphCutter  # your cutter


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)


def _infer_num_nodes(g: Data) -> int:
    """Robust num_nodes inference."""
    if getattr(g, "num_nodes", None):
        return int(g.num_nodes)
    if getattr(g, "x", None) is not None and g.x.dim() > 0:
        return int(g.x.size(0))
    if getattr(g, "edge_index", None) is not None and g.edge_index.numel() > 0:
        return int(g.edge_index.max().item()) + 1
    raise ValueError("Cannot infer num_nodes for the graph.")


class PatchPositionalEncoder:
    """RWSE over patch -> aggregate to (1, k)."""
    def __init__(self, k: int = 16, attr_name: str = "rwse"):
        self.t = AddRandomWalkPE(walk_length=k, attr_name=attr_name)
        self.attr_name = attr_name
        self.k = k

    @torch.no_grad()
    def patch_pos(self, patch: Data) -> Tensor:
        if getattr(patch, "edge_index", None) is None:
            raise ValueError("patch has no edge_index")
        if getattr(patch, "num_nodes", None) is None:
            if getattr(patch, "x", None) is not None:
                patch.num_nodes = patch.x.size(0)
            else:
                patch.num_nodes = _infer_num_nodes(patch)

        patch = patch.clone()
        patch.edge_index = to_undirected(patch.edge_index, num_nodes=patch.num_nodes)

        patch = self.t(patch)  # adds attr_name: [N, k]
        rw: Tensor = getattr(patch, self.attr_name)
        return rw.max(dim=0).values.unsqueeze(0)  # (1, k)


class CutPatchesDataset(Dataset):
    """
    For each base graph:
      - context: one graph (Data)
      - target_graphs: list of patch graphs (List[Data])
      - encoded: list of tensors (1, k) for each patch
    """
    def __init__(
        self,
        base_graphs: List[Data],
        targets_cutter: GraphCutter,
        context_cutter: GraphCutter,
        rwse_k: int,
        max_targets: int = 1,
        seed: int = 42,
        use_cache: bool = True,
    ):
        self.base_graphs = base_graphs
        self.targets_cutter = targets_cutter
        self.context_cutter = context_cutter
        self.posenc = PatchPositionalEncoder(k=rwse_k)
        self.seed = seed
        self.use_cache = use_cache
        self._cache: Dict[Tuple[int, int, Tuple[int, ...], int, int], Dict[str, Any]] = {}
        self.max_targets = max_targets

    def __len__(self) -> int:
        return len(self.base_graphs)

    def _prepare(self, idx: int, ctx_seed: int, tgts_seed: List[int]) -> Dict[str, Any]:
        cache_key = (idx, int(ctx_seed), tuple(sorted(map(int, tgts_seed))), self.max_targets, self.posenc.k)
        if self.use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        g = self.base_graphs[idx]

        # Context: try to cut, fallback to full graph if empty
        ctx_patches = self.context_cutter.cut(g, [ctx_seed])
        if len(ctx_patches) == 0:
            ctx_graph: Data = g
        else:
            ctx_graph = ctx_patches[0].to_data()

        # Targets: may be empty (allowed)
        tgt_patches = self.targets_cutter.cut(g, tgts_seed)
        tgt_graphs: List[Data] = [p.to_data() for p in tgt_patches if getattr(p, "num_nodes", 0) > 0]

        encoded_list: List[Tensor] = []
        for p in tgt_graphs:
            enc = self.posenc.patch_pos(p)  # (1, k)
            encoded_list.append(enc)

        pack = {
            "context": ctx_graph,
            "target_graphs": tgt_graphs,
            "encoded": encoded_list,  # list of (1, k)
            "index": idx,
        }
        if self.use_cache:
            self._cache[cache_key] = pack
        return pack

    def get_node_seed(self, idx: int, num_targets: int) -> Tuple[int, List[int]]:
        set_seed(self.seed + idx)
        n = _infer_num_nodes(self.base_graphs[idx])
        ctx_seed = random.randrange(n)
        num_targets = min(num_targets, n)
        targets_seed = random.sample(range(n), num_targets)
        return ctx_seed, targets_seed

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ctx_seed, tgts_seed = self.get_node_seed(idx, self.max_targets)
        return self._prepare(idx, ctx_seed, tgts_seed)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(N={len(self)}, "
                f"rwse_k={self.posenc.k}, cache={self.use_cache})")


def collate_cut_patches(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Returns:
      - context_batch: Batch (contexts)
      - target_batch:  Batch (all targets concatenated)
      - encoded: Tensor [sum_T, k] (RWSE per target)
      - ptr: LongTensor [B+1] (prefix sums of #targets per item)
      - sizes: LongTensor [B] (#targets per item)
      - indices: LongTensor [B] (dataset indices)
    """
    contexts = [item["context"] for item in batch]
    targets_list_all: List[Data] = []
    encoded_all: List[Tensor] = []
    sizes: List[int] = []

    k_dim: Optional[int] = None

    for item in batch:
        tgts = item["target_graphs"]
        encs = item["encoded"]
        if len(encs) > 0 and k_dim is None:
            k_dim = int(encs[0].size(1))

        if len(tgts) != len(encs):
            raise RuntimeError("len(target_graphs) != len(encoded) for an item")

        targets_list_all.extend(tgts)
        if len(encs) > 0:
            encoded_all.append(torch.cat(encs, dim=0))  # [Ti, k]
        sizes.append(len(tgts))

    sizes_t = torch.tensor(sizes, dtype=torch.long)
    ptr = torch.zeros(len(sizes) + 1, dtype=torch.long)
    if sizes_t.numel() > 0:
        ptr[1:] = torch.cumsum(sizes_t, dim=0)

    if len(encoded_all) > 0:
        encoded = torch.cat(encoded_all, dim=0)  # [sum_T, k]
    else:
        encoded = torch.empty((0, k_dim if k_dim is not None else 0), dtype=torch.float32)

    context_batch = Batch.from_data_list(contexts) if len(contexts) else None
    target_batch = Batch.from_data_list(targets_list_all) if len(targets_list_all) else None
    indices = torch.tensor([item["index"] for item in batch], dtype=torch.long)

    return {
        "context_batch": context_batch,
        "target_batch": target_batch,
        "encoded": encoded,    # [sum_T, k]
        "ptr": ptr,            # [B+1]
        "sizes": sizes_t,      # [B]
        "indices": indices,    # [B]
    }


def pad_encoded(encoded: Tensor, sizes: torch.Tensor) -> Tuple[Tensor, Tensor]:
    """
    Convert ragged [sum_T, k] + sizes[B] into dense (B, T_max, k) + mask (B, T_max).
    """
    B = int(sizes.size(0))
    T_max = int(sizes.max()) if sizes.numel() > 0 else 0
    k = int(encoded.size(1)) if encoded.numel() > 0 else 0

    out = encoded.new_zeros((B, T_max, k))
    mask = torch.zeros((B, T_max), dtype=torch.bool)

    start = 0
    for i, t in enumerate(sizes.tolist()):
        if t > 0:
            out[i, :t] = encoded[start:start+t]
            mask[i, :t] = True
        start += t
    return out, mask


# DataLoader
def worker_init_fn(worker_id: int):
    base_seed = torch.initial_seed() % (2**31 - 1)
    random.seed(base_seed + worker_id)
    torch.manual_seed(base_seed + worker_id)


def make_loader(dataset: Dataset,
                batch_size: int = 4,
                shuffle: bool = True,
                num_workers: int = 0,
                pin_memory: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_cut_patches,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )
