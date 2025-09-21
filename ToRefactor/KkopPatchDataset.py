from typing import List, Tuple, Optional
import random
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.utils import k_hop_subgraph, to_undirected
from torch_geometric.transforms import AddRandomWalkPE

# ---- (опционально) METIS
try:
    import pymetis
    HAS_METIS = True
except Exception:
    HAS_METIS = False


# ========== утилиты

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)

def _adjacency_from_edge_index(edge_index: Tensor, num_nodes: int):
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    src, dst = edge_index
    nei = [[] for _ in range(num_nodes)]
    for u, v in zip(src.tolist(), dst.tolist()):
        if u != v:
            nei[u].append(v); nei[v].append(u)
    return nei

def metis_partition_labels(edge_index: Tensor, num_nodes: int, p: int) -> Tensor:
    adjacency = _adjacency_from_edge_index(edge_index, num_nodes)
    _, parts = pymetis.part_graph(nparts=p, adjacency=adjacency)
    return torch.as_tensor(parts, dtype=torch.long)

def random_partition_labels(num_nodes: int, p: int) -> Tensor:
    idx = torch.randperm(num_nodes)
    chunks = torch.chunk(idx, p)
    labels = torch.empty(num_nodes, dtype=torch.long)
    for cid, ch in enumerate(chunks):
        labels[ch] = cid
    return labels

def _copy_node_attrs(subset: Tensor, data: Data, out: Data, keys=("x","pos","y")):
    for k in keys:
        if hasattr(data, k):
            v = getattr(data, k)
            if isinstance(v, Tensor) and v.size(0) == data.num_nodes:
                setattr(out, k, v[subset])

def _copy_edge_attrs(edge_mask: Optional[Tensor], data: Data, out: Data, keys=("edge_attr",)):
    # edge_mask приходит из k_hop_subgraph (четвёртый возврат)
    if edge_mask is None: return
    for k in keys:
        if hasattr(data, k):
            v = getattr(data, k)
            if isinstance(v, Tensor) and v.size(0) == data.edge_index.size(1):
                setattr(out, k, v[edge_mask])

def make_khop_patch(data: Data, node_mask: Tensor, khop: int,
                    node_keys=("x","pos","y"), edge_keys=("edge_attr",)) -> Optional[Data]:
    if node_mask.sum() == 0:
        return None
    seeds = node_mask.nonzero(as_tuple=False).view(-1)
    subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
        seeds, num_hops=khop, edge_index=data.edge_index,
        relabel_nodes=True, num_nodes=data.num_nodes, flow="source_to_target"
    )
    if subset.numel() == 0:
        return None
    sub = Data(edge_index=edge_index_sub, num_nodes=int(subset.numel()))
    _copy_node_attrs(subset, data, sub, node_keys)
    _copy_edge_attrs(edge_mask, data, sub, edge_keys)
    return sub


# ========== позиционные энкодинги патчей (RWSE-пуллинг)

class PatchPositionalEncoder:
    def __init__(self, k: int = 16, attr_name: str = "rwse"):
        self.t = AddRandomWalkPE(walk_length=k, attr_name=attr_name)
        self.attr_name = attr_name

    @torch.no_grad()
    def patch_pos(self, patch: Data) -> Tensor:
        p = patch.clone()
        p = self.t(p)                              # добавит p.rwse: [N, k]
        rw = getattr(p, self.attr_name)
        return rw.max(dim=0).values.unsqueeze(0)   # (1, k)


# ========== сам датасет

class KHopPatchDataset(Dataset):
    """
    __getitem__ -> (context_patch, patches_list, P_list, ctx_id)

    - Граф режется на p кластеров вершин (METIS или случайно).
    - Для каждого кластера берём k-hop окрестность (индуцированный подграф на объединении вершин).
    - Контекст — один из полученных патчей (равновероятно), он же присутствует в patches_list по индексу ctx_id.
    """
    def __init__(
        self,
        base_graphs: List[Data],
        p: int = 8,
        khop: int = 1,
        rwse_k: int = 16,
        min_patch_nodes: int = 2,
        use_metis: bool = False,
        node_keys: Tuple[str, ...] = ("x","pos","y"),
        edge_keys: Tuple[str, ...] = ("edge_attr",),
        seed: int = 42,
        max_ratio: float = 1.0,   # верхняя граница на долю узлов патча от графа (1.0 = без ограничения)
        shrink_khop_if_big: bool = True,  # если патч слишком большой, понижаем k-hop пока не влезет
    ):
        self.graphs = base_graphs
        self.p = p
        self.khop = khop
        self.min_patch_nodes = min_patch_nodes
        self.use_metis = (use_metis and HAS_METIS)
        self.node_keys = node_keys
        self.edge_keys = edge_keys
        self.posenc = PatchPositionalEncoder(k=rwse_k)
        self.max_ratio = float(max(1e-6, max_ratio))
        self.shrink_khop_if_big = shrink_khop_if_big
        self._cache = {}  # idx -> dict(full, patches, P)
        set_seed(seed)

    def __len__(self): return len(self.graphs)

    def _partition(self, data: Data) -> Tuple[Tensor, int]:
        n = int(data.num_nodes)
        # ограничиваем число кластеров размером патчей
        max_p_by_size = max(1, n // max(self.min_patch_nodes, 1))
        p_eff = max(1, min(self.p, max_p_by_size))
        if self.use_metis and HAS_METIS and n >= p_eff and p_eff >= 2:
            try:
                labels = metis_partition_labels(data.edge_index, n, p_eff)
            except Exception:
                labels = random_partition_labels(n, p_eff)
        else:
            labels = random_partition_labels(n, p_eff)
        return labels, p_eff

    def _build_patches(self, data: Data, labels: Tensor, p_eff: int) -> List[Data]:
        patches: List[Data] = []
        n_all = int(data.num_nodes)
        max_nodes = int(self.max_ratio * n_all + 1e-9)

        for cid in range(p_eff):
            mask = (labels == cid)
            if int(mask.sum()) == 0:
                continue

            kh = self.khop
            sub = make_khop_patch(data, mask, kh, self.node_keys, self.edge_keys)
            if sub is None: 
                continue

            # при необходимости ужимаем khop, чтобы не брать почти весь граф
            if self.shrink_khop_if_big and self.max_ratio < 1.0:
                while sub is not None and sub.num_nodes > max_nodes and kh > 0:
                    kh -= 1
                    sub = make_khop_patch(data, mask, kh, self.node_keys, self.edge_keys)

            if sub is not None and sub.num_nodes >= self.min_patch_nodes:
                patches.append(sub)

        return patches

    def _fallback_random_patch(self, data: Data) -> Data:
        # делаем k-hop патч вокруг случайного узла; если не вышло — весь граф
        n = int(data.num_nodes)
        seed = torch.randint(0, n, (1,))
        mask = torch.zeros(n, dtype=torch.bool)
        mask[seed] = True
        sub = make_khop_patch(data, mask, self.khop, self.node_keys, self.edge_keys)
        return sub if (sub is not None and sub.num_nodes >= self.min_patch_nodes) else data.clone()

    def _prepare(self, idx: int):
        if idx in self._cache:
            return self._cache[idx]

        data = self.graphs[idx]
        labels, p_eff = self._partition(data)
        patches = self._build_patches(data, labels, p_eff)

        # --- ГАРАНТИЯ: минимум 2 патча ---
        if len(patches) < 2:
            # берём весь граф как патч + ещё один fallback-патч
            patches = [data.clone(), self._fallback_random_patch(data)]

        P_list = [self.posenc.patch_pos(p) for p in patches]
        pack = {"full": data, "patches": patches, "P": P_list}
        self._cache[idx] = pack
        return pack

    def __getitem__(self, idx: int):
        pack = self._prepare(idx)
        patches, P_list = pack["patches"], pack["P"]

        # выбираем контекст (НЕ мутируем кэш!)
        ctx_id = random.randrange(len(patches))
        ctx = patches[ctx_id].clone()

        tgt_idx = [i for i in range(len(patches)) if i != ctx_id]
        targets  = [patches[i].clone() for i in tgt_idx]
        P_targets = [P_list[i] for i in tgt_idx]          # список (1,K), согласованный с targets

        y_src = getattr(self.graphs[idx], "y", None)
        if y_src is not None:
            ctx.y = y_src
            for t in targets:
                t.y = y_src

        return ctx, targets, P_targets, ctx_id




# ========== коллейты (если нужны)

from typing import List, Tuple, Any
import torch
from torch_geometric.data import Data, Batch

def _unpack_item(item: Tuple[Any, ...]):
    """
    Поддержка форматов:
      (ctx, targets)                         # 2
      (ctx, targets, P_list)                 # 3
      (ctx, targets, P_list, ctx_id)         # 4
    Возвращает: (ctx, targets)
    """
    if len(item) == 2:
        ctx, targets = item
    elif len(item) >= 3:
        ctx, targets = item[0], item[1]   # игнорируем P_list и ctx_id
    else:
        raise ValueError(f"Unexpected item len={len(item)}")
    return ctx, targets


def collate_ctx_targets_single(batch):
    """
    batch_size = 1
    Вход: [(ctx, targets) | (ctx, targets, P) | (ctx, targets, P, ctx_id)]
    Выход: (ctx_b: Batch[1], tgt_b: Batch|None, m: int)
    """
    assert len(batch) == 1
    ctx, targets = _unpack_item(batch[0])
    ctx_b = Batch.from_data_list([ctx])
    tgt_b = Batch.from_data_list(targets) if len(targets) > 0 else None
    return ctx_b, tgt_b, torch.tensor([len(targets)],dtype=torch.float16)


def collate_ctx_targets_many(batch):
    """
    batch_size >= 1
    Вход: список элементов в одном из форматов выше.
    Выход:
      - ctx_batch: Batch из B контекстов
      - tgt_batch: Batch из всех таргетов подряд (или None)
      - tgt_ptr:   LongTensor длины (B+1) — offsets по таргетам
    """
    ctx_list: List[Data] = []
    all_targets: List[Data] = []
    tgt_ptr = [0]

    for item in batch:
        ctx, targets = _unpack_item(item)
        ctx_list.append(ctx)
        all_targets.extend(targets)
        tgt_ptr.append(tgt_ptr[-1] + len(targets))

    ctx_batch = Batch.from_data_list(ctx_list)
    tgt_batch = Batch.from_data_list(all_targets) if all_targets else None
    tgt_ptr = torch.tensor(tgt_ptr, dtype=torch.float16)
    return ctx_batch, tgt_batch, tgt_ptr

from torch_geometric.data import Batch

def collate_ctx_with_targets(batch):
    """
    batch_size == 1
    Вход: [(ctx: Data, targets: list[Data], P_list: list[Tensor[1,K]], ctx_id)]
    Выход: (ctx_batch[B=1], tgt_batch[B=M], P: (M,K))
    """
    assert len(batch) == 1
    ctx, targets, P_list, _ = batch[0]

    ctx_b = Batch.from_data_list([ctx])
    if len(targets) > 0:
        tgt_b = Batch.from_data_list(targets)
        # P_list — список (1, K) → склеиваем в (M, K)
        P = torch.cat(P_list, dim=0)
    else:
        tgt_b = None
        # если таргетов нет, вернём корректный пустой тензор
        pe_dim = P_list[0].size(1) if len(P_list) > 0 else 0
        P = torch.empty((0, pe_dim), dtype=torch.float32)
    return ctx_b, tgt_b, P
