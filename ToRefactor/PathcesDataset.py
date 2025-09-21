import random
import torch
from torch_geometric.data import InMemoryDataset, Data, Batch
from torch_geometric.transforms import AddRandomWalkPE
from torch_geometric.utils import k_hop_subgraph, to_undirected

# ---- (опционально) METIS, но для маленьких графов можно без него
try:
    import pymetis
    HAS_METIS = True
except Exception:
    HAS_METIS = False

def _adjacency_from_edge_index(edge_index, num_nodes):
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    src, dst = edge_index
    nei = [[] for _ in range(num_nodes)]
    for u, v in zip(src.tolist(), dst.tolist()):
        if u != v:
            nei[u].append(v)
            nei[v].append(u)
    return nei

def metis_partition_labels(edge_index, num_nodes, p):
    adjacency = _adjacency_from_edge_index(edge_index, num_nodes)
    _, parts = pymetis.part_graph(nparts=p, adjacency=adjacency)
    return torch.tensor(parts, dtype=torch.long)

def random_partition_labels(num_nodes, p):
    # простая «равномерная» разметка по кластерам
    idx = torch.randperm(num_nodes)
    # делим на p кусков почти равного размера
    chunks = torch.chunk(idx, p)
    labels = torch.empty(num_nodes, dtype=torch.long)
    for cid, ch in enumerate(chunks):
        labels[ch] = cid
    return labels

def make_patch_subgraph(data: Data, node_mask: torch.Tensor, khop: int = 1) -> Data:
    if node_mask.sum() == 0:
        # пустышку не возвращаем
        return None
    seed_idx = node_mask.nonzero(as_tuple=False).view(-1)
    subset, edge_index, mapping, _ = k_hop_subgraph(
        seed_idx, num_hops=khop, edge_index=data.edge_index,
        relabel_nodes=True, num_nodes=data.num_nodes
    )
    if subset.numel() == 0:
        return None
    sub = Data()
    sub.edge_index = edge_index
    sub.num_nodes = subset.numel()
    if hasattr(data, "x") and data.x is not None:
        sub.x = data.x[subset]
    return sub

def build_patches(data: Data, p: int, labels: torch.Tensor, khop: int = 1, min_nodes: int = 2):
    patches = []
    for cid in range(p):
        mask = (labels == cid)
        if mask.sum() >= 1:
            sub = make_patch_subgraph(data, mask, khop=khop)
            if sub is not None and sub.num_nodes >= min_nodes:
                patches.append(sub)
    return patches

class PatchPositionalEncoder:
    def __init__(self, k: int = 16, attr_name: str = "rwse"):
        self.t = AddRandomWalkPE(walk_length=k, attr_name=attr_name)
        self.attr_name = attr_name

    def patch_pos(self, patch: Data) -> torch.Tensor:
        patch = self.t(patch)            # добавит patch.rwse: [num_nodes, k]
        rw = getattr(patch, self.attr_name)
        return rw.max(dim=0).values.unsqueeze(0)  # (1, k)

class PatchGroupDataset(InMemoryDataset):
    """
    Возвращает за раз: (контекст, список таргетов, список позиционок).
    По умолчанию контекст = целый граф, таргеты = его подпэтчи.
    """
    def __init__(
        self, base_graphs, p=8, rwse_k=16, khop=1, targets_per_ctx=4,
        context_mode: str = "graph",   # "graph" | "patch"
        min_patch_nodes: int = 2,
        use_metis: bool = False,       # для маленьких графов можно False
    ):
        super().__init__()
        self.graphs = base_graphs
        self.p = p
        self.khop = khop
        self.targets_per_ctx = targets_per_ctx
        self.context_mode = context_mode
        self.min_patch_nodes = min_patch_nodes
        self.use_metis = use_metis and HAS_METIS
        self.posenc = PatchPositionalEncoder(k=rwse_k)
        self._cache = {}  # idx -> dict(patches=[Data], P=[(1,K)], maybe ctx_patches)

    def __len__(self):
        return len(self.graphs)

    def _partition_small_graph(self, data: Data):
        n = data.num_nodes
        # ограничим число кластеров, чтобы патчи не развалились
        p_eff = max(2, min(self.p, n // max(self.min_patch_nodes, 1)))
        if p_eff < 2:
            p_eff = 2 if n >= 2 else 1
        if self.use_metis and HAS_METIS and n >= p_eff and p_eff >= 2:
            try:
                labels = metis_partition_labels(data.edge_index, n, p_eff)
            except Exception:
                labels = random_partition_labels(n, p_eff)
        else:
            labels = random_partition_labels(n, p_eff)
        return labels, p_eff

    def _prepare_graph(self, idx):
        if idx in self._cache:
            return self._cache[idx]
        data = self.graphs[idx]
        labels, p_eff = self._partition_small_graph(data)
        patches = build_patches(
            data, p_eff, labels, khop=self.khop, min_nodes=self.min_patch_nodes
        )
        # посчитаем позиционки для каждого патча
        P_list = [self.posenc.patch_pos(p) for p in patches]
        out = {"patches": patches, "P": P_list, "full_graph": data}
        self._cache[idx] = out
        return out

    def __getitem__(self, idx):
        pack = self._prepare_graph(idx)
        patches, P_list, full_graph = pack["patches"], pack["P"], pack["full_graph"]

        # фильтр на случай слишком малого числа патчей
        if len(patches) == 0:
            # деградация: таргет = весь граф
            return full_graph, [full_graph], [self.posenc.patch_pos(full_graph)]

        # выбираем контекст
        if self.context_mode == "graph":
            ctx = full_graph
        else:
            # контекст — один патч
            ctx = random.choice(patches)

        # выбираем M таргетов (без совпадений по возможности)
        M = min(self.targets_per_ctx, len(patches))
        tgt_idx = random.sample(range(len(patches)), M) if len(patches) >= M else list(range(len(patches)))
        tgt_list = [patches[i] for i in tgt_idx]
        P_sel = [P_list[i] for i in tgt_idx]   # позиционки ИМЕННО таргетов

        return ctx, tgt_list, P_sel





# inference_dataset.py
import torch, random
from torch_geometric.data import InMemoryDataset, Data, Batch

# Используем твои util'ы: build_patches, PatchPositionalEncoder и т.д.
# Предполагаем, что у тебя уже есть:
# - build_patches(data, p, labels, khop, min_nodes)
# - random_partition_labels(...) или metis_partition_labels(...)
# - PatchPositionalEncoder(k)

class GraphPatchesDataset(InMemoryDataset):
    """
    На каждый __getitem__ возвращает (full_graph, [patch_1..patch_M])
    Контекст тут не нужен — мы просто хотим латенты патчей для pooling.
    """
    def __init__(self, base_graphs, p=8, khop=1, min_patch_nodes=2, use_metis=False):
        super().__init__()
        self.graphs = base_graphs
        self.p = p
        self.khop = khop
        self.min_patch_nodes = min_patch_nodes
        self.use_metis = use_metis
        self._cache = {}

    def __len__(self): return len(self.graphs)

    def _partition_small_graph(self, data: Data):
        n = data.num_nodes
        p_eff = max(2, min(self.p, n // max(self.min_patch_nodes, 1)))
        if p_eff < 2: p_eff = 2 if n >= 2 else 1
        if self.use_metis and n >= p_eff and p_eff >= 2:
            try:
                labels = metis_partition_labels(data.edge_index, n, p_eff)
            except Exception:
                labels = random_partition_labels(n, p_eff)
        else:
            labels = random_partition_labels(n, p_eff)
        return labels, p_eff

    def _prepare(self, idx):
        if idx in self._cache: return self._cache[idx]
        data = self.graphs[idx]
        labels, p_eff = self._partition_small_graph(data)
        patches = build_patches(data, p_eff, labels, khop=self.khop, min_nodes=self.min_patch_nodes)
        self._cache[idx] = (data, patches)
        return self._cache[idx]

    def __getitem__(self, idx):
        full_graph, patches = self._prepare(idx)
        return full_graph, patches

def collate_infer(batch):
 
    assert len(batch) == 1
    from torch_geometric.data import Batch
    full_graph, patches = batch[0]
    full_b = Batch.from_data_list([full_graph])   # B=1
    tgt_b  = Batch.from_data_list(patches)        # B=M (может быть 0)
    return full_b, tgt_b, len(patches)
