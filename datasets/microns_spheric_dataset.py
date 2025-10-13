
from pathlib import Path
import pickle, copy, itertools, heapq
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import from_networkx

FEATURES = [
    "head_area", "head_volume", "head_skeletal_length",
    "head_width_ray", "head_width_ray_80_perc",
    "head_bbox_max", "head_bbox_min", "head_bbox_middle",
    "neck_area", "neck_volume", "neck_skeletal_length",
    "neck_width_ray", "neck_width_ray_80_perc",
]

# ---------- utils ----------

def to_scalar(x):
    if x is None:
        return 0.0
    if np.isscalar(x):
        v = float(x)
        if not np.isfinite(v): return 0.0
        return v
    x = np.asarray(x).flatten()
    v = float(x[0]) if x.size else 0.0
    if not np.isfinite(v): v = 0.0
    return v

def nx_to_data(G: nx.Graph) -> Data:
    # гарантированно делаем x: float32, без NaN
    for n in G.nodes:
        feat = [to_scalar(G.nodes[n].get(f, 0.0)) for f in FEATURES]
        G.nodes[n].clear()
        G.nodes[n]["x"] = torch.tensor(feat, dtype=torch.float32)
    data = from_networkx(G, group_node_attrs=["x"])
    if not hasattr(data, "x") or data.x is None:
        data.x = torch.zeros((G.number_of_nodes(), len(FEATURES)), dtype=torch.float32)
    return data

def graph_density(G: nx.Graph) -> float:
    n = G.number_of_nodes()
    if n <= 1: return 0.0
    return 2.0 * G.number_of_edges() / (n * (n - 1))


import numpy as np
import torch
from torch_geometric.data import Data

import torch
import networkx as nx
from torch_geometric.data import Data
from typing import Dict, Tuple

def nx_to_pyg(
    G: nx.Graph,
    feature_key: str = "sph_harm_coeffs",
    undirected: bool = True,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Data, Dict]:
    """
    Convert a NetworkX graph to PyTorch Geometric Data.
    Keeps only node features from `feature_key` (same length for all).
    Returns (data, id_map) where id_map maps original node ids -> integer indices.
    """

    # 1) Stable node order and id->index map
    nodes = list(G.nodes())
    id_map = {nid: i for i, nid in enumerate(nodes)}

    # 2) Collect node features (and validate lengths)
    feats = []
    for nid in nodes:
        attr = G.nodes[nid]
        if feature_key not in attr:
            print(G.nodes,G.nodes[nid])                                 # Я не для всех посчитал гармоники надо исправить а пока просто скип
            raise ValueError(f"Node {nid!r} missing '{feature_key}'")
        feats.append(attr[feature_key])

    lengths = {len(v) for v in feats}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent feature lengths: {sorted(lengths)}")
    x = torch.tensor(feats, dtype=dtype)

    # 3) Build edge_index with remapped integer ids
    edges = []
    for u, v in G.edges():
        ui, vi = id_map[u], id_map[v]
        edges.append((ui, vi))
        if undirected:
            edges.append((vi, ui))

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # 4) Create PyG Data
    data = Data(x=x, edge_index=edge_index)
    return data
def graph_density(G: nx.Graph) -> float:
    n = G.number_of_nodes()
    if n <= 1: return 0.0
    return 2.0 * G.number_of_edges() / (n * (n - 1))

def knn_graph(G: nx.Graph, k: int, weight: str = "weight", mutual: bool = False) -> nx.Graph:
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes(data=True))
    for u in G.nodes:
        # k лучших соседей по минимальному весу
        k_best = heapq.nsmallest(
            k, G[u].items(),
            key=lambda item: item[1].get(weight, 1.0)
        )
        for v, edata in k_best:
            H.add_edge(u, v, **edata)
    if mutual:
        to_remove = [(u, v) for u, v in H.edges if not H.has_edge(v, u)]
        H.remove_edges_from(to_remove)
        H = nx.Graph(H)
    else:
        # если нужен неориентированный kNN, раскомментируй:
        # H = nx.Graph(H)
        pass
    return H

# ---------- permutation strategies ----------

def _permute_markov_once(G: nx.Graph, num_steps: int = 10, gamma: float = 0.8) -> nx.Graph:
    G = copy.deepcopy(G)
    node_list = list(G.nodes)
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    N = len(node_list)
    if N == 0:
        return G
    probs = torch.ones(N) / N

    for _ in range(num_steps):
        idx = torch.multinomial(probs, num_samples=1).item()
        v = node_list[idx]
        neighbors = list(G.neighbors(v))
        if not neighbors:
            continue

        neighbor_idxs = torch.tensor([node_to_idx[n] for n in neighbors])
        neighbor_probs = probs[neighbor_idxs]
        s = neighbor_probs.sum()
        if s <= 0:
            continue
        neighbor_probs = neighbor_probs / s

        u_local_idx = torch.multinomial(neighbor_probs, 1).item()
        u = neighbors[u_local_idx]

        # обмен атрибутами узлов
        v_data = G.nodes[v].copy()
        u_data = G.nodes[u].copy()
        G.nodes[v].clear()
        G.nodes[u].clear()
        G.nodes[v].update(u_data)
        G.nodes[u].update(v_data)

        probs[node_to_idx[v]] *= gamma
        probs[node_to_idx[u]] *= gamma
        probs = probs / probs.sum()

    return G

def generate_markov_graphs(G: nx.Graph, count: int = 20, num_steps: int = 10, gamma: float = 0.8):
    out = [G.copy()]
    for _ in range(count):
        out.append(_permute_markov_once(G, num_steps=num_steps, gamma=gamma))
    return out

def generate_copied_graphs(G: nx.Graph, count: int = 20):
    return [G.copy() for _ in range(count+1)]

def generate_random_permutations(G: nx.Graph, count: int = 20):
    nodes = list(G.nodes)
    n = len(nodes)
    perms = itertools.permutations(nodes)
    out = [G.copy()]
    c = 0
    for perm in perms:
        H = G.copy()
        for i in range(n):
            H.nodes[nodes[i]].clear()
            H.nodes[nodes[i]].update(copy.deepcopy(G.nodes[perm[i]]))
        out.append(H)
        c += 1
        if c >= count:
            break
    return out

# ---------- dataset fabric ----------

def _label_from_path(path: Path) -> float:
    # подстрой под твою схему
    return 1.0 if "Wt" in path.parts else 0.0

def _load_base_graph(path: Path, knn: int) -> nx.Graph:
    with open(path, "rb") as fh:
        G = pickle.load(fh)
    if knn and knn > 0:
        G = knn_graph(G, k=knn, mutual=True)
    return G
def pyg_data_to_nx(data) -> nx.Graph:
    """
    Конвертирует PyG Data -> nx.Graph.
    Переносит:
      - x[i]  -> атрибут узла 'x' (np.ndarray)
      - node_names[i] -> 'name'
      - file_paths[i] -> 'path'
    """
    G = nx.Graph()

    # Узлы с атрибутами
    N = data.num_nodes
    has_x = hasattr(data, "x") and data.x is not None
    has_names = hasattr(data, "node_names")
    has_paths = hasattr(data, "file_paths")

    for i in range(N):
        attrs = {}
        if has_x:
            attrs["x"] = data.x[i].detach().cpu().numpy()
        if has_names:
            attrs["name"] = data.node_names[i]
        if has_paths:
            attrs["path"] = data.file_paths[i]
        G.add_node(int(i), **attrs)

    # Рёбра (убираем дубли, self-loops)
    ei = data.edge_index.detach().cpu().numpy()
    for u, v in ei.T:
        u, v = int(u), int(v)
        if u == v:
            continue
        if u < v:  # оставляем одно ребро на пару
            G.add_edge(u, v)

    # Если есть edge_attr — перенеси при желании:
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        edge_attr = data.edge_attr.detach().cpu().numpy()
        for (u, v), w in zip(ei.T, edge_attr):
            if u != v and u < v:
                G[u][v]["attr"] = w

    return G
def build_graph_list(dataset: Dataset, mode: str, count: int,classical_feat ,gamma: float = 0.8):
    
    all_data = []
    for path in dataset:
        with open(path, "rb") as f:
            G = pickle.load(f)
        skip = False
        for node in list(G.nodes()):
            if 'sph_harm_coeffs' not in G.nodes[node]:
                skip = True
                break
        if skip or len(G.nodes()) < 5: continue
        if mode == "MarkovGraphs":
            graphs = generate_markov_graphs(G, count=count, num_steps=10, gamma=gamma)
        elif mode == "CopiedGraphs":
            graphs = generate_copied_graphs(G, count=count)
        elif mode == "RandomPermutate":
            graphs = generate_random_permutations(G, count=count)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        for H in graphs:
            if classical_feat :
                data = nx_to_data(H)
            else:
                data = nx_to_pyg(H)
           # data.y = torch.tensor([label], dtype=torch.float32)
            #data.group_id = str(path)
            all_data.append(copy.deepcopy(data))
    return all_data

from torch_geometric.data import Dataset, Data

class SpineGraphDataset(Dataset):
    """
    mode:
      - 'MarkovGraphs'  — локальные свопы атрибутов у соседей (марковские)
      - 'CopiedGraphs'  — копии графа
      - 'RandomPermutate' — глобальная перестановка атрибутов узлов
    """
    def __init__(self, ds: Dataset,
                 count_per_item: int = 0,
                 mode: str = "CopiedGraphs",
                 max_size: int = 100000,
                 gamma: float = 0.8,
                 transform=None,            
                 pre_transform=None,
                 classical_feat = False):        
        super().__init__()
        self.mode = mode
        self.count_per_item = count_per_item
        self.gamma = gamma
        self.transform = transform
        self.pre_transform = pre_transform

        data_list = build_graph_list(
            dataset=ds, mode=mode, count=count_per_item, gamma=gamma,classical_feat=classical_feat
        )
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        if len(data_list) > max_size:
            data_list = data_list[:max_size]

        self._data_list = data_list

    def len(self):
        return len(self._data_list)

    def get(self, idx):
        data = self._data_list[idx]

        if self.transform is not None:
            return self.transform(data)

        return data
    









class WightlessDS(Dataset):
    def __init__(self,root:str):
        super().__init__()
        self.files = sorted(Path(root).rglob("*.gpickle"))
    def len(self):
        return len(self.files)
    def get(self,idx):
        return self.files[idx]