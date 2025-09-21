# src/datasets/spine_graph.py
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
    # делаем копии графа (именно .copy(), не один и тот же объект)
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

def build_graph_list(dataset: Dataset, mode: str, knn: int, count: int, gamma: float = 0.8):
    
    all_data = []
    for path in dataset:
        label = _label_from_path(path)
        G = _load_base_graph(path, knn)
        if mode == "MarkovGraphs":
            graphs = generate_markov_graphs(G, count=count, num_steps=10, gamma=gamma)
        elif mode == "CopiedGraphs":
            graphs = generate_copied_graphs(G, count=count)
        elif mode == "RandomPermutate":
            graphs = generate_random_permutations(G, count=count)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        for H in graphs:
            data = nx_to_data(H)
            data.y = torch.tensor([label], dtype=torch.float32)
            #data.group_id = str(path)
            all_data.append(copy.deepcopy(data))
    return all_data

# ---------- main Dataset ----------

class SpineGraphDataset(Dataset):
    """
    Режимы:
      - 'MarkovGraphs': марковские перестановки атрибутов (локальные свопы у соседей)
      - 'CopiedGraphs': просто много копий исходного графа
      - 'RandomPermutate': глобальные перестановки атрибутов узлов
    """
    def __init__(self, ds:Dataset,
                 knn: int = 4,
                 count_per_item: int = 20,
                 mode: str = "MarkovGraphs",
                 max_size: int = 1000,
                 gamma: float = 0.8,
                 transform=None, pre_transform=None):
        super().__init__( )
        self.mode = mode
        self.knn = knn
        self.count_per_item = count_per_item
        self.gamma = gamma

        self._data_list = build_graph_list(
            dataset=ds, mode=mode, knn=knn, count=count_per_item, gamma=gamma
        )
        if len(self._data_list) > max_size:
            self._data_list = self._data_list[:max_size]

    def len(self):
        return len(self._data_list)

    def get(self, idx):
        return self._data_list[idx]


class WightlessDS(Dataset):
    def __init__(self,root:str):
        super().__init__()
        self.files = sorted(Path(root).glob("*/*.pkl"))
    def len(self):
        return len(self.files)
    def get(self,idx):
        return self.files[idx]

