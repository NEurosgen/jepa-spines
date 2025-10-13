
from pathlib import Path
import pickle, copy, itertools, heapq
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import from_networkx


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

import numpy as np
import torch
from torch_geometric.data import Data

def nx_to_pyg(G, make_undirected=True) -> Data:
    # 1) фиксируем порядок узлов
    nodes = sorted(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    # 2) узловые признаки
    # x: ожидаем массив одинаковой длины; если нет — не создаём
    xs, names, paths, poss = [], [], [], []
    has_x = all("x" in G.nodes[n] for n in nodes)
    has_pos = all("pos" in G.nodes[n] for n in nodes)
    has_name = any("name" in G.nodes[n] for n in nodes)
    has_path = any("path" in G.nodes[n] for n in nodes)

    if has_x:
        for n in nodes:
            xs.append(np.asarray(G.nodes[n]["x"], dtype=np.float32).ravel())
        F = len(xs[0])
        assert all(len(x)==F for x in xs), "Несовпадение размерностей x"
        x = torch.from_numpy(np.stack(xs, axis=0))  # [N,F]
    else:
        x = None

    if has_pos:
        for n in nodes:
            poss.append(np.asarray(G.nodes[n]["pos"], dtype=np.float32).ravel())
        pos = torch.from_numpy(np.stack(poss, axis=0))  # [N,D]
    else:
        pos = None

    if has_name:
        names = [G.nodes[n].get("name") for n in nodes]
    if has_path:
        paths = [G.nodes[n].get("path") for n in nodes]

    # 3) рёбра
    edges = []
    weights = []
    for u, v, data in G.edges(data=True):
        ui, vi = idx[u], idx[v]
        w = data.get("weight", None)
        if make_undirected:
            edges.append((ui, vi))
            edges.append((vi, ui))
            if w is not None:
                weights.extend([float(w), float(w)])
        else:
            edges.append((ui, vi))
            if w is not None:
                weights.append(float(w))

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2,E]
    else:
        edge_index = torch.empty((2,0), dtype=torch.long)

    edge_attr = None
    if weights:
        edge_attr = torch.tensor(weights, dtype=torch.float32).view(-1, 1)   # [E,1]

    # 4) сборка Data
    data = Data(x=x, edge_index=edge_index)
    if edge_attr is not None:
        data.edge_attr = edge_attr
    if pos is not None:
        data.pos = pos
    if has_name:
        data.node_names = names
    if has_path:
        data.file_paths = paths
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
def build_graph_list(dataset: Dataset, mode: str, knn: int, count: int, gamma: float = 0.8):
    
    all_data = []
    for path in dataset:
        label = _label_from_path(path)
        pyg_data_to_nx
        data: Data = torch.load(path, map_location="cpu",  weights_only=False)
        G = pyg_data_to_nx(data)
        if knn and knn > 0:
            G = knn_graph(G, k=knn, mutual=True,weight='attr')
        if mode == "MarkovGraphs":
            graphs = generate_markov_graphs(G, count=count, num_steps=10, gamma=gamma)
        elif mode == "CopiedGraphs":
            graphs = generate_copied_graphs(G, count=count)
        elif mode == "RandomPermutate":
            graphs = generate_random_permutations(G, count=count)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        for H in graphs:
            data = nx_to_pyg(H)
            data.y = torch.tensor([label], dtype=torch.float32)
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
                 knn: int = 4,
                 count_per_item: int = 20,
                 mode: str = "MarkovGraphs",
                 max_size: int = 1000,
                 gamma: float = 0.8,
                 transform=None,            
                 pre_transform=None):        
        super().__init__()
        self.mode = mode
        self.knn = knn
        self.count_per_item = count_per_item
        self.gamma = gamma
        self.transform = transform
        self.pre_transform = pre_transform

        data_list = build_graph_list(
            dataset=ds, mode=mode, knn=knn, count=count_per_item, gamma=gamma
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
            data = self.transform(data)
        return data


class WightlessDS(Dataset):
    def __init__(self,root:str):
        super().__init__()
        self.files = sorted(Path(root).rglob("*.pt"))
    def len(self):
        return len(self.files)
    def get(self,idx):
        return self.files[idx]