# dataset_jepa.py
import torch, random, numpy as np, networkx as nx, pickle, heapq
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

FEATURES = [
    "head_area", "head_volume", "head_skeletal_length",
    "head_width_ray", "head_width_ray_80_perc",
    "head_bbox_max", "head_bbox_min", "head_bbox_middle",
    "neck_area", "neck_volume", "neck_skeletal_length",
    "neck_width_ray", "neck_width_ray_80_perc",
]


def knn_graph(G, k, weight="weight", mutual=False):
    H = nx.DiGraph(); H.add_nodes_from(G.nodes(data=True))
    for u in G.nodes:
        k_best = heapq.nsmallest(k, G[u].items(),
                                 key=lambda it: it[1].get(weight, 1.0))
        for v, edata in k_best: H.add_edge(u, v, **edata)
    if mutual:
        to_remove = [(u,v) for u,v in H.edges if not H.has_edge(v,u)]
        H.remove_edges_from(to_remove); H = nx.Graph(H)
    return H


def to_scalar(x):
    """Гарантированно вернуть float."""
    if x is None:
        return 0.0
    if np.isscalar(x):
        return float(x)
    x = np.asarray(x).flatten()
    return float(x[0])     



def nx_to_data(G):
    for n in G.nodes:
        feat = [to_scalar(G.nodes[n].get(f, 0.0)) for f in FEATURES]
#       feat.append(to_scalar(graph_density(G)))
        G.nodes[n].clear()                       
        G.nodes[n]["x"] = torch.tensor(feat, dtype=torch.float32)
    return from_networkx(G, group_node_attrs=["x"])  



def k_hop_subgraph(G: nx.Graph, seeds, k=1):
    nodes = set(seeds)
    frontier = set(seeds)
    for _ in range(k):
        nxt = set()
        for u in frontier: nxt.update(G.neighbors(u))
        frontier = nxt - nodes
        nodes |= nxt
    H = G.subgraph(nodes).copy()
    return H, {n:i for i,n in enumerate(H.nodes())}

def build_context_and_targets(G: nx.Graph, *, m=4, k_ctx=1, k_tgt=1):
    """Пример: контекст = k_ctx-hop вокруг случайной ветви/якоря,
       таргеты = m отдельных k_tgt-hop патчей вокруг разных узлов этой же области."""
    if G.number_of_nodes() == 0:
        raise ValueError("Empty graph")
    anchor = random.choice(list(G.nodes()))
    Gc, relabel_ctx = k_hop_subgraph(G, [anchor], k=k_ctx)
    context = nx_to_data(Gc)


    ctx_nodes = list(Gc.nodes())
    if len(ctx_nodes) <= 1:
        targets, z_list = [], []
        return context, targets, torch.zeros((0, 1))
    cands = [n for n in ctx_nodes if n != anchor]
    random.shuffle(cands)
    chosen = cands[:m] if len(cands) >= m else cands + random.choices(cands, k=m-len(cands))

    targets, z_list = [], []
    for u in chosen:
        Gt, relabel_t = k_hop_subgraph(G, [u], k=k_tgt)
        targets.append(nx_to_data(Gt))
        try:
            dist = nx.shortest_path_length(G, source=anchor, target=u)
        except nx.NetworkXNoPath:
            dist = 99
        z_list.append([float(dist), float(G.degree[u])])
    Z = torch.tensor(np.asarray(z_list, dtype=np.float32)) 
    return context, targets, Z
from pathlib import Path
import copy
def GetDatasetGrap(root,knn = 0 ):
    files =  sorted(Path(root).glob("*/*.pkl"))
    All_graphs = []
    for path in files:
        label = 1.0 if "Wt" in path.parts else 0.0  # adjust to your schema
        with open(path, "rb") as fh:
            if knn == 0:
                G = pickle.load(fh)
            else:
                G = knn_graph(pickle.load(fh),knn,mutual=True)
        
        All_graphs.append(copy.deepcopy(nx_to_data(G)))

    return All_graphs
        

