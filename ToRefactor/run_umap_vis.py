# run_umap_vis.py
import torch
from torch.utils.data import DataLoader
import numpy as np
from umap import UMAP
import matplotlib.pyplot as plt
from PathcesDataset import GraphPatchesDataset, collate_infer

import torch
from torch_geometric.data import Batch

@torch.no_grad()
def graph_embedding_from_patches(model, patches_batch: Batch, pool: str = "mean"):
    """
    model: твой JEPAModel c .teacher_encoder (или .student_encoder)
    patches_batch: Batch из M подпэтчей одного графа
    Возвращает тензор (d,) — эмбеддинг графа.
    """
    if patches_batch.num_graphs == 0:
        # fallback: пустых патчей быть не должно, но на всякий случай
        return torch.zeros(model.cfg.latent_dim, device=next(model.parameters()).device)

    # как в статье для downstream: target-encoder -> усреднить по патчам
    Z = model.teacher_encoder(patches_batch)  # (M, d)
    if pool == "mean":
        g = Z.mean(dim=0)                     # (d,)
    elif pool == "max":
        g = Z.max(dim=0).values
    else:
        raise ValueError("pool must be 'mean' or 'max'")
    return g




def collect_graph_embeddings(model, base_graphs, p=8, khop=1, batch_workers=0):
    """
    Возвращает: np.array [N, d] эмбеддингов графов.
    Если у базовых графов есть .y — вернёт ещё labels (np.array [N,])
    """
    ds = GraphPatchesDataset(base_graphs, p=p, khop=khop, min_patch_nodes=2, use_metis=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=batch_workers,
                        collate_fn=collate_infer)

    device = next(model.parameters()).device
    model.eval()

    embs = []
    labels = []
    with torch.no_grad():
        for full_b, tgt_b in loader:
            full_b = full_b.to(device)
            tgt_b  = tgt_b.to(device)
            # графовый эмбеддинг через target-encoder на патчах
            g = graph_embedding_from_patches(model, tgt_b, pool="mean")  # (d,)
            embs.append(g.cpu().numpy())
            # если у графа есть метка:
            y = getattr(full_b, "y", None)
            if y is not None:
                # full_b.y shape: (B, ...) -> B=1
                labels.append(full_b.y.view(-1)[0].cpu().item())

    embs = np.stack(embs, axis=0)
    labels = np.array(labels) if len(labels) == len(embs) else None
    return embs, labels




# простая сборка эмбеддингов графов БЕЗ патчей
import torch
import numpy as np
from torch_geometric.loader import DataLoader  # именно отсюда!

@torch.no_grad()
def collect_graph_embeddings_simple(model, base_graphs, batch_size: int = 10, num_workers: int = 0):
    """
    Возвращает:
      X: np.ndarray [N, d] — эмбеддинги графов (teacher_encoder)
      y: np.ndarray [N]     — метки, если есть (иначе None)
      g: np.ndarray [N]     — group_id, если есть (иначе None)

    Требования:
      - model.teacher_encoder(batch) -> (B, d)
      - model.eval() выключает dropout/BN
      - GNN внутри делает global_mean_pool по batch
    """
    device = next(model.parameters()).device
    model.eval()

    loader = DataLoader(
        base_graphs, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    embs = []
    labels = []
    groups = []

    for batch in loader:
        batch = batch.to(device)
        Z = model.teacher_encoder(batch)     # (B, d)
        embs.append(Z.detach().cpu())

        # метки (если есть)
        y = getattr(batch, "y", None)
        if y is not None:
            y = y.view(-1).detach().cpu().numpy()
            labels.append(y)

  
        gid = getattr(batch, "group_id", None)
        if gid is not None:

            gid = gid.view(-1).detach().cpu().numpy()
            groups.append(gid)

    X = torch.cat(embs, dim=0).numpy()                   
    y = np.concatenate(labels) if labels else None       
    g = np.concatenate(groups) if groups else None       
    return X, y







def plot_umap_3d(X, y=None, title="JEPA graph embeddings (UMAP 3D)"):
    reducer = UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42)
    Z = reducer.fit_transform(X)
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    if y is None:
        ax.scatter(Z[:,0], Z[:,1], Z[:,2], s=10, alpha=0.85)
    else:
        p = ax.scatter(Z[:,0], Z[:,1], Z[:,2], c=y, s=10, alpha=0.85, cmap="Spectral")
        fig.colorbar(p, ax=ax, label="label")
    ax.set_title(title)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2"); ax.set_zlabel("UMAP-3")
    plt.tight_layout(); plt.show()
