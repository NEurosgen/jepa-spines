# infer_embeddings.py
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
