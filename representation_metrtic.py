import torch

from log import config_logger
from asam import ASAM
from torch_geometric.loader import DataLoader
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error


from get_model import create_model



import numpy as np
from umap import UMAP
import matplotlib.pyplot as plt
def plot_umap_2d(X, y=None, title="JEPA graph embeddings (UMAP 2D)"):
    reducer = UMAP(n_components=2, n_neighbors=5, min_dist=0.1, metric="euclidean", random_state=42)
    Z = reducer.fit_transform(X)
    plt.figure(figsize=(7,6))
    if y is None:
        plt.scatter(Z[:,0], Z[:,1], s=14, alpha=0.8)
    else:
        sc = plt.scatter(Z[:,0], Z[:,1], c=y, s=14, alpha=0.8, cmap="Spectral")
        plt.colorbar(sc, label="label")
    plt.title(title)
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.tight_layout(); plt.show()


def load_model(cfg):
    ckpt = torch.load("checkpoints/checkpoint.ckpt", map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]       
    model = create_model(cfg).to(cfg.device)
    model.load_state_dict(state_dict, strict=False)
    return model









def encode_repr(loader,model,device = 'cuda'):
    X, y = [], []
    for data in loader:
        data.to(device)
        with torch.no_grad():
            features = model.encode(data)
            X.append(features.detach().cpu().numpy())
            y.append(data.y.detach().cpu().numpy())

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X,y


from torch import Tensor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def fit_and_eval_linear(X_tr, y_tr, X_te, y_te):
    # Если к тебе приходят torch.Tensor — безопасно переведём в numpy
    import numpy as np, torch
    if isinstance(X_tr, torch.Tensor): X_tr = X_tr.detach().cpu().numpy()
    if isinstance(y_tr, torch.Tensor): y_tr = y_tr.detach().cpu().numpy().reshape(-1)
    if isinstance(X_te, torch.Tensor): X_te = X_te.detach().cpu().numpy()
    if isinstance(y_te, torch.Tensor): y_te = y_te.detach().cpu().numpy().reshape(-1)

    # Импьютим NaN/Inf → сначала заменим Inf на NaN, потом заполним 0
    X_tr = np.where(np.isfinite(X_tr), X_tr, np.nan)
    X_te = np.where(np.isfinite(X_te), X_te, np.nan)

    clf = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0.0),  # <-- главное
        StandardScaler(),
        LogisticRegression(max_iter=500, n_jobs=None)
    )
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    y_pred = clf.predict(X_te)

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    return {
        "acc": float(accuracy_score(y_te, y_pred)),
        "f1_macro": float(f1_score(y_te, y_pred, average="macro")),
        "roc_auc": float(roc_auc_score(y_te, proba))
    }

    
        
       