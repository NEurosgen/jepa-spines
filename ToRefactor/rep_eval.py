# rep_eval.py
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score,
    silhouette_score, pairwise_distances,
    adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score
)
from sklearn.cluster import KMeans

def _standardize(X: np.ndarray) -> np.ndarray:
    """Zero-mean / unit-variance по признакам."""
    return StandardScaler().fit_transform(X)

def _cv_scores(clf, X: np.ndarray, y: np.ndarray, cv) -> Dict[str, float]:
    """Усреднить по CV набор стандартных метрик для бинарной/мульти-классовой задачи."""
    # accuracy
    acc = cross_val_score(clf, X, y, scoring="accuracy", cv=cv, n_jobs=-1)
    # f1 (macro)
    f1m = cross_val_score(clf, X, y, scoring="f1_macro", cv=cv, n_jobs=-1)
    # balanced accuracy
    bacc = cross_val_score(clf, X, y, scoring="balanced_accuracy", cv=cv, n_jobs=-1)
    out = {
        "acc_mean": float(acc.mean()), "acc_std": float(acc.std(ddof=1)),
        "f1_macro_mean": float(f1m.mean()), "f1_macro_std": float(f1m.std(ddof=1)),
        "bacc_mean": float(bacc.mean()), "bacc_std": float(bacc.std(ddof=1)),
    }
    # ROC-AUC (только если >1 класс и логрег даёт predict_proba)
    try:
        auc = cross_val_score(clf, X, y, scoring="roc_auc_ovr", cv=cv, n_jobs=-1)
        out.update({"auc_ovr_mean": float(auc.mean()), "auc_ovr_std": float(auc.std(ddof=1))})
    except Exception:
        pass
    return out

def neighborhood_purity(X: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """Средняя доля соседей того же класса (kNN-граф на эмбеддингах, без обучения)."""
    D = pairwise_distances(X, X, metric="euclidean")
    np.fill_diagonal(D, np.inf)
    nn = np.argpartition(D, kth=k, axis=1)[:, :k]  # (N, k) индексы k ближайших

    purity = (y[nn] == y[:, None]).float().mean(dim=1)

    return float(purity.mean())

def intra_inter_distances(X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Средние расстояния внутри/между классами (евклид)."""
    D = pairwise_distances(X, X, metric="euclidean")
    N = len(y)
    same = (y[:, None] == y[None, :])
    diff = ~same
    np.fill_diagonal(D, np.nan)
    intra = np.nanmean(D[same])
    inter = np.nanmean(D[diff])
    return {"intra_mean": float(intra), "inter_mean": float(inter), "margin": float(inter - intra)}

@dataclass
class RepEvalConfig:
    seed: int = 42
    kfolds: int = 5
    knn_k: int = 5
    kmeans_init: int = 10
    standardize: bool = True

class RepEvaluator:
    def __init__(self, cfg: RepEvalConfig = RepEvalConfig()):
        self.cfg = cfg
        self.rng = np.random.RandomState(cfg.seed)

    def evaluate_supervised(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Линейная оценка (LogReg) + kNN-классификатор в k-fold CV."""
        assert y is not None, "y is required for supervised evaluation"
        Xs = _standardize(X) if self.cfg.standardize else X

        cv = StratifiedKFold(n_splits=self.cfg.kfolds, shuffle=True, random_state=self.cfg.seed)

        # Logistic Regression (saga — подходит для L2/L1 и мультикласса)
        lr = LogisticRegression(
            solver="saga", penalty="l2", C=1.0, max_iter=1000, random_state=self.cfg.seed, n_jobs=-1
        )
        lr_scores = _cv_scores(lr, Xs, y, cv)

        # kNN classifier
        #knn = KNeighborsClassifier(n_neighbors=self.cfg.knn_k, weights="distance", n_jobs=-1)
        #knn_scores = _cv_scores(knn, Xs, y, cv)

        # соседская чистота (без обучения)
        knn_purity = neighborhood_purity(Xs, y, k=self.cfg.knn_k)

        out = {f"lr/{k}": v for k, v in lr_scores.items()}
        #out.update({f"knn/{k}": v for k, v in knn_scores.items()})
        out["knn/neigh_purity"] = float(knn_purity)

        # дистанции
        out.update({f"dist/{k}": v for k, v in intra_inter_distances(Xs, y).items()})
        return out

    def evaluate_clustering(self, X: np.ndarray, y: Optional[np.ndarray] = None, n_clusters: Optional[int] = None) -> Dict[str, float]:
        """Кластерные метрики: KMeans → ARI/NMI/… + silhouette."""
        Xs = _standardize(X) if self.cfg.standardize else X
        if n_clusters is None:
            if y is None:
                raise ValueError("n_clusters must be provided if y is None")
            n_clusters = int(len(np.unique(y)))

        km = KMeans(n_clusters=n_clusters, n_init=self.cfg.kmeans_init, random_state=self.cfg.seed)
        labels = km.fit_predict(Xs)

        out = {}
        # silhouette по предсказанным кластерам
        if n_clusters > 1 and len(Xs) > n_clusters:
            try:
                out["silhouette"] = float(silhouette_score(Xs, labels, metric="euclidean"))
            except Exception:
                pass

        # внешние метрики — только если есть y
        if y is not None:
            out["ari"] = float(adjusted_rand_score(y, labels))
            out["nmi"] = float(normalized_mutual_info_score(y, labels))
            out["homogeneity"] = float(homogeneity_score(y, labels))
            out["completeness"] = float(completeness_score(y, labels))

        return out

    def evaluate_all(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Сводка: supervised (если есть y) + clustering."""
        result = {}
        if y is not None:
            result.update(self.evaluate_supervised(X, y))
            result.update({f"clust/{k}": v for k, v in self.evaluate_clustering(X, y).items()})
        else:
            result.update({f"clust/{k}": v for k, v in self.evaluate_clustering(X, y=None, n_clusters=10).items()})
        return result

# --- удобная обёртка ---
def evaluate_representations(X: np.ndarray, y: Optional[np.ndarray] = None, seed: int = 42) -> Dict[str, float]:
    """Быстрый путь: одно вызов — полный отчёт."""
    evaluator = RepEvaluator(RepEvalConfig(seed=seed))
    return evaluator.evaluate_all(X, y)
