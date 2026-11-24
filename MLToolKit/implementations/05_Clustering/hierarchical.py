"""Agglomerative clustering and a SciPy dendrogram."""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler


def _matrix(X):
    values = np.asarray(X, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return values


def train_hierarchical(X, n_clusters: int = 3, linkage_method: str = "ward", scale: bool = True):
    values = _matrix(X)
    scaler = StandardScaler() if scale else None
    X_used = scaler.fit_transform(values) if scaler is not None else values
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = model.fit_predict(X_used)
    return {"model": model, "scaler": scaler, "X_scaled": X_used, "labels": labels}


def linkage_matrix(X, method: str = "ward", scale: bool = True):
    values = _matrix(X)
    X_used = StandardScaler().fit_transform(values) if scale else values
    return linkage(X_used, method=method)


def plot_dendrogram(X, method: str = "ward"):
    import matplotlib.pyplot as plt

    linked = linkage_matrix(X, method=method)
    fig, ax = plt.subplots(figsize=(8, 5))
    dendrogram(linked, ax=ax)
    ax.set_title("Hierarchical clustering dendrogram")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    return fig, ax
