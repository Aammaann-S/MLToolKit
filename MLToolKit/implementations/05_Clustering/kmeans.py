"""K-Means clustering with scaling fit on the matrix being clustered."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def _matrix(X):
    values = np.asarray(X, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return values


def train_kmeans(X, n_clusters: int = 3, random_state: int = 42, scale: bool = True):
    values = _matrix(X)
    scaler = StandardScaler() if scale else None
    X_used = scaler.fit_transform(values) if scaler is not None else values
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = model.fit_predict(X_used)
    return {"model": model, "scaler": scaler, "X_scaled": X_used, "labels": labels}
