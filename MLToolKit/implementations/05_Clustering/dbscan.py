"""DBSCAN clustering. Features are scaled before distance calculations."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def train_dbscan(X, eps: float = 0.5, min_samples: int = 5, scale: bool = True):
    values = np.asarray(X, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    scaler = StandardScaler() if scale else None
    X_used = scaler.fit_transform(values) if scaler is not None else values
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_used)
    return {"model": model, "scaler": scaler, "X_scaled": X_used, "labels": labels}
