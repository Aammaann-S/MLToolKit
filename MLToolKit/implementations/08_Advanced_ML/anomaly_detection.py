"""Isolation Forest and local outlier factor."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


def _scaled(X):
    values = np.asarray(X, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    scaler = StandardScaler()
    return scaler.fit_transform(values), scaler


def isolation_forest(X, contamination: float = 0.05, random_state: int = 42):
    scaled, scaler = _scaled(X)
    model = IsolationForest(contamination=contamination, random_state=random_state)
    labels = model.fit_predict(scaled)
    return {"model": model, "scaler": scaler, "labels": labels, "anomaly": labels == -1}


def local_outlier_factor(X, n_neighbors: int = 20, contamination: float = 0.05):
    scaled, scaler = _scaled(X)
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    labels = model.fit_predict(scaled)
    return {"model": model, "scaler": scaler, "labels": labels, "anomaly": labels == -1}
