"""Gaussian mixture models and spectral clustering."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def _scaled(X):
    values = np.asarray(X, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    scaler = StandardScaler()
    return scaler.fit_transform(values), scaler


def gaussian_mixture(X, n_components: int = 3, random_state: int = 42):
    scaled, scaler = _scaled(X)
    model = GaussianMixture(n_components=n_components, random_state=random_state)
    labels = model.fit_predict(scaled)
    return {"model": model, "scaler": scaler, "labels": labels}


def spectral_clustering(X, n_clusters: int = 3, random_state: int = 42):
    scaled, scaler = _scaled(X)
    model = SpectralClustering(n_clusters=n_clusters, random_state=random_state, affinity="nearest_neighbors")
    labels = model.fit_predict(scaled)
    return {"model": model, "scaler": scaler, "labels": labels}
