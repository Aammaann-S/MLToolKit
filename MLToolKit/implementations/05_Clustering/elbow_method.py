"""Elbow curve using K-Means inertia on scaled features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def elbow_inertias(X, k_values: range | list[int] = range(1, 11), random_state: int = 42) -> pd.DataFrame:
    values = np.asarray(X, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    scaled = StandardScaler().fit_transform(values)
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=int(k), n_init=10, random_state=random_state)
        model.fit(scaled)
        rows.append({"k": int(k), "inertia": float(model.inertia_)})
    return pd.DataFrame(rows)


def plot_elbow(X, k_values: range | list[int] = range(1, 11), random_state: int = 42):
    import matplotlib.pyplot as plt

    table = elbow_inertias(X, k_values=k_values, random_state=random_state)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(table["k"], table["inertia"], marker="o")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow method")
    fig.tight_layout()
    return fig, ax, table
