"""Scatter plots of cluster assignments."""

from __future__ import annotations

import numpy as np
import pandas as pd


def plot_clusters(X, labels, x_index: int = 0, y_index: int = 1):
    import matplotlib.pyplot as plt

    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("Cluster visualization needs at least two features")
    fig, ax = plt.subplots(figsize=(8, 5))
    frame = pd.DataFrame({"x": values[:, x_index], "y": values[:, y_index], "cluster": labels})
    for cluster, group in frame.groupby("cluster"):
        ax.scatter(group["x"], group["y"], label=str(cluster), s=30)
    ax.set_xlabel(f"Feature {x_index}")
    ax.set_ylabel(f"Feature {y_index}")
    ax.set_title("Cluster assignments")
    ax.legend(title="Cluster")
    fig.tight_layout()
    return fig, ax
