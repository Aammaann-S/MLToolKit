"""Silhouette scores for candidate cluster counts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def silhouette_scores(X, k_values: range | list[int] = range(2, 8), random_state: int = 42) -> pd.DataFrame:
    values = np.asarray(X, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    scaled = StandardScaler().fit_transform(values)
    rows = []
    for k in k_values:
        labels = KMeans(n_clusters=int(k), n_init=10, random_state=random_state).fit_predict(scaled)
        if len(np.unique(labels)) < 2:
            score = float("nan")
        else:
            score = float(silhouette_score(scaled, labels))
        rows.append({"k": int(k), "silhouette": score})
    return pd.DataFrame(rows)


def best_silhouette_k(X, k_values: range | list[int] = range(2, 8), random_state: int = 42) -> int:
    table = silhouette_scores(X, k_values=k_values, random_state=random_state)
    return int(table.loc[table["silhouette"].idxmax(), "k"])
