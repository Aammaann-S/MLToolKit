"""Synthetic tables for examples, tests, and practical exams."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_classification, make_regression


def regression_frame(n_samples: int = 200, n_features: int = 3, noise: float = 10.0, random_state: int = 42) -> pd.DataFrame:
    features, target = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state,
    )
    frame = pd.DataFrame(features, columns=[f"feature_{index}" for index in range(n_features)])
    frame["target"] = target
    return frame


def classification_frame(
    n_samples: int = 200,
    n_features: int = 4,
    n_classes: int = 2,
    random_state: int = 42,
) -> pd.DataFrame:
    features, target = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(3, n_features),
        n_redundant=0,
        n_classes=n_classes,
        random_state=random_state,
    )
    frame = pd.DataFrame(features, columns=[f"feature_{index}" for index in range(n_features)])
    frame["target"] = target
    return frame


def cluster_frame(n_samples: int = 180, n_features: int = 2, centers: int = 3, random_state: int = 42) -> pd.DataFrame:
    features, labels = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        random_state=random_state,
        cluster_std=1.0,
    )
    frame = pd.DataFrame(features, columns=[f"feature_{index}" for index in range(n_features)])
    frame["cluster"] = labels
    return frame


def missing_frame(random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    frame = pd.DataFrame(
        {
            "age": rng.normal(35, 8, 40),
            "income": rng.normal(50000, 8000, 40),
            "city": rng.choice(["Pune", "Delhi", "pune", " Delhi "], 40),
        }
    )
    frame.loc[frame.sample(frac=0.15, random_state=random_state).index, "age"] = np.nan
    frame.loc[frame.sample(frac=0.1, random_state=random_state + 1).index, "income"] = np.nan
    return frame
