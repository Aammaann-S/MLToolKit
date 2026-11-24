"""Cross-validation helpers. Pass a Pipeline when preprocessing learns from the data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score


def cross_validate(model, X, y, scoring: str = "accuracy", cv: int = 5, task: str = "classification", random_state: int = 42):
    if task == "classification":
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    elif task == "regression":
        splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    else:
        raise ValueError("task must be 'classification' or 'regression'")
    scores = cross_val_score(model, X, y, scoring=scoring, cv=splitter)
    return {
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
    }


def cross_validation_table(models: dict, X, y, scoring: str, task: str, cv: int = 5, random_state: int = 42) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        result = cross_validate(model, X, y, scoring=scoring, cv=cv, task=task, random_state=random_state)
        rows.append({"model": name, "mean": result["mean"], "std": result["std"]})
    return pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
