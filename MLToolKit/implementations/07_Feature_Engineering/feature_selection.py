"""DataFrame feature selection. Selectors are fit on the training rows only."""

from __future__ import annotations

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif


def select_k_best_columns(train: pd.DataFrame, target: str, k: int = 5, score_func=f_classif):
    features = [column for column in train.columns if column != target]
    numeric = train[features].select_dtypes(include="number")
    if numeric.shape[1] == 0:
        raise ValueError("Feature selection needs numeric predictors")
    k_used = min(k, numeric.shape[1])
    selector = SelectKBest(score_func=score_func, k=k_used)
    selector.fit(numeric, train[target])
    chosen = numeric.columns[selector.get_support()].tolist()
    return chosen, selector


def apply_selected_columns(train: pd.DataFrame, test: pd.DataFrame, columns: list[str], target: str):
    return train[columns + [target]].copy(), test[columns + [target]].copy()
