"""Scale numeric features. Every scaler is fit on training data only."""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, RobustScaler, StandardScaler


def _scale(scaler, train: pd.DataFrame, test: pd.DataFrame | None, columns: Sequence[str] | None):
    cols = list(columns) if columns is not None else train.select_dtypes(include="number").columns.tolist()
    if not cols:
        raise ValueError("No numeric columns are available to scale")
    scaler.fit(train[cols])
    train_out = train.copy()
    train_out[cols] = scaler.transform(train[cols])
    if test is None:
        return train_out, scaler
    test_out = test.copy()
    test_out[cols] = scaler.transform(test[cols])
    return train_out, test_out, scaler


def standard_scale(train: pd.DataFrame, test: pd.DataFrame | None = None, columns: Sequence[str] | None = None):
    return _scale(StandardScaler(), train, test, columns)


def minmax_scale(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    columns: Sequence[str] | None = None,
    feature_range: tuple[float, float] = (0, 1),
):
    return _scale(MinMaxScaler(feature_range=feature_range), train, test, columns)


def robust_scale(train: pd.DataFrame, test: pd.DataFrame | None = None, columns: Sequence[str] | None = None):
    return _scale(RobustScaler(), train, test, columns)


def maxabs_scale(train: pd.DataFrame, test: pd.DataFrame | None = None, columns: Sequence[str] | None = None):
    return _scale(MaxAbsScaler(), train, test, columns)


def normalize_rows(df: pd.DataFrame, columns: Sequence[str] | None = None, norm: str = "l2"):
    """Scale each row independently. This transformer does not learn column statistics."""
    cols = list(columns) if columns is not None else df.select_dtypes(include="number").columns.tolist()
    normalizer = Normalizer(norm=norm)
    out = df.copy()
    out[cols] = normalizer.fit_transform(df[cols])
    return out, normalizer
