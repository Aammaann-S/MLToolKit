"""Transform numeric features. Learned transformers are fit on training data only."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, PowerTransformer, QuantileTransformer


def _columns(df: pd.DataFrame, columns: Sequence[str] | None) -> list[str]:
    if columns is not None:
        return list(columns)
    numeric = df.select_dtypes(include="number").columns.tolist()
    if not numeric:
        raise ValueError("No numeric columns were selected")
    return numeric


def _apply_frame(df: pd.DataFrame, columns: Sequence[str], values: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out[list(columns)] = values
    return out


def log_transform(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    base: float | None = None,
) -> pd.DataFrame:
    """Apply log1p, or a chosen logarithm, to non-negative columns."""
    cols = _columns(df, columns)
    out = df.copy()
    for column in cols:
        values = pd.to_numeric(out[column], errors="coerce")
        if (values.dropna() < 0).any():
            raise ValueError(f"Column '{column}' has negative values and cannot be log-transformed")
        if base is None:
            out[column] = np.log1p(values)
        else:
            out[column] = np.log(values.clip(lower=1e-12)) / np.log(base)
    return out


def sqrt_transform(df: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    cols = _columns(df, columns)
    out = df.copy()
    for column in cols:
        values = pd.to_numeric(out[column], errors="coerce")
        if (values.dropna() < 0).any():
            raise ValueError(f"Column '{column}' has negative values and cannot use a square root")
        out[column] = np.sqrt(values)
    return out


def power_transform(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    columns: Sequence[str] | None = None,
    method: str = "yeo-johnson",
):
    cols = _columns(train, columns)
    transformer = PowerTransformer(method=method, standardize=True)
    train_values = transformer.fit_transform(train[cols])
    train_out = _apply_frame(train, cols, train_values)
    if test is None:
        return train_out, transformer
    test_out = _apply_frame(test, cols, transformer.transform(test[cols]))
    return train_out, test_out, transformer


def quantile_transform(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    columns: Sequence[str] | None = None,
    n_quantiles: int = 100,
    output_distribution: str = "uniform",
):
    cols = _columns(train, columns)
    quantiles = min(n_quantiles, len(train))
    transformer = QuantileTransformer(
        n_quantiles=quantiles,
        output_distribution=output_distribution,
        random_state=42,
    )
    train_out = _apply_frame(train, cols, transformer.fit_transform(train[cols]))
    if test is None:
        return train_out, transformer
    test_out = _apply_frame(test, cols, transformer.transform(test[cols]))
    return train_out, test_out, transformer


def bin_numeric(
    df: pd.DataFrame,
    column: str,
    bins: int | Sequence[float] = 4,
    labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Bin with explicit edges or a fixed count. Edges supplied by the caller are not learned."""
    out = df.copy()
    out[column] = pd.cut(out[column], bins=bins, labels=labels, include_lowest=True)
    return out


def discretize(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    columns: Sequence[str] | None = None,
    n_bins: int = 4,
    strategy: str = "quantile",
):
    """Fit bin edges on the training data, then transform train and test."""
    cols = _columns(train, columns)
    encoder = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
    train_out = _apply_frame(train, cols, encoder.fit_transform(train[cols]))
    if test is None:
        return train_out, encoder
    test_out = _apply_frame(test, cols, encoder.transform(test[cols]))
    return train_out, test_out, encoder
