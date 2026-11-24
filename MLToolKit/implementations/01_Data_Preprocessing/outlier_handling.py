"""Detect, remove, cap, and winsorize outliers using training-set limits."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize


def _numeric_columns(df: pd.DataFrame, columns: Sequence[str] | None) -> list[str]:
    if columns is not None:
        return list(columns)
    cols = df.select_dtypes(include="number").columns.tolist()
    if not cols:
        raise ValueError("No numeric columns are available")
    return cols


def iqr_bounds(
    train: pd.DataFrame,
    columns: Sequence[str] | None = None,
    factor: float = 1.5,
) -> tuple[pd.Series, pd.Series]:
    cols = _numeric_columns(train, columns)
    q1 = train[cols].quantile(0.25)
    q3 = train[cols].quantile(0.75)
    spread = q3 - q1
    return q1 - factor * spread, q3 + factor * spread


def zscore_parameters(
    train: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> tuple[pd.Series, pd.Series]:
    cols = _numeric_columns(train, columns)
    mean = train[cols].mean()
    std = train[cols].std(ddof=0).replace(0, np.nan)
    return mean, std


def iqr_mask(df: pd.DataFrame, lower: pd.Series, upper: pd.Series) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for column in lower.index:
        mask |= df[column] < lower[column]
        mask |= df[column] > upper[column]
    return mask


def zscore_mask(
    df: pd.DataFrame,
    mean: pd.Series,
    std: pd.Series,
    threshold: float = 3.0,
) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for column in mean.index:
        score = (df[column] - mean[column]) / std[column]
        mask |= score.abs() > threshold
    return mask


def remove_outliers(df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    return df.loc[~mask].reset_index(drop=True)


def cap_outliers(df: pd.DataFrame, lower: pd.Series, upper: pd.Series) -> pd.DataFrame:
    out = df.copy()
    for column in lower.index:
        out[column] = out[column].clip(lower=lower[column], upper=upper[column])
    return out


def winsorize_series(series: pd.Series, limits: tuple[float, float] = (0.05, 0.05)) -> pd.Series:
    """Winsorize one series. For supervised work, prefer ``clip_to_train_quantiles``."""
    clipped = winsorize(np.asarray(series, dtype=float), limits=limits)
    return pd.Series(np.asarray(clipped), index=series.index, name=series.name)


def clip_to_train_quantiles(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    columns: Sequence[str] | None = None,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
):
    cols = _numeric_columns(train, columns)
    lower = train[cols].quantile(lower_quantile)
    upper = train[cols].quantile(upper_quantile)
    train_out = cap_outliers(train, lower, upper)
    if test is None:
        return train_out, lower, upper
    return train_out, cap_outliers(test, lower, upper), lower, upper
