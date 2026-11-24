"""Correlation matrices for numeric columns."""

from __future__ import annotations

import pandas as pd


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("method must be pearson, spearman, or kendall")
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        raise ValueError("Correlation needs at least two numeric columns")
    return numeric.corr(method=method)


def target_correlation(df: pd.DataFrame, target: str, method: str = "pearson") -> pd.Series:
    matrix = correlation_matrix(df, method=method)
    return matrix[target].drop(labels=[target]).sort_values(key=lambda values: values.abs(), ascending=False)
