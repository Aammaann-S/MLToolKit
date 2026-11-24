"""Descriptive statistics for numeric and grouped data."""

from __future__ import annotations

import pandas as pd


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number")
    summary = numeric.describe().T
    summary["missing"] = numeric.isna().sum()
    summary["skew"] = numeric.skew(numeric_only=True)
    summary["kurtosis"] = numeric.kurt(numeric_only=True)
    return summary


def column_mode(df: pd.DataFrame, column: str):
    mode = df[column].mode(dropna=True)
    if mode.empty:
        return None
    return mode.iloc[0]


def grouped_statistics(df: pd.DataFrame, by: str, column: str) -> pd.DataFrame:
    return (
        df.groupby(by, dropna=False)[column]
        .agg(count="count", mean="mean", median="median", std="std", minimum="min", maximum="max")
        .reset_index()
    )
