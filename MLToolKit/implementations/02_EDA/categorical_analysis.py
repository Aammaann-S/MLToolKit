"""Frequency and proportion tables for categorical columns."""

from __future__ import annotations

import pandas as pd


def frequency_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    counts = df[column].value_counts(dropna=False)
    proportions = df[column].value_counts(dropna=False, normalize=True)
    table = pd.DataFrame({"count": counts, "proportion": proportions})
    return table.rename_axis(column).reset_index()


def category_by_target(df: pd.DataFrame, column: str, target: str) -> pd.DataFrame:
    return pd.crosstab(df[column], df[target], normalize="index")
