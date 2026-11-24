"""Summaries, group-by aggregations, joins, concatenation, and pivot tables."""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def dataset_overview(df: pd.DataFrame) -> dict[str, object]:
    missing = df.isna().sum()
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(missing.sum()),
        "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
        "categorical_columns": df.select_dtypes(exclude="number").columns.tolist(),
    }


def groupby_aggregate(
    df: pd.DataFrame,
    by: str | Sequence[str],
    aggregations: dict[str, str | list[str]],
) -> pd.DataFrame:
    return df.groupby(by, dropna=False).agg(aggregations).reset_index()


def merge_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str | Sequence[str] | None = None,
    how: str = "inner",
    left_on: str | None = None,
    right_on: str | None = None,
) -> pd.DataFrame:
    return left.merge(right, on=on, how=how, left_on=left_on, right_on=right_on)


def join_on_index(left: pd.DataFrame, right: pd.DataFrame, how: str = "left") -> pd.DataFrame:
    return left.join(right, how=how, lsuffix="_left", rsuffix="_right")


def concatenate_rows(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(list(frames), axis=0, ignore_index=True)


def concatenate_columns(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(list(frames), axis=1)


def pivot_table(
    df: pd.DataFrame,
    index: str,
    columns: str,
    values: str,
    aggfunc: str = "mean",
) -> pd.DataFrame:
    return pd.pivot_table(df, index=index, columns=columns, values=values, aggfunc=aggfunc)


def cross_tab(df: pd.DataFrame, row: str, column: str, normalize: bool | str = False) -> pd.DataFrame:
    return pd.crosstab(df[row], df[column], normalize=normalize)
