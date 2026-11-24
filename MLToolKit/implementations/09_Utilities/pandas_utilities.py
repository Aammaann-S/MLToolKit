"""Everyday Pandas operations used in practicals."""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def rename_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    return df.rename(columns=mapping)


def select_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return df.loc[:, list(columns)].copy()


def filter_rows(df: pd.DataFrame, query: str) -> pd.DataFrame:
    return df.query(query).copy()


def sort_frame(df: pd.DataFrame, by: str | list[str], ascending: bool = True) -> pd.DataFrame:
    return df.sort_values(by=by, ascending=ascending).reset_index(drop=True)


def merge_frames(left: pd.DataFrame, right: pd.DataFrame, on: str, how: str = "inner") -> pd.DataFrame:
    return left.merge(right, on=on, how=how)


def concat_rows(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(list(frames), ignore_index=True)
