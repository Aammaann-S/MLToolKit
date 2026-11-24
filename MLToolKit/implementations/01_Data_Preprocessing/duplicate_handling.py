"""Find and remove duplicate records."""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def duplicated_mask(
    df: pd.DataFrame,
    subset: Sequence[str] | None = None,
    keep: str | bool = "first",
) -> pd.Series:
    return df.duplicated(subset=subset, keep=keep)


def find_duplicates(
    df: pd.DataFrame,
    subset: Sequence[str] | None = None,
    keep: str | bool = "first",
) -> pd.DataFrame:
    return df.loc[duplicated_mask(df, subset=subset, keep=keep)].copy()


def drop_duplicate_rows(
    df: pd.DataFrame,
    subset: Sequence[str] | None = None,
    keep: str | bool = "first",
) -> pd.DataFrame:
    return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)


def drop_duplicates_keep_first(df: pd.DataFrame, subset: Sequence[str] | None = None) -> pd.DataFrame:
    return drop_duplicate_rows(df, subset=subset, keep="first")


def drop_duplicates_keep_last(df: pd.DataFrame, subset: Sequence[str] | None = None) -> pd.DataFrame:
    return drop_duplicate_rows(df, subset=subset, keep="last")
