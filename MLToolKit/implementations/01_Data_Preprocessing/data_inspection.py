"""Inspect a Pandas DataFrame before modeling."""

from __future__ import annotations

import io
from typing import Any

import pandas as pd


def head(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return df.head(n)


def tail(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    return df.tail(n)


def info_text(df: pd.DataFrame) -> str:
    """Return the same report as ``DataFrame.info`` without printing it."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()


def describe(df: pd.DataFrame, include: Any = "all") -> pd.DataFrame:
    return df.describe(include=include)


def shape_of(df: pd.DataFrame) -> tuple[int, int]:
    return int(df.shape[0]), int(df.shape[1])


def column_names(df: pd.DataFrame) -> list[str]:
    return [str(column) for column in df.columns]


def dtypes_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.dtypes.rename("dtype").rename_axis("column").reset_index()


def value_counts(df: pd.DataFrame, column: str, dropna: bool = False) -> pd.Series:
    return df[column].value_counts(dropna=dropna)


def nunique(df: pd.DataFrame, dropna: bool = False) -> pd.Series:
    return df.nunique(dropna=dropna)


def missing_counts(df: pd.DataFrame) -> pd.Series:
    return df.isna().sum()


def missing_mask(df: pd.DataFrame) -> pd.DataFrame:
    return df.isna()


def memory_usage(df: pd.DataFrame, deep: bool = True) -> pd.Series:
    return df.memory_usage(deep=deep)


def sample_rows(df: pd.DataFrame, n: int = 5, random_state: int = 42) -> pd.DataFrame:
    count = min(n, len(df))
    return df.sample(n=count, random_state=random_state)


def inspection_report(df: pd.DataFrame) -> dict[str, Any]:
    """Collect the checks normally printed during a practical exam."""
    rows, columns = shape_of(df)
    missing = missing_counts(df)
    return {
        "rows": rows,
        "columns": columns,
        "column_names": column_names(df),
        "dtypes": {name: str(dtype) for name, dtype in df.dtypes.items()},
        "missing_counts": missing.to_dict(),
        "missing_percent": (missing / rows * 100 if rows else missing).round(2).to_dict(),
        "nunique": nunique(df).to_dict(),
        "memory_bytes": int(memory_usage(df).sum()),
    }
