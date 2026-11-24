"""Create straightforward numeric and datetime features."""

from __future__ import annotations

import pandas as pd


def add_ratio(df: pd.DataFrame, numerator: str, denominator: str, name: str) -> pd.DataFrame:
    out = df.copy()
    out[name] = out[numerator] / out[denominator].replace(0, pd.NA)
    return out


def add_difference(df: pd.DataFrame, left: str, right: str, name: str) -> pd.DataFrame:
    out = df.copy()
    out[name] = out[left] - out[right]
    return out


def add_datetime_parts(df: pd.DataFrame, column: str, prefix: str | None = None) -> pd.DataFrame:
    out = df.copy()
    values = pd.to_datetime(out[column], errors="coerce")
    stem = prefix or column
    out[f"{stem}_year"] = values.dt.year
    out[f"{stem}_month"] = values.dt.month
    out[f"{stem}_day"] = values.dt.day
    out[f"{stem}_day_of_week"] = values.dt.day_of_week
    return out


def add_interaction(df: pd.DataFrame, left: str, right: str, name: str | None = None) -> pd.DataFrame:
    out = df.copy()
    out[name or f"{left}_x_{right}"] = out[left] * out[right]
    return out
