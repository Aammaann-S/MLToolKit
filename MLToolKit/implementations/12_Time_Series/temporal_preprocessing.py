"""Time-aware preprocessing. Aggregations do not use future rows."""

from __future__ import annotations

import pandas as pd


def parse_datetime(df: pd.DataFrame, column: str) -> pd.DataFrame:
    out = df.copy()
    out[column] = pd.to_datetime(out[column], errors="coerce")
    return out.sort_values(column).reset_index(drop=True)


def set_time_index(df: pd.DataFrame, column: str) -> pd.DataFrame:
    out = parse_datetime(df, column)
    return out.set_index(column)


def resample_frame(df: pd.DataFrame, rule: str, column: str, agg: str = "mean") -> pd.DataFrame:
    series = df[column].resample(rule).agg(agg)
    return series.to_frame(name=column)


def rolling_features(df: pd.DataFrame, column: str, window: int = 3) -> pd.DataFrame:
    """Rolling statistics of past values. The current row is excluded with shift(1)."""
    out = df.copy()
    past = out[column].shift(1)
    out[f"{column}_roll_mean_{window}"] = past.rolling(window).mean()
    out[f"{column}_roll_std_{window}"] = past.rolling(window).std()
    return out
