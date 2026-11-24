"""Lag features and calendar features for forecasting tables."""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def add_lags(df: pd.DataFrame, column: str, lags: Sequence[int]) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        out[f"{column}_lag_{lag}"] = out[column].shift(lag)
    return out


def add_calendar_features(df: pd.DataFrame, column: str) -> pd.DataFrame:
    out = df.copy()
    values = pd.to_datetime(out[column], errors="coerce")
    out["year"] = values.dt.year
    out["month"] = values.dt.month
    out["day_of_week"] = values.dt.day_of_week
    return out
