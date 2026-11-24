"""Distribution checks for numeric columns."""

from __future__ import annotations

import pandas as pd
from scipy import stats


def distribution_table(df: pd.DataFrame, column: str) -> dict[str, float]:
    series = df[column].dropna()
    return {
        "count": float(series.count()),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=1)) if len(series) > 1 else 0.0,
        "min": float(series.min()),
        "q1": float(series.quantile(0.25)),
        "median": float(series.median()),
        "q3": float(series.quantile(0.75)),
        "max": float(series.max()),
        "skew": float(series.skew()),
        "kurtosis": float(series.kurt()),
    }


def normality_test(series: pd.Series) -> dict[str, float]:
    """Shapiro-Wilk test. Very large samples are reduced to 500 rows for the test limit."""
    clean = series.dropna().astype(float)
    if len(clean) < 3:
        raise ValueError("A normality test needs at least 3 observations")
    sample = clean.sample(n=min(len(clean), 500), random_state=42)
    statistic, p_value = stats.shapiro(sample)
    return {"statistic": float(statistic), "p_value": float(p_value)}
