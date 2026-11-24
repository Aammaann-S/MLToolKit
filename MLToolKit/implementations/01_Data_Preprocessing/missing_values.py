"""Detect and impute missing values.

Statistics are learned from the training frame only. The test frame is
transformed with those frozen values.
"""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.impute import MissingIndicator


def missing_counts(df: pd.DataFrame) -> pd.Series:
    return df.isna().sum()


def missing_percentages(df: pd.DataFrame) -> pd.Series:
    if len(df) == 0:
        return pd.Series(0.0, index=df.columns)
    return df.isna().mean() * 100


def drop_missing(
    df: pd.DataFrame,
    axis: int = 0,
    how: str = "any",
    subset: Sequence[str] | None = None,
    thresh: int | None = None,
) -> pd.DataFrame:
    return df.dropna(axis=axis, how=how, subset=subset, thresh=thresh)


def fill_missing(df: pd.DataFrame, value: Any, columns: Sequence[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    if columns is None:
        return out.fillna(value)
    out[list(columns)] = out[list(columns)].fillna(value)
    return out


def _numeric_columns(df: pd.DataFrame, columns: Sequence[str] | None) -> list[str]:
    if columns is not None:
        return list(columns)
    numeric = df.select_dtypes(include="number").columns.tolist()
    if not numeric:
        raise ValueError("No numeric columns are available to impute")
    return numeric


def _impute_with_train_statistic(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    statistics: pd.Series,
    columns: Sequence[str],
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    train_out = train.copy()
    train_out[list(columns)] = train_out[list(columns)].fillna(statistics)
    if test is None:
        return train_out
    test_out = test.copy()
    test_out[list(columns)] = test_out[list(columns)].fillna(statistics)
    return train_out, test_out


def impute_mean(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    columns: Sequence[str] | None = None,
) -> tuple[Any, ...]:
    cols = _numeric_columns(train, columns)
    means = train[cols].mean()
    imputed = _impute_with_train_statistic(train, test, means, cols)
    if test is None:
        return imputed, means
    return imputed[0], imputed[1], means


def impute_median(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    columns: Sequence[str] | None = None,
) -> tuple[Any, ...]:
    cols = _numeric_columns(train, columns)
    medians = train[cols].median()
    imputed = _impute_with_train_statistic(train, test, medians, cols)
    if test is None:
        return imputed, medians
    return imputed[0], imputed[1], medians


def impute_mode(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    columns: Sequence[str] | None = None,
) -> tuple[Any, ...]:
    cols = list(columns) if columns is not None else list(train.columns)
    modes: dict[str, Any] = {}
    for column in cols:
        mode = train[column].mode(dropna=True)
        if mode.empty:
            raise ValueError(f"Column '{column}' has no mode in the training data")
        modes[column] = mode.iloc[0]
    statistics = pd.Series(modes)
    imputed = _impute_with_train_statistic(train, test, statistics, cols)
    if test is None:
        return imputed, statistics
    return imputed[0], imputed[1], statistics


def impute_constant(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    value: Any = 0,
    columns: Sequence[str] | None = None,
) -> tuple[Any, ...]:
    cols = list(columns) if columns is not None else list(train.columns)
    filled_train = fill_missing(train, value, cols)
    if test is None:
        return filled_train, value
    return filled_train, fill_missing(test, value, cols), value


def forward_fill(df: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Forward-fill inside one already-split frame."""
    out = df.copy()
    if columns is None:
        return out.ffill()
    out[list(columns)] = out[list(columns)].ffill()
    return out


def backward_fill(df: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Backward-fill inside one already-split frame."""
    out = df.copy()
    if columns is None:
        return out.bfill()
    out[list(columns)] = out[list(columns)].bfill()
    return out


def forward_fill_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill each split independently, then fill a leading test gap from the last train row."""
    train_out = forward_fill(train, columns)
    test_out = forward_fill(test, columns)
    cols = list(columns) if columns is not None else list(train.columns)
    if len(train_out):
        test_out[cols] = test_out[cols].fillna(train_out[cols].iloc[-1])
    return train_out, test_out


def backward_fill_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backward-fill each split independently so test rows do not fill the training set."""
    return backward_fill(train, columns), backward_fill(test, columns)


def simple_impute(
    X_train,
    X_test=None,
    strategy: str = "mean",
    fill_value: Any = None,
):
    """Fit ``SimpleImputer`` on training data and transform the test data."""
    imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
    transformed_train = imputer.fit_transform(X_train)
    if X_test is None:
        return transformed_train, imputer
    return transformed_train, imputer.transform(X_test), imputer


def knn_impute(X_train, X_test=None, n_neighbors: int = 5):
    """Fit ``KNNImputer`` on training data only."""
    imputer = KNNImputer(n_neighbors=n_neighbors)
    transformed_train = imputer.fit_transform(X_train)
    if X_test is None:
        return transformed_train, imputer
    return transformed_train, imputer.transform(X_test), imputer


def add_missing_indicator(X_train, X_test=None):
    """Add missingness flags learned from the columns that exist in training data."""
    indicator = MissingIndicator(features="all")
    train_flags = indicator.fit_transform(X_train)
    if X_test is None:
        return train_flags, indicator
    return train_flags, indicator.transform(X_test), indicator
