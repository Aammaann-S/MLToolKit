"""Split data for supervised learning and cross-validation."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit, train_test_split


def split_train_test(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def split_stratified(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def split_train_test_val(X, y, test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42, stratify: bool = False):
    """Split into train, validation, and test. Validation is taken from the training portion."""
    stratify_y = y if stratify else None
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_y
    )
    relative_val = val_size / (1 - test_size)
    stratify_train = y_train_val if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val,
        random_state=random_state,
        stratify=stratify_train,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def kfold_splits(X, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
    splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    return list(splitter.split(X))


def stratified_kfold_splits(X, y, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    return list(splitter.split(X, y))


def time_series_splits(X, n_splits: int = 5):
    splitter = TimeSeriesSplit(n_splits=n_splits)
    return list(splitter.split(X))


def features_and_target(df, target: str, features: list[str] | None = None):
    predictors = features if features is not None else [column for column in df.columns if column != target]
    return df[predictors], df[target]
