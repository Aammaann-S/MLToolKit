"""Small NumPy helpers for exam-style numeric work."""

from __future__ import annotations

import numpy as np


def column_means(values) -> np.ndarray:
    return np.asarray(values, dtype=float).mean(axis=0)


def column_std(values) -> np.ndarray:
    return np.asarray(values, dtype=float).std(axis=0)


def reshape_column(values) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def train_mean_scale(train, test=None):
    """Standardize with the training mean and standard deviation."""
    train = np.asarray(train, dtype=float)
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std == 0, 1, std)
    scaled_train = (train - mean) / std
    if test is None:
        return scaled_train, mean, std
    scaled_test = (np.asarray(test, dtype=float) - mean) / std
    return scaled_train, scaled_test, mean, std
