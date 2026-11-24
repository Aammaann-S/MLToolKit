"""Simple linear regression: one predictor, fit after the train/test split."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from toolkit_loader import load_impl

_evaluation = load_impl("implementations/03_Regression/regression_evaluation.py")


def _column(X):
    values = np.asarray(X)
    if values.ndim == 1:
        return values.reshape(-1, 1)
    if values.shape[1] != 1:
        raise ValueError("Simple linear regression expects exactly one feature")
    return values


def train_simple_linear_regression(X, y, test_size: float = 0.2, random_state: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        _column(X), y, test_size=test_size, random_state=random_state
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return _evaluation.regression_result(model, X_train, X_test, y_train, y_test, y_pred)
