"""Polynomial regression. Feature expansion and scaling are fit on training rows only."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from toolkit_loader import load_impl

_evaluation = load_impl("implementations/03_Regression/regression_evaluation.py")


def train_polynomial_regression(X, y, degree: int = 2, test_size: float = 0.2, random_state: int = 42):
    values = np.asarray(X)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(
        values, y, test_size=test_size, random_state=random_state
    )
    model = Pipeline(
        steps=[
            ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return _evaluation.regression_result(model, X_train, X_test, y_train, y_test, y_pred)
