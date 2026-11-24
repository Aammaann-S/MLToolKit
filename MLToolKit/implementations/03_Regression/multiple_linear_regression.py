"""Multiple linear regression fit on the training split only."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from toolkit_loader import load_impl

_evaluation = load_impl("implementations/03_Regression/regression_evaluation.py")


def train_multiple_linear_regression(X, y, test_size: float = 0.2, random_state: int = 42):
    values = np.asarray(X)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("Multiple linear regression expects at least two features")
    X_train, X_test, y_train, y_test = train_test_split(
        values, y, test_size=test_size, random_state=random_state
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return _evaluation.regression_result(model, X_train, X_test, y_train, y_test, y_pred)
