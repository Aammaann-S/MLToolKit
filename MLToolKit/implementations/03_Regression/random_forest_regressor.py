"""Random forest regression fit on the training split only."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from toolkit_loader import load_impl

_evaluation = load_impl("implementations/03_Regression/regression_evaluation.py")


def train_random_forest_regressor(
    X,
    y,
    n_estimators: int = 100,
    max_depth: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    values = np.asarray(X)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(
        values, y, test_size=test_size, random_state=random_state
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return _evaluation.regression_result(model, X_train, X_test, y_train, y_test, y_pred)
