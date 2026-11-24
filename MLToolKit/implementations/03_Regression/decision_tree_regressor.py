"""Decision tree regression fit on the training split only."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from toolkit_loader import load_impl

_evaluation = load_impl("implementations/03_Regression/regression_evaluation.py")


def train_decision_tree_regressor(
    X,
    y,
    max_depth: int | None = 4,
    test_size: float = 0.2,
    random_state: int = 42,
):
    values = np.asarray(X)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(
        values, y, test_size=test_size, random_state=random_state
    )
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return _evaluation.regression_result(model, X_train, X_test, y_train, y_test, y_pred)
