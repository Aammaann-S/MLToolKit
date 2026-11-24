"""Regression metrics, including adjusted R-squared."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred, n_features: int | None = None) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mse = float(mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": r2,
    }
    if n_features is not None:
        n_samples = int(y_true.shape[0])
        if n_samples > n_features + 1:
            metrics["adjusted_r2"] = float(1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1))
        else:
            metrics["adjusted_r2"] = float("nan")
    return metrics


def regression_result(model, X_train, X_test, y_train, y_test, y_pred) -> dict[str, object]:
    n_features = int(np.asarray(X_train).shape[1])
    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "metrics": regression_metrics(y_test, y_pred, n_features=n_features),
    }
