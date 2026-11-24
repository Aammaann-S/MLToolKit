"""Learning curves, validation curves, and residual summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve, validation_curve


def learning_curve_frame(estimator, X, y, cv: int = 5, scoring: str = "accuracy", random_state: int = 42) -> pd.DataFrame:
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        scoring=scoring,
        train_sizes=np.linspace(0.2, 1.0, 5),
        shuffle=True,
        random_state=random_state,
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "train_size": train_sizes,
            "train_mean": train_scores.mean(axis=1),
            "validation_mean": test_scores.mean(axis=1),
        }
    )


def validation_curve_frame(estimator, X, y, param_name: str, param_range, cv: int = 5, scoring: str = "accuracy") -> pd.DataFrame:
    train_scores, test_scores = validation_curve(
        estimator,
        X,
        y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "param": list(param_range),
            "train_mean": train_scores.mean(axis=1),
            "validation_mean": test_scores.mean(axis=1),
        }
    )


def residual_summary(y_true, y_pred) -> dict[str, float]:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    return {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "min": float(np.min(residuals)),
        "max": float(np.max(residuals)),
    }
