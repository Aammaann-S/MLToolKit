"""Naive forecast and an ARIMA fit through statsmodels."""

from __future__ import annotations

import numpy as np
import pandas as pd


def naive_forecast(series, horizon: int = 1) -> np.ndarray:
    values = pd.Series(series).dropna()
    if values.empty:
        raise ValueError("series has no observations")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    return np.repeat(float(values.iloc[-1]), horizon)


def forecast_errors(y_true, y_pred) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    error = actual - predicted
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


def fit_arima(series, order: tuple[int, int, int] = (1, 1, 1)):
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as exc:
        raise ImportError("Install statsmodels to fit ARIMA") from exc
    model = ARIMA(pd.Series(series, dtype=float), order=order)
    return model.fit()
