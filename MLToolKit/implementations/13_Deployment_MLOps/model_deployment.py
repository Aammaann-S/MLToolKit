"""Train, save, load, and score a model without a web framework."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_deployable_classifier(X, y, random_state: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )
    model.fit(X_train, y_train)
    return {"model": model, "X_test": X_test, "y_test": y_test, "y_pred": model.predict(X_test)}


def save_model(model, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    return destination


def load_model(path: str | Path):
    return joblib.load(path)


def predict_records(model, records) -> list:
    values = np.asarray(records, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return model.predict(values).tolist()
