"""Shared train/test runner for classification modules."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from toolkit_loader import load_impl

_evaluation = load_impl("implementations/04_Classification/classification_evaluation.py")


def train_classifier(model, X, y, test_size: float = 0.2, random_state: int = 42, stratify: bool = True):
    values = np.asarray(X)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    target = np.asarray(y)
    split_target = target if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        values,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=split_target,
    )
    model.fit(X_train, y_train)
    return _evaluation.classification_result(model, X_train, X_test, y_train, y_test)
