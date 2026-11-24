"""Logistic regression with a stratified train/test split."""

from __future__ import annotations

import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from implementations_path import load_sibling

_runner = load_sibling(__file__, "_runner.py")


def train_logistic_regression(X, y, max_iter: int = 1000, test_size: float = 0.2, random_state: int = 42):
    model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    return _runner.train_classifier(model, X, y, test_size=test_size, random_state=random_state)


def train_multinomial_logistic_regression(X, y, max_iter: int = 1000, test_size: float = 0.2, random_state: int = 42):
    """Multiclass logistic regression. The solver handles the multinomial objective."""
    return train_logistic_regression(X, y, max_iter=max_iter, test_size=test_size, random_state=random_state)
