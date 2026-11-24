"""Gradient boosting classification with a stratified split."""

from __future__ import annotations

import sys
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from implementations_path import load_sibling

_runner = load_sibling(__file__, "_runner.py")


def train_gradient_boosting_classifier(
    X,
    y,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    test_size: float = 0.2,
    random_state: int = 42,
):
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
    )
    return _runner.train_classifier(model, X, y, test_size=test_size, random_state=random_state)
