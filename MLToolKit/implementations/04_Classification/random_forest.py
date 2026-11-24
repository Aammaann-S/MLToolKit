"""Random forest classification with a stratified split."""

from __future__ import annotations

import sys
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from implementations_path import load_sibling

_runner = load_sibling(__file__, "_runner.py")


def train_random_forest_classifier(
    X,
    y,
    n_estimators: int = 100,
    max_depth: int | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=1,
    )
    return _runner.train_classifier(model, X, y, test_size=test_size, random_state=random_state)
