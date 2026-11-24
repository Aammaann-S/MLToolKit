"""Decision tree classification with a stratified split."""

from __future__ import annotations

import sys
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from implementations_path import load_sibling

_runner = load_sibling(__file__, "_runner.py")


def train_decision_tree_classifier(
    X,
    y,
    max_depth: int | None = 4,
    test_size: float = 0.2,
    random_state: int = 42,
):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    return _runner.train_classifier(model, X, y, test_size=test_size, random_state=random_state)
