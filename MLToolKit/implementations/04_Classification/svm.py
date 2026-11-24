"""Support vector classification with a stratified split."""

from __future__ import annotations

import sys
from pathlib import Path

from sklearn.svm import LinearSVC, SVC

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from implementations_path import load_sibling

_runner = load_sibling(__file__, "_runner.py")


def train_svc(X, y, kernel: str = "rbf", C: float = 1.0, test_size: float = 0.2, random_state: int = 42):
    model = SVC(kernel=kernel, C=C, random_state=random_state)
    return _runner.train_classifier(model, X, y, test_size=test_size, random_state=random_state)


def train_linear_svc(X, y, C: float = 1.0, test_size: float = 0.2, random_state: int = 42):
    model = LinearSVC(C=C, random_state=random_state, max_iter=5000)
    return _runner.train_classifier(model, X, y, test_size=test_size, random_state=random_state)
