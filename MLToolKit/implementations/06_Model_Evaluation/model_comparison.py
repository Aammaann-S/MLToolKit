"""Compare estimators with the same cross-validation splitter."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from toolkit_loader import load_impl

_cv = load_impl("implementations/06_Model_Evaluation/cross_validation.py")


def compare_models(models: dict, X, y, scoring: str, task: str, cv: int = 5, random_state: int = 42):
    return _cv.cross_validation_table(models, X, y, scoring=scoring, task=task, cv=cv, random_state=random_state)
