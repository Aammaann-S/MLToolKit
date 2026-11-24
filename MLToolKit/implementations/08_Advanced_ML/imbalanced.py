"""Class-weighted logistic regression, plus optional SMOTE resampling."""

from __future__ import annotations

import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from implementations_path import load_sibling

_runner = load_sibling(__file__, "../04_Classification/_runner.py")


def train_balanced_logistic_regression(X, y, test_size: float = 0.2, random_state: int = 42):
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    return _runner.train_classifier(model, X, y, test_size=test_size, random_state=random_state)


def resample_smote(X_train, y_train, random_state: int = 42):
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise ImportError("Install imbalanced-learn with: pip install imbalanced-learn") from exc
    sampler = SMOTE(random_state=random_state)
    return sampler.fit_resample(X_train, y_train)
