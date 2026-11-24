"""Gradient boosting plus optional XGBoost, LightGBM, and CatBoost trainers."""

from __future__ import annotations

import sys
from pathlib import Path

from sklearn.ensemble import GradientBoostingClassifier

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from implementations_path import load_sibling

_runner = load_sibling(__file__, "../04_Classification/_runner.py")


def train_gradient_boosting(X, y, test_size: float = 0.2, random_state: int = 42):
    model = GradientBoostingClassifier(random_state=random_state)
    return _runner.train_classifier(model, X, y, test_size=test_size, random_state=random_state)


def _optional(import_name: str, package: str):
    try:
        return __import__(import_name)
    except ImportError as exc:
        raise ImportError(f"Install the optional dependency with: pip install {package}") from exc


def train_xgboost_classifier(X, y, test_size: float = 0.2, random_state: int = 42):
    xgb = _optional("xgboost", "xgboost")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=random_state,
        n_jobs=1,
        eval_metric="logloss",
    )
    return _runner.train_classifier(model, X, y, test_size=test_size, random_state=random_state)


def train_lightgbm_classifier(X, y, test_size: float = 0.2, random_state: int = 42):
    lgb = _optional("lightgbm", "lightgbm")
    model = lgb.LGBMClassifier(n_estimators=100, random_state=random_state, verbose=-1)
    return _runner.train_classifier(model, X, y, test_size=test_size, random_state=random_state)


def train_catboost_classifier(X, y, test_size: float = 0.2, random_state: int = 42):
    catboost = _optional("catboost", "catboost")
    model = catboost.CatBoostClassifier(iterations=100, depth=4, verbose=0, random_state=random_state)
    return _runner.train_classifier(model, X, y, test_size=test_size, random_state=random_state)
