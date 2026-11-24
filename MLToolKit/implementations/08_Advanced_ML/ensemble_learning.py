"""Voting, bagging, AdaBoost, and extra trees."""

from __future__ import annotations

import sys
from pathlib import Path

from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from implementations_path import load_sibling

_runner = load_sibling(__file__, "../04_Classification/_runner.py")


def train_voting_classifier(X, y, test_size: float = 0.2, random_state: int = 42):
    model = VotingClassifier(
        estimators=[
            ("logistic", LogisticRegression(max_iter=1000, random_state=random_state)),
            ("tree", DecisionTreeClassifier(max_depth=4, random_state=random_state)),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
        ],
        voting="hard",
    )
    return _runner.train_classifier(model, X, y, test_size=test_size, random_state=random_state)


def train_bagging_classifier(X, y, test_size: float = 0.2, random_state: int = 42):
    model = BaggingClassifier(n_estimators=50, random_state=random_state, n_jobs=1)
    return _runner.train_classifier(model, X, y, test_size=test_size, random_state=random_state)


def train_adaboost_classifier(X, y, test_size: float = 0.2, random_state: int = 42):
    model = AdaBoostClassifier(n_estimators=50, random_state=random_state)
    return _runner.train_classifier(model, X, y, test_size=test_size, random_state=random_state)


def train_extra_trees_classifier(X, y, test_size: float = 0.2, random_state: int = 42):
    model = ExtraTreesClassifier(n_estimators=100, random_state=random_state, n_jobs=1)
    return _runner.train_classifier(model, X, y, test_size=test_size, random_state=random_state)
