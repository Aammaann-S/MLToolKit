"""Gaussian, multinomial, and Bernoulli naive Bayes.

Multinomial naive Bayes expects non-negative features such as counts.
Bernoulli naive Bayes expects binary features.
"""

from __future__ import annotations

import sys
from pathlib import Path

from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from implementations_path import load_sibling

_runner = load_sibling(__file__, "_runner.py")


def train_gaussian_nb(X, y, test_size: float = 0.2, random_state: int = 42):
    return _runner.train_classifier(GaussianNB(), X, y, test_size=test_size, random_state=random_state)


def train_multinomial_nb(X, y, test_size: float = 0.2, random_state: int = 42):
    return _runner.train_classifier(MultinomialNB(), X, y, test_size=test_size, random_state=random_state)


def train_bernoulli_nb(X, y, test_size: float = 0.2, random_state: int = 42):
    return _runner.train_classifier(BernoulliNB(), X, y, test_size=test_size, random_state=random_state)
