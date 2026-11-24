"""Save and load fitted models with joblib or pickle."""

from __future__ import annotations

import pickle
from pathlib import Path

import joblib


def save_joblib(model, path: str | Path) -> None:
    joblib.dump(model, path)


def load_joblib(path: str | Path):
    return joblib.load(path)


def save_pickle(model, path: str | Path) -> None:
    with open(path, "wb") as handle:
        pickle.dump(model, handle)


def load_pickle(path: str | Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)
