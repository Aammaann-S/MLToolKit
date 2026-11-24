"""Small helpers shared by workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def as_frame(values, columns: list[str] | None = None) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        return values.copy()
    array = np.asarray(values)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    names = columns or [f"feature_{index}" for index in range(array.shape[1])]
    return pd.DataFrame(array, columns=names)
