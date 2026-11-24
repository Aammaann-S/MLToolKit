"""Load datasets with Pandas and NumPy.

These helpers accept relative and absolute paths. Database and spreadsheet
readers use the normal library APIs directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

PathLike = str | Path


def load_csv(
    path: PathLike,
    sep: str = ",",
    encoding: str = "utf-8",
    header: int | None = 0,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a CSV or other delimited text file."""
    return pd.read_csv(path, sep=sep, encoding=encoding, header=header, **kwargs)


def load_excel(
    path: PathLike,
    sheet_name: str | int = 0,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read an Excel worksheet. Requires the openpyxl package for .xlsx files."""
    return pd.read_excel(path, sheet_name=sheet_name, **kwargs)


def load_json(path: PathLike, **kwargs: Any) -> pd.DataFrame:
    """Read a JSON file into a DataFrame."""
    return pd.read_json(path, **kwargs)


def load_sql(query: str, connection: Any, **kwargs: Any) -> pd.DataFrame:
    """Read a SQL query or table name through Pandas."""
    return pd.read_sql(query, connection, **kwargs)


def load_numpy_text(path: PathLike, delimiter: str = ",", **kwargs: Any) -> np.ndarray:
    """Load a delimited numeric text file with NumPy."""
    return np.loadtxt(path, delimiter=delimiter, **kwargs)


def load_numpy(path: PathLike, allow_pickle: bool = False) -> np.ndarray:
    """Load an array saved with ``numpy.save``."""
    return np.load(path, allow_pickle=allow_pickle)


def save_numpy(path: PathLike, values: np.ndarray) -> None:
    """Save an array as a ``.npy`` file."""
    np.save(path, values)


def load_multiple_csv(paths: Iterable[PathLike], **kwargs: Any) -> dict[str, pd.DataFrame]:
    """Load several CSV files, keyed by file stem."""
    frames: dict[str, pd.DataFrame] = {}
    for path in paths:
        frames[Path(path).stem] = load_csv(path, **kwargs)
    if not frames:
        raise ValueError("paths must contain at least one CSV file")
    return frames


def load_dataset(path: PathLike, **kwargs: Any) -> pd.DataFrame | np.ndarray:
    """Load a dataset by file extension.

    Supported suffixes: .csv, .tsv, .txt, .xlsx, .xls, .json, .npy.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return load_csv(path, **kwargs)
    if suffix == ".tsv":
        kwargs.setdefault("sep", "\t")
        return load_csv(path, **kwargs)
    if suffix == ".txt":
        return load_csv(path, **kwargs)
    if suffix in {".xlsx", ".xls"}:
        return load_excel(path, **kwargs)
    if suffix == ".json":
        return load_json(path, **kwargs)
    if suffix == ".npy":
        return load_numpy(path)
    raise ValueError(f"Unsupported dataset type '{suffix}'. Use csv, excel, json, or npy.")


def dataframe_from_records(records: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Build a DataFrame from an in-memory list of mappings."""
    return pd.DataFrame(list(records))
