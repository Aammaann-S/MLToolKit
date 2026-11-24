"""Clean inconsistent text, dates, dtypes, ranges, and units."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd


def text_columns(df: pd.DataFrame, columns: Sequence[str] | None = None) -> list[str]:
    if columns is not None:
        return list(columns)
    selected = []
    for column in df.columns:
        if pd.api.types.is_string_dtype(df[column]) or df[column].dtype == object:
            selected.append(column)
    return selected


def strip_whitespace(df: pd.DataFrame, columns: Sequence[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    for column in text_columns(out, columns):
        out[column] = out[column].astype("string").str.strip()
    return out


def normalize_case(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    case: str = "lower",
) -> pd.DataFrame:
    if case not in {"lower", "upper"}:
        raise ValueError("case must be 'lower' or 'upper'")
    out = strip_whitespace(df, columns)
    for column in text_columns(out, columns):
        text = out[column].astype("string")
        out[column] = text.str.lower() if case == "lower" else text.str.upper()
    return out


def standardize_categories(
    df: pd.DataFrame,
    column: str,
    mapping: Mapping[Any, Any],
) -> pd.DataFrame:
    out = df.copy()
    out[column] = out[column].replace(mapping)
    return out


def convert_dates(
    df: pd.DataFrame,
    columns: Sequence[str],
    dayfirst: bool = False,
) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        out[column] = pd.to_datetime(out[column], errors="coerce", dayfirst=dayfirst)
    return out


def correct_dtypes(df: pd.DataFrame, dtype_map: Mapping[str, Any]) -> pd.DataFrame:
    return df.astype(dict(dtype_map))


def coerce_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def invalid_value_mask(df: pd.DataFrame, column: str, valid_values: Sequence[Any]) -> pd.Series:
    return ~df[column].isin(list(valid_values))


def range_mask(
    df: pd.DataFrame,
    column: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if minimum is not None:
        mask &= df[column] >= minimum
    if maximum is not None:
        mask &= df[column] <= maximum
    return mask


def convert_units(
    df: pd.DataFrame,
    column: str,
    factor: float,
    new_column: str | None = None,
) -> pd.DataFrame:
    """Multiply a numeric column by a conversion factor, for example miles to kilometres."""
    out = df.copy()
    destination = new_column or column
    out[destination] = pd.to_numeric(out[column], errors="coerce") * factor
    return out
