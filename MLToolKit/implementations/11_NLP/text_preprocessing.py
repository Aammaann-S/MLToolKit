"""Text cleaning that uses the Python standard library and Pandas."""

from __future__ import annotations

import re

import pandas as pd

_URL = re.compile(r"https?://\S+|www\.\S+")
_NON_WORD = re.compile(r"[^a-z0-9\s]")
_SPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    lowered = str(text).lower()
    without_urls = _URL.sub(" ", lowered)
    without_punctuation = _NON_WORD.sub(" ", without_urls)
    return _SPACE.sub(" ", without_punctuation).strip()


def tokenize(text: str) -> list[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    return cleaned.split(" ")


def clean_text_column(df: pd.DataFrame, column: str, new_column: str = "clean_text") -> pd.DataFrame:
    out = df.copy()
    out[new_column] = out[column].map(clean_text)
    return out


def ngrams(tokens: list[str], n: int = 2) -> list[str]:
    if n < 1:
        raise ValueError("n must be at least 1")
    return [" ".join(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
