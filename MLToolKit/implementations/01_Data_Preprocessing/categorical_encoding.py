"""Encode categorical features without fitting on the test set."""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder


def label_encode(train: pd.Series, test: pd.Series | None = None):
    """Label-encode one column. Unseen test categories become -1."""
    encoder = LabelEncoder()
    train_text = train.astype("string").fillna("missing")
    encoded_train = encoder.fit_transform(train_text)
    if test is None:
        return encoded_train, encoder
    classes = {label: index for index, label in enumerate(encoder.classes_)}
    test_text = test.astype("string").fillna("missing")
    encoded_test = test_text.map(classes).fillna(-1).astype(int)
    return encoded_train, encoded_test.to_numpy(), encoder


def ordinal_encode(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    columns: Sequence[str] | None = None,
    categories: Sequence[Sequence[str]] | None = None,
):
    cols = list(columns) if columns is not None else list(train.columns)
    encoder = OrdinalEncoder(
        categories=list(categories) if categories is not None else "auto",
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    encoded_train = encoder.fit_transform(train[cols].astype("string").fillna("missing"))
    train_out = train.copy()
    train_out[cols] = encoded_train
    if test is None:
        return train_out, encoder
    test_out = test.copy()
    test_out[cols] = encoder.transform(test[cols].astype("string").fillna("missing"))
    return train_out, test_out, encoder


def one_hot_encode(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    columns: Sequence[str] | None = None,
):
    """One-hot encode selected columns. Unknown test categories are ignored."""
    cols = list(columns) if columns is not None else list(train.columns)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    train_values = encoder.fit_transform(train[cols].astype("string").fillna("missing"))
    feature_names = encoder.get_feature_names_out(cols)
    train_encoded = pd.DataFrame(train_values, columns=feature_names, index=train.index)
    train_out = pd.concat([train.drop(columns=cols), train_encoded], axis=1)
    if test is None:
        return train_out, encoder
    test_values = encoder.transform(test[cols].astype("string").fillna("missing"))
    test_encoded = pd.DataFrame(test_values, columns=feature_names, index=test.index)
    test_out = pd.concat([test.drop(columns=cols), test_encoded], axis=1)
    return train_out, test_out, encoder


def get_dummies_aligned(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Create dummies from the training columns and align the test frame to them."""
    train_out = pd.get_dummies(train, columns=list(columns) if columns is not None else None)
    if test is None:
        return train_out
    test_out = pd.get_dummies(test, columns=list(columns) if columns is not None else None)
    test_out = test_out.reindex(columns=train_out.columns, fill_value=0)
    return train_out, test_out
