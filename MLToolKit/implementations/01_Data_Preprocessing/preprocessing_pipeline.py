"""Column-wise preprocessing pipelines that fit on training data only."""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def numeric_pipeline(strategy: str = "median") -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=strategy)),
            ("scaler", StandardScaler()),
        ]
    )


def categorical_pipeline(strategy: str = "most_frequent") -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=strategy)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )


def build_column_transformer(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> ColumnTransformer:
    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline(), list(numeric_features)))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline(), list(categorical_features)))
    if not transformers:
        raise ValueError("Select at least one numeric or categorical feature")
    return ColumnTransformer(transformers=transformers)


def build_model_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    estimator,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_column_transformer(numeric_features, categorical_features)),
            ("model", estimator),
        ]
    )


def fit_model_pipeline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    estimator,
):
    """Fit a full preprocessing-plus-model pipeline on train and predict test."""
    features = list(numeric_features) + list(categorical_features)
    pipeline = build_model_pipeline(numeric_features, categorical_features, estimator)
    pipeline.fit(train[features], train[target])
    predictions = pipeline.predict(test[features])
    return pipeline, predictions, test[target].to_numpy()
