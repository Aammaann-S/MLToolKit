"""Polynomial feature expansion fit inside a training workflow."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


def polynomial_features(train: pd.DataFrame, test: pd.DataFrame | None = None, degree: int = 2):
    encoder = PolynomialFeatures(degree=degree, include_bias=False)
    train_values = encoder.fit_transform(train)
    names = encoder.get_feature_names_out(list(train.columns))
    train_out = pd.DataFrame(train_values, columns=names, index=train.index)
    if test is None:
        return train_out, encoder
    test_out = pd.DataFrame(encoder.transform(test), columns=names, index=test.index)
    return train_out, test_out, encoder
