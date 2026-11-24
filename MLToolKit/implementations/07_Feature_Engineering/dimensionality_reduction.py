"""PCA and truncated SVD for reducing numeric matrices."""

from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler


def reduce_pca(train: pd.DataFrame, test: pd.DataFrame | None = None, n_components: int = 2):
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train)
    model = PCA(n_components=n_components, random_state=42)
    train_values = model.fit_transform(scaled_train)
    columns = [f"component_{index + 1}" for index in range(n_components)]
    train_out = pd.DataFrame(train_values, columns=columns, index=train.index)
    if test is None:
        return train_out, model, scaler
    test_out = pd.DataFrame(model.transform(scaler.transform(test)), columns=columns, index=test.index)
    return train_out, test_out, model, scaler


def reduce_truncated_svd(train: pd.DataFrame, test: pd.DataFrame | None = None, n_components: int = 2):
    model = TruncatedSVD(n_components=n_components, random_state=42)
    train_values = model.fit_transform(train)
    columns = [f"component_{index + 1}" for index in range(n_components)]
    train_out = pd.DataFrame(train_values, columns=columns, index=train.index)
    if test is None:
        return train_out, model
    test_out = pd.DataFrame(model.transform(test), columns=columns, index=test.index)
    return train_out, test_out, model
