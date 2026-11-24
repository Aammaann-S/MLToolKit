"""Principal component analysis fit on the training split only."""

from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca_transform(train: pd.DataFrame, test: pd.DataFrame | None = None, n_components: int = 2):
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train)
    pca = PCA(n_components=n_components, random_state=42)
    train_values = pca.fit_transform(scaled_train)
    columns = [f"pc{index + 1}" for index in range(n_components)]
    train_out = pd.DataFrame(train_values, columns=columns, index=train.index)
    if test is None:
        return train_out, pca, scaler
    test_out = pd.DataFrame(pca.transform(scaler.transform(test)), columns=columns, index=test.index)
    return train_out, test_out, pca, scaler
