"""Confusion matrix values and a heatmap."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


def build_confusion_matrix(y_true, y_pred, labels=None) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)


def plot_confusion_matrix(y_true, y_pred, labels=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    matrix = build_confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    return fig, ax, matrix
