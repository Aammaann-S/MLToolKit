"""Classification metrics for binary and multiclass problems."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _positive_scores(model, X_test, y_test):
    y_test = np.asarray(y_test)
    classes = np.unique(y_test)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        if len(classes) == 2:
            if proba.shape[1] == 1:
                return proba[:, 0]
            positive = 1 if 1 in getattr(model, "classes_", []) else model.classes_[-1]
            index = list(model.classes_).index(positive)
            return proba[:, index]
        return proba
    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)
    return None


def classification_metrics(model, X_test, y_test) -> dict[str, object]:
    y_test = np.asarray(y_test)
    y_pred = model.predict(X_test)
    average = "binary" if len(np.unique(y_test)) == 2 else "weighted"
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average=average, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }
    scores = _positive_scores(model, X_test, y_test)
    if scores is not None and len(np.unique(y_test)) >= 2:
        try:
            if np.ndim(scores) == 1:
                metrics["roc_auc"] = float(roc_auc_score(y_test, scores))
            else:
                metrics["roc_auc"] = float(roc_auc_score(y_test, scores, multi_class="ovr"))
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None
    return metrics


def classification_result(model, X_train, X_test, y_train, y_test) -> dict[str, object]:
    y_pred = model.predict(X_test)
    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "metrics": classification_metrics(model, X_test, y_test),
    }
