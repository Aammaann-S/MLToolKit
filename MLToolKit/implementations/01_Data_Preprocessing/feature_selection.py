"""Select features with selectors fit on the training split only."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE, SelectKBest, VarianceThreshold, chi2, f_classif, mutual_info_classif
from sklearn.linear_model import LinearRegression, LogisticRegression


def variance_threshold(X_train, X_test=None, threshold: float = 0.0):
    selector = VarianceThreshold(threshold=threshold)
    transformed_train = selector.fit_transform(X_train)
    if X_test is None:
        return transformed_train, selector
    return transformed_train, selector.transform(X_test), selector


def select_k_best(X_train, y_train, X_test=None, k: int = 5, score_func=f_classif):
    selector = SelectKBest(score_func=score_func, k=k)
    transformed_train = selector.fit_transform(X_train, y_train)
    if X_test is None:
        return transformed_train, selector
    return transformed_train, selector.transform(X_test), selector


def select_anova(X_train, y_train, X_test=None, k: int = 5):
    return select_k_best(X_train, y_train, X_test, k=k, score_func=f_classif)


def select_chi2(X_train, y_train, X_test=None, k: int = 5):
    frame = pd.DataFrame(X_train)
    if (frame.select_dtypes(include="number").lt(0).any()).any():
        raise ValueError("Chi-square selection requires non-negative features")
    return select_k_best(X_train, y_train, X_test, k=k, score_func=chi2)


def select_mutual_info(X_train, y_train, X_test=None, k: int = 5, random_state: int = 42):
    def score_func(features, target):
        return mutual_info_classif(features, target, random_state=random_state)

    return select_k_best(X_train, y_train, X_test, k=k, score_func=score_func)


def rfe_select(X_train, y_train, X_test=None, n_features: int = 5, task: str = "classification"):
    if task == "classification":
        estimator = LogisticRegression(max_iter=1000)
    elif task == "regression":
        estimator = LinearRegression()
    else:
        raise ValueError("task must be 'classification' or 'regression'")
    selector = RFE(estimator, n_features_to_select=n_features)
    transformed_train = selector.fit_transform(X_train, y_train)
    if X_test is None:
        return transformed_train, selector
    return transformed_train, selector.transform(X_test), selector


def tree_importance(X_train, y_train, task: str = "classification", random_state: int = 42) -> pd.DataFrame:
    if task == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    elif task == "regression":
        model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    else:
        raise ValueError("task must be 'classification' or 'regression'")
    model.fit(X_train, y_train)
    columns = list(getattr(X_train, "columns", range(model.n_features_in_)))
    importance = pd.DataFrame({"feature": columns, "importance": model.feature_importances_})
    return importance.sort_values("importance", ascending=False).reset_index(drop=True)
