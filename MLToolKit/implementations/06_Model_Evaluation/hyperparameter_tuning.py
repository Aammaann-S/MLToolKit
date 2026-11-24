"""Grid search and randomized search.

Fit preprocessing inside the estimator pipeline. Searching over a model that
was already scaled on the full dataset leaks the validation folds.
"""

from __future__ import annotations

from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold


def _splitter(task: str, cv: int, random_state: int):
    if task == "classification":
        return StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    if task == "regression":
        return KFold(n_splits=cv, shuffle=True, random_state=random_state)
    raise ValueError("task must be 'classification' or 'regression'")


def grid_search(estimator, param_grid: dict, X, y, scoring: str, task: str = "classification", cv: int = 5, random_state: int = 42):
    search = GridSearchCV(
        estimator,
        param_grid,
        scoring=scoring,
        cv=_splitter(task, cv, random_state),
        n_jobs=1,
    )
    search.fit(X, y)
    return search


def random_search(
    estimator,
    param_distributions: dict,
    X,
    y,
    scoring: str,
    task: str = "classification",
    n_iter: int = 10,
    cv: int = 5,
    random_state: int = 42,
):
    search = RandomizedSearchCV(
        estimator,
        param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=_splitter(task, cv, random_state),
        random_state=random_state,
        n_jobs=1,
    )
    search.fit(X, y)
    return search
