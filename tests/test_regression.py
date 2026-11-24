import numpy as np
from sklearn.datasets import make_regression

from MLToolKit.toolkit_loader import load_impl


def _data():
    features, target = make_regression(n_samples=160, n_features=3, noise=12.0, random_state=42)
    return features, target


def test_linear_models_beat_a_weak_baseline():
    features, target = _data()
    simple_features, simple_target = make_regression(n_samples=160, n_features=1, noise=12.0, random_state=42)
    simple = load_impl("implementations/03_Regression/simple_linear_regression.py")
    multiple = load_impl("implementations/03_Regression/multiple_linear_regression.py")
    polynomial = load_impl("implementations/03_Regression/polynomial_regression.py")
    ridge = load_impl("implementations/03_Regression/ridge_regression.py")
    lasso = load_impl("implementations/03_Regression/lasso_regression.py")
    elastic = load_impl("implementations/03_Regression/elasticnet_regression.py")
    tree = load_impl("implementations/03_Regression/decision_tree_regressor.py")
    forest = load_impl("implementations/03_Regression/random_forest_regressor.py")

    single = simple.train_simple_linear_regression(simple_features, simple_target)
    assert single["metrics"]["r2"] > 0.4
    assert "adjusted_r2" in single["metrics"]
    for trainer in (
        multiple.train_multiple_linear_regression,
        polynomial.train_polynomial_regression,
        ridge.train_ridge_regression,
        lasso.train_lasso_regression,
        elastic.train_elasticnet_regression,
        tree.train_decision_tree_regressor,
        forest.train_random_forest_regressor,
    ):
        result = trainer(features, target)
        assert result["y_pred"].shape == result["y_test"].shape
        assert np.isfinite(result["metrics"]["rmse"])
