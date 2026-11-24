import numpy as np
from sklearn.datasets import make_classification

from MLToolKit.toolkit_loader import load_impl


def _data(n_classes=2, n_samples=180):
    features, target = make_classification(
        n_samples=n_samples,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=n_classes,
        random_state=42,
    )
    return features, target


def test_binary_classifiers_are_better_than_chance():
    features, target = _data()
    logistic = load_impl("implementations/04_Classification/logistic_regression.py")
    knn = load_impl("implementations/04_Classification/knn.py")
    bayes = load_impl("implementations/04_Classification/naive_bayes.py")
    tree = load_impl("implementations/04_Classification/decision_tree.py")
    forest = load_impl("implementations/04_Classification/random_forest.py")
    svm = load_impl("implementations/04_Classification/svm.py")
    boosting = load_impl("implementations/04_Classification/gradient_boosting.py")

    result = logistic.train_logistic_regression(features, target)
    assert result["metrics"]["accuracy"] > 0.7
    assert result["metrics"]["roc_auc"] > 0.7
    assert result["metrics"]["confusion_matrix"].shape == (2, 2)

    for trainer in (
        knn.train_knn_classifier,
        bayes.train_gaussian_nb,
        tree.train_decision_tree_classifier,
        forest.train_random_forest_classifier,
        svm.train_svc,
        svm.train_linear_svc,
        boosting.train_gradient_boosting_classifier,
    ):
        trained = trainer(features, target)
        assert trained["metrics"]["accuracy"] > 0.55

    non_negative = np.abs(features)
    multinomial = bayes.train_multinomial_nb(non_negative, target)
    assert multinomial["metrics"]["f1"] > 0.4
    binary = (features > 0).astype(int)
    bernoulli = bayes.train_bernoulli_nb(binary, target)
    assert bernoulli["metrics"]["accuracy"] > 0.5


def test_multiclass_logistic_regression():
    features, target = _data(n_classes=3, n_samples=180)
    logistic = load_impl("implementations/04_Classification/logistic_regression.py")
    result = logistic.train_multinomial_logistic_regression(features, target)
    assert result["metrics"]["confusion_matrix"].shape == (3, 3)
    assert result["metrics"]["accuracy"] > 0.6
