from pathlib import Path

from MLToolKit.toolkit_loader import load_impl


def test_tuning_comparison_persistence_and_diagnostics(tmp_path: Path):
    data = load_impl("implementations/09_Utilities/synthetic_datasets.py")
    tuning = load_impl("implementations/06_Model_Evaluation/hyperparameter_tuning.py")
    comparison = load_impl("implementations/06_Model_Evaluation/model_comparison.py")
    persistence = load_impl("implementations/06_Model_Evaluation/model_persistence.py")
    diagnostics = load_impl("implementations/06_Model_Evaluation/model_diagnostics.py")
    matrix = load_impl("implementations/06_Model_Evaluation/confusion_matrix.py")
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    frame = data.classification_frame(n_samples=120)
    features = frame.drop(columns=["target"])
    target = frame["target"]
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier()),
    ])
    search = tuning.grid_search(
        pipeline,
        {"model__n_neighbors": [3, 5]},
        features,
        target,
        scoring="accuracy",
        cv=3,
    )
    assert search.best_params_["model__n_neighbors"] in {3, 5}
    table = comparison.compare_models(
        {
            "logistic": LogisticRegression(max_iter=500),
            "knn": pipeline,
        },
        features,
        target,
        scoring="accuracy",
        task="classification",
        cv=3,
    )
    assert set(table["model"]) == {"logistic", "knn"}
    curve = diagnostics.learning_curve_frame(LogisticRegression(max_iter=500), features, target, cv=3)
    assert len(curve) == 5

    model = LogisticRegression(max_iter=500).fit(features, target)
    joblib_path = tmp_path / "model.joblib"
    pickle_path = tmp_path / "model.pkl"
    persistence.save_joblib(model, joblib_path)
    persistence.save_pickle(model, pickle_path)
    assert persistence.load_joblib(joblib_path).predict(features.iloc[:2]).shape == (2,)
    assert persistence.load_pickle(pickle_path).predict(features.iloc[:2]).shape == (2,)
    fig, _ax, values = matrix.plot_confusion_matrix(target, model.predict(features))
    assert values.shape == (2, 2)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_workflows_execute():
    from workflows.classification_workflow import main as classify
    from workflows.clustering_workflow import main as cluster
    from workflows.complete_preprocessing_workflow import main as preprocess
    from workflows.regression_workflow import main as regress

    assert regress()["metrics"]["r2"] > 0.5
    assert classify()["metrics"]["accuracy"] > 0.7
    assert len(cluster()["labels"]) > 0
    train, test = preprocess()
    assert len(train) > len(test)


def test_examples_execute():
    import runpy

    root = Path(__file__).resolve().parents[1] / "examples"
    for path in root.rglob("*.py"):
        runpy.run_path(str(path), run_name="__main__")


def test_snippet_catalog_and_prefix_engine():
    import runpy
    import subprocess

    root = Path(__file__).resolve().parents[1]
    runpy.run_path(str(root / "scripts" / "validate_snippets.py"), run_name="__main__")
    subprocess.run(["node", str(root / "vscode-extension" / "test" / "completion.test.js")], check=True)
