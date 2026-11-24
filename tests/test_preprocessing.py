import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from MLToolKit.toolkit_loader import load_impl


def test_load_csv_excel_json_numpy_and_sql(tmp_path: Path):
    loading = load_impl("implementations/01_Data_Preprocessing/data_loading.py")
    frame = pd.DataFrame({"city": ["Pune", "Delhi"], "age": [20, 30]})
    csv_path = tmp_path / "people.csv"
    excel_path = tmp_path / "people.xlsx"
    json_path = tmp_path / "people.json"
    npy_path = tmp_path / "values.npy"
    text_path = tmp_path / "values.txt"
    frame.to_csv(csv_path, index=False)
    frame.to_excel(excel_path, index=False)
    frame.to_json(json_path, orient="records")
    loading.save_numpy(npy_path, frame[["age"]].to_numpy())
    np.savetxt(text_path, frame[["age"]].to_numpy(), delimiter=",")

    assert loading.load_csv(csv_path)["city"].tolist() == ["Pune", "Delhi"]
    assert loading.load_dataset(csv_path).shape == (2, 2)
    assert loading.load_excel(excel_path)["age"].tolist() == [20, 30]
    assert loading.load_json(json_path)["city"].iloc[0] == "Pune"
    assert loading.load_numpy(npy_path).shape == (2, 1)
    assert loading.load_numpy_text(text_path).shape == (2,)
    assert set(loading.load_multiple_csv([csv_path])) == {"people"}

    connection = sqlite3.connect(":memory:")
    frame.to_sql("people", connection, index=False)
    loaded = loading.load_sql("SELECT city FROM people", connection)
    assert loaded["city"].tolist() == ["Pune", "Delhi"]


def test_inspection_and_duplicates():
    inspection = load_impl("implementations/01_Data_Preprocessing/data_inspection.py")
    duplicates = load_impl("implementations/01_Data_Preprocessing/duplicate_handling.py")
    frame = pd.DataFrame({"city": ["Pune", "Pune", None], "age": [1, 1, 2]})
    report = inspection.inspection_report(frame)
    assert report["rows"] == 3
    assert report["missing_counts"]["city"] == 1
    assert "age" in inspection.info_text(frame)
    assert inspection.value_counts(frame, "city").loc["Pune"] == 2
    cleaned = duplicates.drop_duplicates_keep_last(frame)
    assert len(cleaned) == 2


def test_mean_imputation_uses_training_values_only():
    missing = load_impl("implementations/01_Data_Preprocessing/missing_values.py")
    train = pd.DataFrame({"age": [0.0, 10.0, np.nan]})
    test = pd.DataFrame({"age": [np.nan, 100.0]})
    train_out, test_out, means = missing.impute_mean(train, test)
    assert means["age"] == pytest.approx(5.0)
    assert test_out["age"].iloc[0] == pytest.approx(5.0)
    assert train_out["age"].isna().sum() == 0

    filled, imputer = missing.simple_impute(train, strategy="median")
    assert filled.shape == (3, 1)
    assert imputer.statistics_[0] == pytest.approx(5.0)
    knn_train, knn_test, _knn = missing.knn_impute(train.fillna(0), test.fillna(0), n_neighbors=1)
    assert knn_train.shape[0] == 3
    assert knn_test.shape[0] == 2


def test_cleaning_encoding_scaling_and_outliers():
    cleaning = load_impl("implementations/01_Data_Preprocessing/inconsistent_data.py")
    encoding = load_impl("implementations/01_Data_Preprocessing/categorical_encoding.py")
    scaling = load_impl("implementations/01_Data_Preprocessing/feature_scaling.py")
    outliers = load_impl("implementations/01_Data_Preprocessing/outlier_handling.py")
    transform = load_impl("implementations/01_Data_Preprocessing/numerical_transformation.py")

    frame = pd.DataFrame({"city": [" Pune ", "delhi"], "miles": [1.0, 2.0], "date": ["2024-01-02", "bad"]})
    cleaned = cleaning.normalize_case(frame, ["city"], case="lower")
    assert cleaned["city"].tolist() == ["pune", "delhi"]
    converted = cleaning.convert_units(frame, "miles", 1.60934, "km")
    assert converted["km"].iloc[0] == pytest.approx(1.60934)
    dated = cleaning.convert_dates(frame, ["date"])
    assert pd.isna(dated["date"].iloc[1])

    train = pd.DataFrame({"city": ["Pune", "Delhi", "Pune"], "age": [0.0, 10.0, 20.0]})
    test = pd.DataFrame({"city": ["Pune", "Mumbai"], "age": [10.0, 30.0]})
    _train_labels, test_labels, _encoder = encoding.label_encode(train["city"], test["city"])
    assert test_labels[-1] == -1
    train_hot, test_hot, _hot = encoding.one_hot_encode(train, test, ["city"])
    assert "city_Mumbai" not in test_hot.columns
    train_dummies, test_dummies = encoding.get_dummies_aligned(train, test, ["city"])
    assert list(test_dummies.columns) == list(train_dummies.columns)

    train_scaled, test_scaled, scaler = scaling.standard_scale(train, test, ["age"])
    assert train_scaled["age"].mean() == pytest.approx(0, abs=1e-8)
    assert scaler.mean_[0] == pytest.approx(10)
    assert test_scaled.shape[0] == 2

    lower, upper = outliers.iqr_bounds(train, ["age"])
    capped, capped_test, _, _ = outliers.clip_to_train_quantiles(train, test, ["age"], 0.05, 0.95)
    assert capped["age"].max() <= upper["age"] + 1
    assert capped_test.shape[0] == 2
    logged = transform.log_transform(pd.DataFrame({"age": [0.0, 10.0]}), ["age"])
    assert logged["age"].iloc[0] == pytest.approx(0)


def test_feature_selection_split_and_pipeline():
    selection = load_impl("implementations/01_Data_Preprocessing/feature_selection.py")
    splitting = load_impl("implementations/01_Data_Preprocessing/train_test_split.py")
    pipeline = load_impl("implementations/01_Data_Preprocessing/preprocessing_pipeline.py")
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression

    features, target = make_classification(n_samples=80, n_features=6, n_informative=3, n_redundant=0, random_state=42)
    X_train, X_test, y_train, y_test = splitting.split_stratified(features, target, test_size=0.25, random_state=42)
    selected_train, selected_test, selector = selection.select_anova(X_train, y_train, X_test, k=3)
    assert selected_train.shape[1] == 3
    assert selected_test.shape[1] == 3
    assert selector.get_support().sum() == 3
    positive = np.abs(features)
    chi_train, chi_test, _chi = selection.select_chi2(positive[:60], target[:60], positive[60:], k=2)
    assert chi_train.shape[1] == 2
    assert chi_test.shape[1] == 2

    frame = pd.DataFrame(features, columns=[f"f{i}" for i in range(6)])
    frame["city"] = np.where(target == 1, "yes", "no")
    frame["target"] = target
    train = frame.iloc[:60]
    test = frame.iloc[60:]
    model, predictions, actual = pipeline.fit_model_pipeline(
        train, test, "target", [f"f{i}" for i in range(6)], ["city"], LogisticRegression(max_iter=500)
    )
    assert len(predictions) == len(actual)
    assert hasattr(model, "predict")


def test_eda_figures_close():
    summary = load_impl("implementations/02_EDA/data_summary.py")
    stats = load_impl("implementations/02_EDA/descriptive_statistics.py")
    correlation = load_impl("implementations/02_EDA/correlation.py")
    categorical = load_impl("implementations/02_EDA/categorical_analysis.py")
    plots = load_impl("implementations/02_EDA/visualization.py")
    frame = pd.DataFrame({"group": ["a", "a", "b"], "value": [1.0, 2.0, 3.0], "city": ["Pune", "Delhi", "Pune"]})
    other = pd.DataFrame({"group": ["a", "b"], "score": [5, 6]})
    assert summary.merge_frames(frame, other, on="group").shape[0] == 3
    assert stats.numeric_summary(frame).loc["value", "mean"] == pytest.approx(2)
    numeric = frame.assign(extra=[1.0, 2.0, 4.0])
    assert correlation.correlation_matrix(numeric).shape == (2, 2)
    assert categorical.frequency_table(frame, "city")["count"].sum() == 3
    fig, _ax = plots.plot_histogram(frame, "value")
    plt.close(fig)
    fig, _ax = plots.plot_missing_heatmap(frame)
    plt.close(fig)
