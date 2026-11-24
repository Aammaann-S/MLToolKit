from pathlib import Path

import numpy as np
import pandas as pd

from MLToolKit.toolkit_loader import load_impl


def test_text_time_series_features_and_local_deployment(tmp_path: Path):
    text = load_impl("implementations/11_NLP/text_preprocessing.py")
    classifier = load_impl("implementations/11_NLP/text_classification.py")
    forecasting = load_impl("implementations/12_Time_Series/forecasting.py")
    lags = load_impl("implementations/12_Time_Series/feature_extraction.py")
    deployment = load_impl("implementations/13_Deployment_MLOps/model_deployment.py")
    tracking = load_impl("implementations/13_Deployment_MLOps/mlflow_tracking.py")
    created = load_impl("implementations/07_Feature_Engineering/feature_creation.py")
    anomaly = load_impl("implementations/08_Advanced_ML/anomaly_detection.py")

    assert text.tokenize("Visit HTTPS://example.com, now!") == ["visit", "now"]
    documents = ["good film"] * 12 + ["bad film"] * 12
    labels = [1] * 12 + [0] * 12
    trained = classifier.train_text_classifier(documents, labels, test_size=0.25)
    assert len(trained["y_pred"]) == len(trained["y_test"])

    series = pd.Series(np.arange(12, dtype=float))
    lagged = lags.add_lags(series.to_frame("value"), "value", [1, 2])
    assert pd.isna(lagged["value_lag_1"].iloc[0])
    assert forecasting.naive_forecast(series, horizon=2).tolist() == [11.0, 11.0]
    fitted = forecasting.fit_arima(pd.Series(np.cumsum(np.arange(12, dtype=float))), order=(0, 1, 0))
    assert len(fitted.forecast(steps=2)) == 2

    frame = pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [2.0, 2.0, 2.0], "date": ["2024-01-01", "2024-01-02", "2024-01-03"]})
    engineered = created.add_datetime_parts(created.add_ratio(frame, "a", "b", "ratio"), "date")
    assert engineered["ratio"].iloc[0] == 0.5
    assert engineered["date_year"].iloc[0] == 2024

    values = np.vstack([np.random.default_rng(0).normal(size=(40, 2)), np.array([[8.0, 8.0]])])
    flags = anomaly.isolation_forest(values, contamination=0.05)
    assert flags["anomaly"].shape[0] == 41

    data = load_impl("implementations/09_Utilities/synthetic_datasets.py").classification_frame(n_samples=80)
    saved = deployment.train_deployable_classifier(data.drop(columns=["target"]), data["target"])
    path = deployment.save_model(saved["model"], tmp_path / "model.joblib")
    loaded = deployment.load_model(path)
    assert deployment.predict_records(loaded, data.drop(columns=["target"]).iloc[0].tolist())
    log_path = tracking.log_local_run(tmp_path / "run.json", {"model": "logistic"}, {"accuracy": 1})
    assert log_path.exists()
