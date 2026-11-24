"""Local metric logs, with an optional MLflow run when MLflow is installed."""

from __future__ import annotations

import json
from pathlib import Path


def log_local_run(path: str | Path, params: dict, metrics: dict) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"params": params, "metrics": metrics}, indent=2), encoding="utf-8")
    return destination


def log_mlflow_run(params: dict, metrics: dict, experiment_name: str = "MLToolKit") -> str:
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError("Install MLflow with: pip install mlflow") from exc
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        return run.info.run_id
