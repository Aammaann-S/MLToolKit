"""FastAPI prediction app. FastAPI and Pydantic are optional dependencies."""

from __future__ import annotations

from pathlib import Path


def create_prediction_app(model_path: str | Path):
    try:
        import joblib
        from fastapi import FastAPI
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise ImportError("Install the deploy extra: pip install fastapi pydantic joblib") from exc

    model = joblib.load(model_path)
    app = FastAPI(title="MLToolKit", version="1.1.0")

    class PredictRequest(BaseModel):
        values: list[float] = Field(min_length=1)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict")
    def predict(payload: PredictRequest):
        prediction = model.predict([payload.values])
        return {"prediction": prediction.tolist()}

    return app
