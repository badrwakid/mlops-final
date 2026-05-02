from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.config import Config, load_config
from src.serving.metrics import (
    FEATURE_HR,
    FEATURE_TEMP,
    INFERENCE_COUNT,
    MODEL_VERSION,
    PREDICTION_CONFIDENCE,
    PREDICTION_VALUE,
)
from src.serving.schemas import (
    BatchPredictionItem,
    BatchPredictRequest,
    BatchPredictResponse,
    BikeRecord,
    HealthResponse,
    PredictResponse,
)


@dataclass
class LoadedModel:
    model: Any
    preprocessor: Any
    model_name: str
    model_version: str


def _resolve_tracking_uri(cfg: Config) -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or cfg.mlflow.tracking_uri


def _feature_columns(cfg: Config) -> list[str]:
    return cfg.data.numeric_features + cfg.data.categorical_features


def _load_model(cfg: Config) -> tuple[Any, str]:
    model_name = cfg.mlflow.registered_model_name
    model_uri = f"models:/{model_name}/Production"
    mlflow.set_tracking_uri(_resolve_tracking_uri(cfg))
    try:
        return mlflow.sklearn.load_model(model_uri), "Production"
    except Exception:
        return joblib.load(cfg.paths.model), "local"


def load_artifacts() -> LoadedModel:
    cfg = load_config()
    model, model_version = _load_model(cfg)
    preprocessor = joblib.load(cfg.paths.preprocessor)
    return LoadedModel(
        model=model,
        preprocessor=preprocessor,
        model_name=cfg.mlflow.registered_model_name,
        model_version=model_version,
    )


def _to_dataframe(records: list[BikeRecord]) -> pd.DataFrame:
    cfg = load_config()
    return pd.DataFrame([record.model_dump() for record in records])[_feature_columns(cfg)]


def _predict_with_confidence(model: Any, features: np.ndarray) -> list[BatchPredictionItem]:
    predictions = np.asarray(model.predict(features), dtype=float)
    if hasattr(model, "estimators_"):
        tree_predictions = np.asarray([tree.predict(features) for tree in model.estimators_])
        std = tree_predictions.std(axis=0)
        confidence = 1.0 / (1.0 + (std / (np.abs(predictions) + 1e-9)))
    else:
        confidence = np.ones_like(predictions, dtype=float)

    return [
        BatchPredictionItem(
            prediction=float(prediction),
            confidence=float(np.clip(score, 0.0, 1.0)),
        )
        for prediction, score in zip(predictions, confidence, strict=True)
    ]


def _loaded_model(app: FastAPI) -> LoadedModel:
    loaded = getattr(app.state, "loaded_model", None)
    if loaded is None:
        raise HTTPException(status_code=503, detail="model is not loaded")
    return loaded


@asynccontextmanager
async def lifespan(app: FastAPI):
    loaded = load_artifacts()
    app.state.loaded_model = loaded
    MODEL_VERSION.labels(version=loaded.model_version).set(1)
    yield


app = FastAPI(title="Bike Sharing Predictor", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = _loaded_model(app)
    return HealthResponse(
        status="ok",
        model_name=loaded.model_name,
        model_version=loaded.model_version,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(record: BikeRecord) -> PredictResponse:
    loaded = _loaded_model(app)
    df = _to_dataframe([record])
    features = loaded.preprocessor.transform(df)
    item = _predict_with_confidence(loaded.model, features)[0]
    FEATURE_TEMP.observe(record.temp)
    FEATURE_HR.observe(record.hr)
    PREDICTION_CONFIDENCE.observe(item.confidence)
    PREDICTION_VALUE.observe(item.prediction)
    INFERENCE_COUNT.labels(endpoint="/predict").inc()
    return PredictResponse(
        prediction=item.prediction,
        confidence=item.confidence,
        model_version=loaded.model_version,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    loaded = _loaded_model(app)
    df = _to_dataframe(request.records)
    features = loaded.preprocessor.transform(df)
    items = _predict_with_confidence(loaded.model, features)
    for record, item in zip(request.records, items, strict=True):
        FEATURE_TEMP.observe(record.temp)
        FEATURE_HR.observe(record.hr)
        PREDICTION_CONFIDENCE.observe(item.confidence)
        PREDICTION_VALUE.observe(item.prediction)
    INFERENCE_COUNT.labels(endpoint="/predict/batch").inc()
    return BatchPredictResponse(predictions=items, model_version=loaded.model_version)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
