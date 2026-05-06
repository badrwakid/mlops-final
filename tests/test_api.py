"""API tests for the FastAPI serving app (Component 4).

GitHub Actions does not ship ``data/splits/model.pkl`` or a local MLflow server, so we patch
``load_artifacts`` with a tiny sklearn model + preprocessor that match the expected tensor shape.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import FunctionTransformer
from src.serving.app import LoadedModel, app

VALID_RECORD = {
    "season": 2,
    "mnth": 6,
    "hr": 12,
    "holiday": 0,
    "weekday": 3,
    "workingday": 1,
    "weathersit": 1,
    "temp": 0.62,
    "atemp": 0.60,
    "hum": 0.45,
    "windspeed": 0.20,
}

# After SelectKBest(k=feature_selection_k) the training pipeline uses that many columns; keep in sync
# with configs/params.yaml preprocessing.feature_selection_k (currently 14).
_N_FEATURES = 14


def _fake_load_artifacts() -> LoadedModel:
    model = DummyRegressor(strategy="constant", constant=100.0)
    model.fit(np.zeros((5, _N_FEATURES)), np.zeros(5))

    def _transform(df):  # noqa: ANN001
        return np.zeros((len(df), _N_FEATURES))

    preprocessor = FunctionTransformer(_transform, validate=False)
    return LoadedModel(
        model=model,
        preprocessor=preprocessor,
        model_name="bike_share_regressor",
        model_version="ci-fixture",
    )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("src.serving.app.load_artifacts", _fake_load_artifacts)
    with TestClient(app) as test_client:
        yield test_client


def test_health_predict_batch_and_metrics(client):
    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["model_name"] == "bike_share_regressor"
    assert body["model_version"] == "ci-fixture"
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"

    prediction = client.post("/predict", json=VALID_RECORD)
    assert prediction.status_code == 200
    prediction_body = prediction.json()
    assert isinstance(prediction_body["prediction"], float)
    assert 0.0 <= prediction_body["confidence"] <= 1.0
    assert prediction_body["model_version"] == body["model_version"]

    batch = client.post("/predict/batch", json={"records": [VALID_RECORD, VALID_RECORD]})
    assert batch.status_code == 200
    batch_body = batch.json()
    assert len(batch_body["predictions"]) == 2
    assert all("prediction" in item and "confidence" in item for item in batch_body["predictions"])

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    text = metrics.text
    for name in (
        "bike_inference_total",
        "bike_prediction_confidence",
        "bike_feature_temp",
        "bike_feature_hr",
        "bike_model_version_info",
        "bike_prediction_latency_seconds",
    ):
        assert name in text, f"expected Prometheus metric {name!r} in /metrics response"
    # Regression rubric: inference count labeled by predicted demand class (binned target).
    assert "output_class" in text
    assert "low" in text  # DummyRegressor constant=100.0 → "low" bin


def test_predict_rejects_invalid_input(client):
    bad = {**VALID_RECORD, "temp": 5.0}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_dashboard_routes_and_helper_endpoints(client):
    for path in ("/", "/dashboard"):
        response = client.get(path)
        assert response.status_code == 200
        assert "Bike Sharing Demand MLOps Dashboard" in response.text

    status = client.get("/api/status")
    assert status.status_code == 200
    assert status.json()["health"]["status"] in {"ok", "degraded"}

    model_info = client.get("/api/model-info")
    assert model_info.status_code == 200
    assert "available" in model_info.json()

    drift = client.get("/api/drift-summary")
    assert drift.status_code == 200
    assert "available" in drift.json()

    summary = client.get("/api/dashboard-summary")
    assert summary.status_code == 200
    payload = summary.json()
    assert "runtime" in payload
    assert "drift" in payload


def test_predict_missing_fields_and_wrong_types(client):
    missing = VALID_RECORD.copy()
    missing.pop("temp")
    response_missing = client.post("/predict", json=missing)
    assert response_missing.status_code == 422

    wrong_type = {**VALID_RECORD, "hr": "nope"}
    response_type = client.post("/predict", json=wrong_type)
    assert response_type.status_code == 422


def test_predict_batch_validation_and_limits(client):
    valid = client.post("/predict/batch", json={"records": [VALID_RECORD]})
    assert valid.status_code == 200

    empty = client.post("/predict/batch", json={"records": []})
    assert empty.status_code == 422

    too_many = client.post("/predict/batch", json={"records": [VALID_RECORD] * 101})
    assert too_many.status_code == 422
    assert "Batch size cannot exceed 100 records" in str(too_many.json())


def test_ready_and_health_when_model_not_loaded(monkeypatch):
    def _boom():
        raise RuntimeError("forced load failure")

    monkeypatch.setattr("src.serving.app.load_artifacts", _boom)
    with TestClient(app) as test_client:
        health = test_client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "degraded"

        ready = test_client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["available"] is False

        pred = test_client.post("/predict", json=VALID_RECORD)
        assert pred.status_code == 503


def test_drift_endpoint_safe_when_file_missing(client, monkeypatch):
    from src.serving import app as serving_module

    monkeypatch.setattr(serving_module, "DRIFT_SUMMARY_PATH", serving_module.Path("missing.json"))
    response = client.get("/api/drift-summary")
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False
    assert "not found" in payload["message"].lower()
