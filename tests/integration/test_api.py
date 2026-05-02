"""API integration tests.

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

# After SelectKBest(k=12) the training pipeline uses 12 columns; keep in sync with configs/params.yaml.
_N_FEATURES = 12


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
    assert "bike_inference_total" in metrics.text
