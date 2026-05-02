from fastapi.testclient import TestClient
from src.serving.app import app

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


def test_health_predict_batch_and_metrics():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_name"] == "bike_share_regressor"
        assert health.json()["model_version"]

        prediction = client.post("/predict", json=VALID_RECORD)
        assert prediction.status_code == 200
        prediction_body = prediction.json()
        assert isinstance(prediction_body["prediction"], float)
        assert 0.0 <= prediction_body["confidence"] <= 1.0
        assert prediction_body["model_version"] == health.json()["model_version"]

        batch = client.post("/predict/batch", json={"records": [VALID_RECORD, VALID_RECORD]})
        assert batch.status_code == 200
        batch_body = batch.json()
        assert len(batch_body["predictions"]) == 2
        assert all("prediction" in item and "confidence" in item for item in batch_body["predictions"])

        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert "bike_inference_total" in metrics.text
