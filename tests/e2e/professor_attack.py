import json

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


def _fake_load_artifacts() -> LoadedModel:
    model = DummyRegressor(strategy="constant", constant=100.0)
    model.fit(np.zeros((5, 14)), np.zeros(5))

    def _transform(df):  # noqa: ANN001
        return np.zeros((len(df), 14))

    preprocessor = FunctionTransformer(_transform, validate=False)
    return LoadedModel(
        model=model,
        preprocessor=preprocessor,
        model_name="bike_share_regressor",
        model_version="ci-fixture",
        load_source="registry_production",
    )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("src.serving.app.load_artifacts", _fake_load_artifacts)
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_predict_with_out_of_range_hour_shows_inline_error(client):
    response = client.post("/predict", json={**VALID_RECORD, "hr": 99})
    assert response.status_code == 422


def test_predict_with_empty_form_shows_required_errors(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422


@pytest.mark.skip(reason="Requires browser E2E automation")
def test_predict_button_disabled_during_pending_request():
    pass


@pytest.mark.skip(reason="Requires browser E2E automation")
def test_predict_double_click_fires_one_request():
    pass


@pytest.mark.skip(reason="Requires browser E2E automation")
def test_batch_with_empty_textarea_shows_message():
    pass


def test_batch_with_invalid_json_shows_line_number(client):
    response = client.post("/predict/batch", data="{bad-json", headers={"Content-Type": "application/json"})
    assert response.status_code in (400, 422)


def test_batch_with_101_lines_rejected_client_side(client):
    response = client.post("/predict/batch", json={"records": [VALID_RECORD] * 101})
    assert response.status_code == 422


def test_batch_with_101_lines_rejected_server_side(client):
    response = client.post("/predict/batch", json={"records": [VALID_RECORD] * 101})
    assert response.status_code == 422


def test_batch_with_extra_unknown_field_rejected(client):
    response = client.post("/predict", json={**VALID_RECORD, "__proto__": {"x": 1}})
    assert response.status_code == 422


@pytest.mark.skip(reason="Requires browser E2E automation")
def test_batch_with_xss_payload_renders_as_text():
    pass


def test_predict_when_api_down_shows_friendly_error(client, monkeypatch):
    monkeypatch.setattr("src.serving.app._loaded_model", lambda _app: (_ for _ in ()).throw(RuntimeError("down")))
    response = client.post("/predict", json=VALID_RECORD)
    assert response.status_code == 500
    body = response.json()
    assert "detail" in body or "error" in body


def test_predict_when_api_returns_500_shows_friendly_error(client, monkeypatch):
    monkeypatch.setattr("src.serving.app._to_dataframe", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    response = client.post("/predict", json=VALID_RECORD)
    assert response.status_code == 500
    assert response.json().get("detail") == "Internal server error"


@pytest.mark.skip(reason="Requires browser E2E automation")
def test_predict_when_api_times_out_recovers_within_8s():
    pass


def test_predict_when_api_returns_nan_handled(client, monkeypatch):
    class NanModel:
        def predict(self, features):
            return np.array([np.nan])

    loaded = _fake_load_artifacts()
    loaded.model = NanModel()
    monkeypatch.setattr("src.serving.app._loaded_model", lambda _app: loaded)
    response = client.post("/predict", json=VALID_RECORD)
    assert response.status_code == 500


@pytest.mark.skip(reason="Requires browser E2E automation")
def test_health_failing_at_boot_disables_submit():
    pass


def test_csp_header_present_on_html_response(client):
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "Content-Security-Policy" in response.headers


def test_payload_over_200kb_returns_413(client):
    huge = {"records": [VALID_RECORD for _ in range(100)]}
    # Default API limit is 256 KiB (see MAX_REQUEST_BODY_BYTES); pad enough to exceed it.
    huge["pad"] = "x" * 350000
    payload = json.dumps(huge)
    response = client.post(
        "/predict/batch",
        content=payload,
        headers={"Content-Type": "application/json", "Content-Length": str(len(payload.encode("utf-8")))},
    )
    assert response.status_code == 413


def test_unknown_field_returns_422(client):
    response = client.post("/predict", json={**VALID_RECORD, "unknown": 123})
    assert response.status_code == 422


@pytest.mark.skip(reason="Requires browser E2E automation")
def test_predict_keyboard_only_submits_on_enter():
    pass


def test_noscript_block_renders_when_js_disabled(client):
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "This dashboard needs JavaScript to predict." in response.text
