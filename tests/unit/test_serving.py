from types import SimpleNamespace

import src.serving.app as serving_app
from mlflow.exceptions import MlflowException


def _cfg(model_path="data/splits/model.pkl"):
    return SimpleNamespace(
        paths=SimpleNamespace(model=model_path),
        mlflow=SimpleNamespace(
            tracking_uri="http://configured-tracking:5000",
            registered_model_name="bike_share_regressor",
        ),
    )


def test_load_model_uses_production_registry_uri_and_env_override(monkeypatch):
    model = object()
    calls = []
    tracking_uris = []

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    monkeypatch.setattr(serving_app.mlflow, "set_tracking_uri", tracking_uris.append)
    monkeypatch.setattr(
        serving_app.mlflow.sklearn,
        "load_model",
        lambda uri: calls.append(uri) or model,
    )
    monkeypatch.setattr(
        serving_app.joblib,
        "load",
        lambda path: (_ for _ in ()).throw(AssertionError(f"unexpected fallback: {path}")),
    )

    loaded, version = serving_app._load_model(_cfg())

    assert loaded is model
    assert version == "Production"
    assert tracking_uris == ["file:./mlruns"]
    assert calls == ["models:/bike_share_regressor/Production"]


def test_load_model_falls_back_to_local_pickle_when_registry_unavailable(monkeypatch):
    local_model = object()
    local_calls = []

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(serving_app.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(
        serving_app.mlflow.sklearn,
        "load_model",
        lambda uri: (_ for _ in ()).throw(MlflowException(f"unavailable: {uri}")),
    )
    monkeypatch.setattr(
        serving_app.joblib,
        "load",
        lambda path: local_calls.append(path) or local_model,
    )

    loaded, version = serving_app._load_model(_cfg("data/splits/model.pkl"))

    assert loaded is local_model
    assert version == "local"
    assert local_calls == ["data/splits/model.pkl"]
