from types import SimpleNamespace

import pytest
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

    monkeypatch.delenv("SERVE_USE_LOCAL_MODEL_ONLY", raising=False)
    monkeypatch.delenv("REQUIRE_REGISTRY_MODEL", raising=False)
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

    loaded, version, meta = serving_app._load_model(_cfg())

    assert loaded is model
    assert version == "Production"
    assert meta.load_source == serving_app.LOAD_SOURCE_REGISTRY
    assert tracking_uris == ["file:./mlruns"]
    assert calls == ["models:/bike_share_regressor/Production"]


def test_load_model_falls_back_to_local_pickle_when_registry_unavailable(monkeypatch):
    local_model = object()
    local_calls = []

    monkeypatch.delenv("SERVE_USE_LOCAL_MODEL_ONLY", raising=False)
    monkeypatch.delenv("REQUIRE_REGISTRY_MODEL", raising=False)
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

    loaded, version, meta = serving_app._load_model(_cfg("data/splits/model.pkl"))

    assert loaded is local_model
    assert version == "local"
    assert meta.load_source == serving_app.LOAD_SOURCE_LOCAL_FALLBACK
    assert meta.fallback_reason is not None
    assert local_calls == ["data/splits/model.pkl"]


def test_load_model_skips_registry_when_serve_use_local_only(monkeypatch):
    local_model = object()
    registry_calls: list[str] = []

    monkeypatch.delenv("REQUIRE_REGISTRY_MODEL", raising=False)
    monkeypatch.setenv("SERVE_USE_LOCAL_MODEL_ONLY", "1")
    monkeypatch.setattr(
        serving_app.mlflow.sklearn,
        "load_model",
        lambda uri: registry_calls.append(uri) or (_ for _ in ()).throw(
            AssertionError("registry should not be used")
        ),
    )
    monkeypatch.setattr(
        serving_app.joblib,
        "load",
        lambda path: local_model if path == "data/splits/model.pkl" else (_ for _ in ()).throw(
            AssertionError(path)
        ),
    )

    loaded, version, meta = serving_app._load_model(_cfg("data/splits/model.pkl"))

    assert loaded is local_model
    assert version == "local"
    assert meta.load_source == serving_app.LOAD_SOURCE_LOCAL_ONLY_ENV
    assert registry_calls == []


def test_strict_production_raises_when_mlflow_load_fails(monkeypatch):
    monkeypatch.setenv("REQUIRE_REGISTRY_MODEL", "1")  # same gate as PRODUCTION_STRICT for registry path
    monkeypatch.delenv("SERVE_USE_LOCAL_MODEL_ONLY", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(serving_app.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(
        serving_app.mlflow.sklearn,
        "load_model",
        lambda uri: (_ for _ in ()).throw(MlflowException("registry unavailable")),
    )

    with pytest.raises(RuntimeError, match="Production registry model required"):
        serving_app._load_model(_cfg())


def test_reference_dataset_path_matches_params_yaml():
    p = serving_app.reference_dataset_path()
    assert p == serving_app.ROOT_DIR / "data" / "splits" / "reference.parquet"
