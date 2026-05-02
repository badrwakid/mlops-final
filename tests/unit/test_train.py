from types import SimpleNamespace

import pytest
from src.training.train import (
    _apply_validation_gate,
    _build_baseline_model,
    _feature_columns,
    _persist_artifacts_if_gate_passes,
    _resolve_tracking_uri,
)


def test_resolve_tracking_uri_prefers_environment_override(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:./mlruns")

    assert _resolve_tracking_uri("http://localhost:5000") == "file:./mlruns"


def test_resolve_tracking_uri_uses_config_value_when_env_missing(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    assert _resolve_tracking_uri("http://localhost:5000") == "http://localhost:5000"


def test_build_baseline_model_uses_required_deterministic_parameters():
    model = _build_baseline_model(random_state=42)
    params = model.get_params()

    assert params["n_estimators"] == 100
    assert params["max_depth"] == 8
    assert params["n_jobs"] == -1
    assert params["random_state"] == 42


def test_feature_columns_returns_numeric_then_categorical_columns():
    cfg = SimpleNamespace(
        data=SimpleNamespace(
            numeric_features=["temp", "atemp", "hum"],
            categorical_features=["season", "holiday"],
        )
    )

    assert _feature_columns(cfg) == ["temp", "atemp", "hum", "season", "holiday"]


def test_apply_validation_gate_passes_when_r2_meets_threshold():
    gate = _apply_validation_gate({"r2": 0.70}, min_test_r2=0.70)

    assert gate == {"min_test_r2": 0.70, "passed": True}


def test_apply_validation_gate_raises_when_r2_is_below_threshold():
    with pytest.raises(RuntimeError, match="Validation gate failed"):
        _apply_validation_gate({"r2": 0.69}, min_test_r2=0.70)


def test_persist_artifacts_if_gate_passes_raises_before_artifact_persistence():
    artifact_persisted = False

    def persist_artifacts():
        nonlocal artifact_persisted
        artifact_persisted = True

    with pytest.raises(RuntimeError, match="Validation gate failed"):
        _persist_artifacts_if_gate_passes(
            {"r2": 0.69},
            min_test_r2=0.70,
            persist_artifacts=persist_artifacts,
        )

    assert artifact_persisted is False
