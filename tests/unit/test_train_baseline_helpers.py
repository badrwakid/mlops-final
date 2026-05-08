"""Coverage for train_baseline helpers (no full MLflow run)."""

from __future__ import annotations

from types import SimpleNamespace

from src.training import train_baseline as tb


def test_resolve_tracking_uri_env_override(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env:5000")
    assert tb._resolve_tracking_uri("http://cfg:5000") == "http://env:5000"


def test_resolve_tracking_uri_from_config(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert tb._resolve_tracking_uri("http://cfg:5000") == "http://cfg:5000"


def test_feature_columns():
    cfg = SimpleNamespace(
        data=SimpleNamespace(
            numeric_features=["a", "b"],
            categorical_features=["c"],
        )
    )
    assert tb._feature_columns(cfg) == ["a", "b", "c"]
