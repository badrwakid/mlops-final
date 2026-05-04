"""Small-path coverage for train_baseline helpers."""

from types import SimpleNamespace

from src.training.train_baseline import _feature_columns, _resolve_tracking_uri


def test_resolve_tracking_uri_prefers_env(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://from-env")
    assert _resolve_tracking_uri("http://configured") == "http://from-env"


def test_resolve_tracking_uri_fallback_when_env_missing(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert _resolve_tracking_uri("http://configured") == "http://configured"


def test_feature_columns_concatenates_lists():
    cfg = SimpleNamespace(
        data=SimpleNamespace(
            numeric_features=["a", "b"],
            categorical_features=["c"],
        ),
    )
    assert _feature_columns(cfg) == ["a", "b", "c"]
