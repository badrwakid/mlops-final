"""Execute dashboard helper paths so CI coverage includes src/dashboard/app.py.

Streamlit is not a CI dependency; inject a minimal fake module before import.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

# --- fake Streamlit (must run before importing src.dashboard.app) ---
if not isinstance(sys.modules.get("streamlit"), MagicMock):
    _st = MagicMock()

    def _cache_data(*_a, **_k):
        def _dec(fn):
            return fn

        return _dec

    _st.cache_data = _cache_data
    sys.modules["streamlit"] = _st

import src.dashboard.app as dash


@pytest.fixture
def st_mock():
    return sys.modules["streamlit"]


def test_demo_record_and_status():
    d = dash._demo_record()
    assert d["season"] == 2 and "temp" in d
    assert dash._status_indicator(True) == "Available"
    assert dash._status_indicator(False) == "Unavailable"


def test_file_status(tmp_path, monkeypatch):
    monkeypatch.setattr(dash, "PROJECT_ROOT", tmp_path)
    p = tmp_path / "a.txt"
    p.write_text("x", encoding="utf-8")
    st = dash._file_status(p)
    assert st["available"] is True
    assert "a.txt" in st["file"]


@patch("src.dashboard.app.requests.get")
def test_safe_get_json_ok(mock_get):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"status": "ok"}
    resp.raise_for_status = MagicMock()
    mock_get.return_value = resp
    out = dash.safe_get_json("/health")
    assert out["ok"] is True
    assert out["data"]["status"] == "ok"


@patch("src.dashboard.app.requests.get")
def test_safe_get_json_timeout(mock_get):
    import requests

    mock_get.side_effect = requests.Timeout("t")
    out = dash.safe_get_json("/health")
    assert out["ok"] is False
    assert "timed out" in out["error"].lower()


@patch("src.dashboard.app.requests.get")
def test_safe_get_json_bad_json(mock_get):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.side_effect = json.JSONDecodeError("x", "doc", 0)
    resp.raise_for_status = MagicMock()
    mock_get.return_value = resp
    out = dash.safe_get_json("/x")
    assert out["ok"] is False


@patch("src.dashboard.app.requests.post")
def test_safe_post_json_ok(mock_post):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"prediction": 42.0}
    resp.raise_for_status = MagicMock()
    mock_post.return_value = resp
    out = dash.safe_post_json("/predict", {"hr": 1})
    assert out["ok"] is True
    assert out["data"]["prediction"] == 42.0


@patch("src.dashboard.app.MlflowClient")
def test_mlflow_status_ok(mock_client_cls):
    mock_client = MagicMock()
    mock_client.search_experiments.return_value = [MagicMock(), MagicMock()]
    mock_client_cls.return_value = mock_client
    out = dash.mlflow_status()
    assert out["ok"] is True
    assert out["experiment_count"] == 2


@patch("src.dashboard.app.MlflowClient")
def test_mlflow_status_error(mock_client_cls):
    mock_client_cls.side_effect = RuntimeError("down")
    out = dash.mlflow_status()
    assert out["ok"] is False


@patch("src.dashboard.app.MlflowClient")
def test_mlflow_latest_runs_ok(mock_client_cls):
    run = MagicMock()
    run.data.tags = {"mlflow.runName": "r1"}
    run.data.metrics = {"test_rmse": 1.23}
    run.info.run_id = "abc"
    run.info.status = "FINISHED"
    run.info.start_time = 1
    run.info.end_time = 1001
    mock_client = MagicMock()
    mock_client.search_experiments.return_value = [MagicMock(experiment_id="1")]
    mock_client.search_runs.return_value = [run]
    mock_client_cls.return_value = mock_client
    out = dash.mlflow_latest_runs()
    assert out["ok"] is True
    assert len(out["items"]) == 1
    assert out["items"][0]["run_id"] == "abc"


@patch("src.dashboard.app.MlflowClient")
def test_mlflow_model_registry_ok(mock_client_cls):
    v = MagicMock()
    v.version = "3"
    v.current_stage = "Production"
    v.status = "READY"
    v.run_id = "rid"
    v.source = "models:/x"
    v.aliases = []
    mock_client = MagicMock()
    mock_client.search_model_versions.return_value = [v]
    mock_client_cls.return_value = mock_client
    out = dash.mlflow_model_registry("bike_share_regressor")
    assert out["ok"] is True
    assert out["latest"]["version"] == "3"


def test_inject_css(st_mock):
    dash._inject_css()
    st_mock.markdown.assert_called()


def test_page_overview_calls_helpers(st_mock):
    st_mock.columns.return_value = tuple(MagicMock() for _ in range(4))
    with (
        patch.object(dash, "safe_get_json", side_effect=[{"ok": True, "data": {}}, {"ok": True, "data": {}}]),
        patch.object(dash, "safe_get_text", return_value={"ok": True, "data": ""}),
        patch.object(dash, "mlflow_status", return_value={"ok": True}),
        patch.object(dash, "mlflow_latest_runs", return_value={"ok": False, "items": []}),
    ):
        dash.page_overview()
    st_mock.subheader.assert_called()


def test_page_mlflow_tracking_empty(st_mock):
    with patch.object(dash, "mlflow_status", return_value={"ok": False}):
        with patch.object(dash, "mlflow_latest_runs", return_value={"ok": False}):
            dash.page_mlflow_tracking()
    st_mock.info.assert_called()


def test_page_model_registry_unavailable(st_mock):
    with patch.object(dash, "mlflow_model_registry", return_value={"ok": False}):
        dash.page_model_registry()
    st_mock.info.assert_called()


def test_page_monitoring_drift(st_mock, tmp_path, monkeypatch):
    monkeypatch.setattr(dash, "PROJECT_ROOT", tmp_path)
    summary = tmp_path / "monitoring" / "evidently_reports" / "drift_summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text("{}", encoding="utf-8")
    with patch.object(dash, "safe_get_text", return_value={"ok": True, "data": "# hi"}):
        dash.page_monitoring_drift()
    st_mock.subheader.assert_called()


def test_page_documentation_evidence_partial(st_mock, tmp_path, monkeypatch):
    monkeypatch.setattr(dash, "PROJECT_ROOT", tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text("# p", encoding="utf-8")
    st_mock.selectbox.return_value = "README.md"
    dash.page_documentation_evidence()
    st_mock.dataframe.assert_called()


def test_main_routes_overview(st_mock):
    st_mock.sidebar.radio.return_value = "Overview"
    with (
        patch.object(dash, "page_overview") as po,
        patch.object(dash, "safe_get_json", return_value={"ok": False}),
        patch.object(dash, "safe_get_text", return_value={"ok": False}),
        patch.object(dash, "mlflow_status", return_value={"ok": False}),
        patch.object(dash, "mlflow_latest_runs", return_value={"ok": False, "items": []}),
    ):
        dash.main()
    po.assert_called_once()


@patch("src.dashboard.app.requests.post")
def test_safe_post_json_invalid_json_body(mock_post):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.side_effect = json.JSONDecodeError("x", "doc", 0)
    resp.raise_for_status = MagicMock()
    mock_post.return_value = resp
    out = dash.safe_post_json("/predict", {})
    assert out["ok"] is False


@patch("src.dashboard.app.requests.post")
def test_safe_post_json_timeout(mock_post):
    import requests

    mock_post.side_effect = requests.Timeout("t")
    out = dash.safe_post_json("/predict", {})
    assert out["ok"] is False


@patch("src.dashboard.app.requests.post")
def test_safe_post_json_request_error(mock_post):
    import requests

    mock_post.side_effect = requests.ConnectionError("x")
    out = dash.safe_post_json("/predict", {})
    assert out["ok"] is False


@patch("src.dashboard.app.MlflowClient")
def test_mlflow_latest_runs_exception(mock_client_cls):
    mock_client_cls.side_effect = RuntimeError("boom")
    out = dash.mlflow_latest_runs()
    assert out["ok"] is False


@patch("src.dashboard.app.MlflowClient")
def test_mlflow_model_registry_exception(mock_client_cls):
    mock_client_cls.side_effect = RuntimeError("boom")
    out = dash.mlflow_model_registry("bike_share_regressor")
    assert out["ok"] is False
