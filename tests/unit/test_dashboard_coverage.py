"""Cover src/dashboard/app.py helper functions without a real Streamlit runtime."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

# --- inject minimal Streamlit stub before the dashboard module is imported ---
if not isinstance(sys.modules.get("streamlit"), MagicMock):
    _st = MagicMock()

    def _cache_data(*_a, **_k):
        def _dec(fn):
            return fn

        return _dec

    _st.cache_data = _cache_data
    sys.modules["streamlit"] = _st

import src.dashboard.app as dash  # noqa: E402

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_percent_formats_float():
    assert dash.percent(0.5) == "50.00%"
    assert dash.percent(0.0) == "0.00%"
    assert dash.percent(None) == "Unavailable"


def test_status_label():
    assert dash.status_label(True) == "Available"
    assert dash.status_label(False) == "Unavailable"


def test_positive_helper():
    assert dash._positive(1.0) is True
    assert dash._positive(0.0) is False
    assert dash._positive(-1.0) is False
    assert dash._positive(None) is False


def test_progress_fraction_normal():
    assert dash._progress_fraction(0.1, 0.2) == pytest.approx(0.5)
    assert dash._progress_fraction(0.4, 0.2) == pytest.approx(1.0)  # capped
    assert dash._progress_fraction(0.0, 0.2) == pytest.approx(0.0)


def test_progress_fraction_guards():
    assert dash._progress_fraction(None, 0.2) == 0.0
    assert dash._progress_fraction(0.1, None) == 0.0
    assert dash._progress_fraction(0.1, 0.0) == 0.0
    assert dash._progress_fraction(0.1, -1.0) == 0.0


def test_status_from_threshold():
    assert dash._status_from_threshold(0.1, 0.2) == "Below threshold"
    assert dash._status_from_threshold(0.19, 0.2) == "Near threshold"
    assert dash._status_from_threshold(0.25, 0.2) == "Above threshold"
    assert dash._status_from_threshold(None, 0.2) == "Unavailable"
    assert dash._status_from_threshold(0.1, None) == "Unavailable"


def test_drift_threshold_from_summary():
    summary = {"threshold": 0.25}
    assert dash._drift_threshold(summary, {}) == pytest.approx(0.25)


def test_drift_threshold_falls_back_to_params():
    summary = {}
    params = {"drift": {"drift_threshold_share": 0.20}}
    assert dash._drift_threshold(summary, params) == pytest.approx(0.20)


def test_drift_threshold_returns_none_when_missing():
    assert dash._drift_threshold({}, {}) is None


def test_drift_share_primary_key():
    summary = {"drift_share_inputs_only": 0.3}
    assert dash._drift_share(summary) == pytest.approx(0.3)


def test_drift_share_fallback_key():
    summary = {"drift_share_inputs_only_unweighted": 0.15}
    assert dash._drift_share(summary) == pytest.approx(0.15)


def test_drift_share_missing():
    assert dash._drift_share({}) is None


# ---------------------------------------------------------------------------
# parse_prometheus_metrics
# ---------------------------------------------------------------------------


def test_parse_prometheus_metrics_extracts_names():
    text = (
        "# HELP bike_inference_total Total inferences\n"
        "# TYPE bike_inference_total counter\n"
        "bike_inference_total{output_class=\"low\"} 5\n"
        "bike_prediction_confidence_bucket{le=\"0.5\"} 2\n"
    )
    result = dash.parse_prometheus_metrics(text)
    assert "bike_inference_total" in result["metric_names"]
    assert "bike_prediction_confidence_bucket" in result["metric_names"]
    df = result["availability_df"]
    assert "inference" in df["metric theme"].values


def test_parse_prometheus_metrics_empty():
    result = dash.parse_prometheus_metrics("")
    assert result["metric_names"] == []


# ---------------------------------------------------------------------------
# File I/O helpers (using tmp_path)
# ---------------------------------------------------------------------------


def test_load_json_file_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(dash, "PROJECT_ROOT", tmp_path)
    f = tmp_path / "data.json"
    f.write_text('{"key": 42}', encoding="utf-8")
    result = dash.load_json_file(f)
    assert result["ok"] is True
    assert result["data"]["key"] == 42


def test_load_json_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(dash, "PROJECT_ROOT", tmp_path)
    result = dash.load_json_file(tmp_path / "missing.json")
    assert result["ok"] is False


def test_load_json_file_malformed(tmp_path, monkeypatch):
    monkeypatch.setattr(dash, "PROJECT_ROOT", tmp_path)
    f = tmp_path / "bad.json"
    f.write_text("not json {{{", encoding="utf-8")
    result = dash.load_json_file(f)
    assert result["ok"] is False


def test_load_text_file_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(dash, "PROJECT_ROOT", tmp_path)
    f = tmp_path / "doc.md"
    f.write_text("# Hello", encoding="utf-8")
    result = dash.load_text_file(f)
    assert result["ok"] is True
    assert "Hello" in result["text"]


def test_load_text_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(dash, "PROJECT_ROOT", tmp_path)
    result = dash.load_text_file(tmp_path / "ghost.md")
    assert result["ok"] is False


def test_file_metadata_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(dash, "PROJECT_ROOT", tmp_path)
    f = tmp_path / "artifact.pkl"
    f.write_bytes(b"data")
    meta = dash.file_metadata(f)
    assert meta["ok"] is True
    assert meta["exists"] is True
    assert meta["size_bytes"] == 4


def test_file_metadata_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(dash, "PROJECT_ROOT", tmp_path)
    meta = dash.file_metadata(tmp_path / "nope.pkl")
    assert meta["ok"] is False
    assert meta["exists"] is False


def test_load_params_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(dash, "PROJECT_ROOT", tmp_path)
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "params.yaml").write_text("drift:\n  drift_threshold_share: 0.2\n", encoding="utf-8")
    result = dash.load_params()
    assert result["ok"] is True
    assert result["data"]["drift"]["drift_threshold_share"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


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


@patch("src.dashboard.app.requests.get")
def test_safe_get_json_error(mock_get):
    import requests

    mock_get.side_effect = requests.ConnectionError("down")
    out = dash.safe_get_json("/health")
    assert out["ok"] is False


@patch("src.dashboard.app.requests.get")
def test_safe_get_text_ok(mock_get):
    resp = MagicMock()
    resp.status_code = 200
    resp.text = "metric 1.0"
    resp.raise_for_status = MagicMock()
    mock_get.return_value = resp
    out = dash.safe_get_text("/metrics")
    assert out["ok"] is True
    assert "metric" in out["data"]


@patch("src.dashboard.app.requests.get")
def test_safe_get_text_timeout(mock_get):
    import requests

    mock_get.side_effect = requests.Timeout("t")
    out = dash.safe_get_text("/metrics")
    assert out["ok"] is False


# ---------------------------------------------------------------------------
# load_feature_scores
# ---------------------------------------------------------------------------


def test_load_feature_scores_ok(tmp_path, monkeypatch):
    monkeypatch.setattr(dash, "PROJECT_ROOT", tmp_path)
    docs = tmp_path / "docs"
    docs.mkdir()
    payload = {"all_features_ranked_by_f_score": [{"feature": "temp", "f_score": 99.0}], "k_selected": 14}
    (docs / "feature_scores.json").write_text(json.dumps(payload), encoding="utf-8")
    result = dash.load_feature_scores()
    assert result["ok"] is True
    assert result["k_selected"] == 14
    assert "temp" in result["df"]["feature"].values


def test_load_feature_scores_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(dash, "PROJECT_ROOT", tmp_path)
    result = dash.load_feature_scores()
    assert result["ok"] is False


# ---------------------------------------------------------------------------
# _inject_css and render_status_card (Streamlit UI — just verify no exception)
# ---------------------------------------------------------------------------


def test_inject_css_no_exception():
    dash._inject_css()
    sys.modules["streamlit"].markdown.assert_called()


def test_render_status_card_no_exception():
    dash.render_status_card("Model", "Available", "some note")
    sys.modules["streamlit"].markdown.assert_called()
