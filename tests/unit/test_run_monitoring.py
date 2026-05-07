from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from monitoring import run_monitoring


class _StubReport:
    def as_dict(self):
        return {
            "metrics": [
                {"metric": "DataQualityTable", "result": {}},
                {
                    "metric": "DataDriftTable",
                    "result": {
                        "drift_by_columns": {
                            "temp": {"drift_detected": True},
                            "hum": {"drift_detected": False},
                        }
                    },
                },
            ]
        }


def test_drift_per_feature_from_report_extracts_detected_flags():
    out = run_monitoring._drift_per_feature_from_report(_StubReport())
    assert out == {"temp": True, "hum": False}


def test_monitoring_mode_marks_operational_and_optional_demo():
    assert run_monitoring._monitoring_mode(True) == "operational_with_demo"
    assert run_monitoring._monitoring_mode(False) == "operational"


def test_main_fails_loud_when_required_artifacts_are_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = SimpleNamespace(
        paths=SimpleNamespace(
            model="data/splits/model.pkl",
            preprocessor="data/splits/preprocessor.pkl",
            reference="data/splits/reference.parquet",
            production="data/splits/production.parquet",
        ),
        drift=SimpleNamespace(
            generate_synthetic_demo_report=True,
            synthetic_demo_report_name="drift_synthetic_demo.html",
        ),
    )
    monkeypatch.setattr(run_monitoring, "load_config", lambda: cfg)

    with pytest.raises(SystemExit) as exc_info:
        run_monitoring.main()
    assert exc_info.value.code == 2

    summary_path = tmp_path / "monitoring" / "evidently_reports" / "drift_summary.json"
    assert summary_path.is_file()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["skipped"] is True
    assert payload["reason"] == "missing_required_artifacts"
    assert set(payload["missing"]) == {"model", "preprocessor", "reference", "production"}
