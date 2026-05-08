from __future__ import annotations

import html
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "monitoring" / "evidently_reports"
DRIFT_SUMMARY_PATH = REPORTS_DIR / "drift_summary.json"
DEFAULT_REPORTS = ("baseline.html", "drift.html")


class _IframeSrcDocParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.srcdoc: str = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "iframe" or self.srcdoc:
            return
        attrs_map = dict(attrs)
        self.srcdoc = attrs_map.get("srcdoc") or ""


def _extract_dashboard_payload(html_text: str) -> dict[str, Any] | None:
    parser = _IframeSrcDocParser()
    parser.feed(html_text)
    if not parser.srcdoc:
        return None
    inner = html.unescape(parser.srcdoc)
    match = re.search(
        r"var\s+evidently_dashboard_[^=]+\s*=\s*(\{.*?\});\s*var\s+additional_graphs_",
        inner,
        flags=re.S,
    )
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def _extract_counter_map(payload: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for widget in payload.get("widgets", []):
        params = widget.get("params") or {}
        if not isinstance(params, dict):
            continue
        counters = params.get("counters", [])
        if not isinstance(counters, list):
            continue
        for counter in counters:
            label = str(counter.get("label", "")).strip()
            value = str(counter.get("value", "")).strip()
            if label:
                result[label] = value
    return result


def _pick_counter(counter_map: dict[str, str], needle: str) -> str | None:
    needle_lower = needle.lower()
    for label, value in counter_map.items():
        if needle_lower in label.lower():
            return value
    return None


def summarize_report(report_path: Path) -> dict[str, Any]:
    if not report_path.exists():
        return {
            "report": report_path.name,
            "available": False,
            "error": "File missing",
        }
    payload = _extract_dashboard_payload(
        report_path.read_text(encoding="utf-8", errors="ignore")
    )
    if payload is None:
        return {
            "report": report_path.name,
            "available": True,
            "error": "Could not parse embedded Evidently payload",
        }

    counter_map = _extract_counter_map(payload)
    return {
        "report": report_path.name,
        "available": True,
        "dataset_drift_state": _pick_counter(counter_map, "Dataset Drift"),
        "columns": _pick_counter(counter_map, "Columns"),
        "drifted_columns": _pick_counter(counter_map, "Drifted Columns"),
        "share_of_drifted_columns": _pick_counter(counter_map, "Share of Drifted Columns"),
        "counter_labels": sorted(counter_map.keys()),
    }


def _load_drift_summary() -> dict[str, Any]:
    if not DRIFT_SUMMARY_PATH.exists():
        return {}
    try:
        return json.loads(DRIFT_SUMMARY_PATH.read_text(encoding="utf-8", errors="ignore"))
    except json.JSONDecodeError:
        return {}


def main() -> None:
    summaries = [summarize_report(REPORTS_DIR / name) for name in DEFAULT_REPORTS]
    drift_summary = _load_drift_summary()

    output = {
        "reports_dir": str(REPORTS_DIR),
        "reports": summaries,
        "drift_summary": {
            "alert": drift_summary.get("alert"),
            "threshold": drift_summary.get("threshold"),
            "drift_share_inputs_only": drift_summary.get("drift_share_inputs_only"),
            "drifted_features": drift_summary.get("drifted_features"),
            "severity": drift_summary.get("severity"),
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

