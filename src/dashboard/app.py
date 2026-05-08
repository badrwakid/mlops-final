from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
import yaml

PAGE_TITLE = "BikeOps Data Monitoring & Drift Detection Dashboard"
PAGE_SUBTITLE = (
    "Operational monitoring dashboard for Evidently drift reports, Prometheus metrics, "
    "and retraining decisions."
)
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
HTTP_TIMEOUT = float(os.getenv("DASHBOARD_HTTP_TIMEOUT_SECONDS", "4"))
DRIFT_SUMMARY_PATH = Path(
    os.getenv("DRIFT_SUMMARY_PATH", "monitoring/evidently_reports/drift_summary.json")
)
EVIDENTLY_REPORTS_DIR = Path(
    os.getenv("EVIDENTLY_REPORTS_DIR", "monitoring/evidently_reports")
)
MLFLOW_UI_URL = os.getenv("MLFLOW_UI_URL", "http://localhost:5001")

PAGES = [
    "Drift Overview",
    "Feature Drift Analysis",
    "Evidently Reports",
    "Prometheus Monitoring",
    "Monitoring Runbook",
    "Model & Data Context",
    "Documentation Evidence",
]

REPORT_PURPOSES = {
    "baseline.html": "Reference-vs-reference sanity report.",
    "drift.html": "Reference-vs-production data drift report.",
    "drift_summary.json": "Machine-readable drift alert payload.",
    "interpretation.md": "Monitoring policy and response runbook.",
    "drift_synthetic_demo.html": "Synthetic drift demo artifact (if generated).",
}


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("streamlit_dashboard")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        log_dir / "streamlit_dashboard.log",
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


LOG = _setup_logger()


def _inject_css() -> None:
    st.markdown(
        """
        <style>
            .stMetric {border: 1px solid #d9dde3; border-radius: 8px; padding: 8px; background: #fcfdff;}
            .status-card {
                border: 1px solid #d9dde3; border-radius: 8px; padding: 10px 12px; margin: 6px 0;
                background: #ffffff;
            }
            .status-title {font-size: 12px; color: #5f6b78; margin: 0;}
            .status-value {font-size: 16px; font-weight: 600; margin: 2px 0 0 0;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def percent(value: float | None) -> str:
    if value is None:
        return "Unavailable"
    return f"{value * 100:.2f}%"


def status_label(ok: bool) -> str:
    return "Available" if ok else "Unavailable"


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


@st.cache_data(ttl=30)
def load_json_file(path: Path) -> dict[str, Any]:
    resolved = _resolve_path(path)
    if not resolved.exists():
        return {"ok": False, "error": "File missing", "path": str(resolved), "data": {}}
    try:
        raw = resolved.read_text(encoding="utf-8", errors="ignore")
        return {"ok": True, "path": str(resolved), "data": json.loads(raw)}
    except json.JSONDecodeError:
        LOG.warning("malformed json file: %s", resolved)
        return {"ok": False, "error": "Malformed JSON", "path": str(resolved), "data": {}}
    except OSError as exc:
        LOG.warning("json read failed for %s: %s", resolved, exc)
        return {"ok": False, "error": "Read failure", "path": str(resolved), "data": {}}


@st.cache_data(ttl=30)
def load_text_file(path: Path) -> dict[str, Any]:
    resolved = _resolve_path(path)
    if not resolved.exists():
        return {"ok": False, "error": "File missing", "path": str(resolved), "text": ""}
    try:
        return {
            "ok": True,
            "path": str(resolved),
            "text": resolved.read_text(encoding="utf-8", errors="ignore"),
        }
    except OSError as exc:
        LOG.warning("text read failed for %s: %s", resolved, exc)
        return {"ok": False, "error": "Read failure", "path": str(resolved), "text": ""}


@st.cache_data(ttl=30)
def file_metadata(path: Path) -> dict[str, Any]:
    resolved = _resolve_path(path)
    if not resolved.exists():
        return {
            "ok": False,
            "exists": False,
            "path": str(resolved),
            "size_bytes": None,
            "modified_at": None,
        }
    try:
        stat = resolved.stat()
        return {
            "ok": True,
            "exists": True,
            "path": str(resolved),
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        }
    except OSError as exc:
        LOG.warning("metadata read failed for %s: %s", resolved, exc)
        return {
            "ok": False,
            "exists": True,
            "path": str(resolved),
            "size_bytes": None,
            "modified_at": None,
        }


@st.cache_data(ttl=30)
def load_params() -> dict[str, Any]:
    resolved = PROJECT_ROOT / "configs/params.yaml"
    if not resolved.exists():
        return {"ok": False, "error": "File missing", "data": {}}
    try:
        data = yaml.safe_load(resolved.read_text(encoding="utf-8", errors="ignore")) or {}
        if not isinstance(data, dict):
            return {"ok": False, "error": "Malformed YAML root", "data": {}}
        return {"ok": True, "data": data}
    except yaml.YAMLError as exc:
        LOG.warning("params yaml parse failed: %s", exc)
        return {"ok": False, "error": "Malformed YAML", "data": {}}
    except OSError as exc:
        LOG.warning("params read failed: %s", exc)
        return {"ok": False, "error": "Read failure", "data": {}}


@st.cache_data(ttl=30)
def load_drift_summary() -> dict[str, Any]:
    return load_json_file(DRIFT_SUMMARY_PATH)


@st.cache_data(ttl=30)
def load_feature_scores() -> dict[str, Any]:
    raw = load_json_file(Path("docs/feature_scores.json"))
    if not raw["ok"]:
        return {"ok": False, "error": raw["error"], "df": pd.DataFrame(), "k_selected": None}
    rows = raw["data"].get("all_features_ranked_by_f_score", [])
    if not isinstance(rows, list):
        return {"ok": False, "error": "Malformed feature score structure", "df": pd.DataFrame(), "k_selected": None}
    try:
        df = pd.DataFrame(rows)
        if {"feature", "f_score"}.issubset(df.columns):
            df = df[["feature", "f_score"]].copy()
            df["f_score"] = pd.to_numeric(df["f_score"], errors="coerce")
        else:
            df = pd.DataFrame(columns=["feature", "f_score"])
        return {"ok": True, "df": df, "k_selected": raw["data"].get("k_selected")}
    except Exception as exc:  # noqa: BLE001
        LOG.warning("feature score dataframe build failed: %s", exc)
        return {"ok": False, "error": "Could not parse feature scores", "df": pd.DataFrame(), "k_selected": None}


@st.cache_data(ttl=10)
def safe_get_json(path: str) -> dict[str, Any]:
    url = f"{API_BASE_URL}{path}"
    try:
        response = requests.get(url, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        try:
            payload = response.json()
        except json.JSONDecodeError:
            LOG.warning("json decode failed for %s", url)
            return {"ok": False, "error": "Invalid JSON response", "url": url, "data": {}}
        return {"ok": True, "data": payload, "url": url}
    except requests.Timeout:
        LOG.warning("timeout on GET %s", url)
        return {"ok": False, "error": "Request timed out", "url": url, "data": {}}
    except requests.RequestException as exc:
        LOG.warning("request error on GET %s: %s", url, exc)
        return {"ok": False, "error": "Service unavailable", "url": url, "data": {}}


@st.cache_data(ttl=10)
def safe_get_text(path: str) -> dict[str, Any]:
    url = f"{API_BASE_URL}{path}"
    try:
        response = requests.get(url, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        return {"ok": True, "data": response.text, "url": url}
    except requests.Timeout:
        LOG.warning("timeout on GET text %s", url)
        return {"ok": False, "error": "Request timed out", "url": url, "data": ""}
    except requests.RequestException as exc:
        LOG.warning("request error on GET text %s: %s", url, exc)
        return {"ok": False, "error": "Service unavailable", "url": url, "data": ""}


def parse_prometheus_metrics(text: str) -> dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    metric_names = []
    for line in lines:
        if line.startswith("#"):
            continue
        if " " in line:
            metric_names.append(line.split(" ", 1)[0].split("{", 1)[0])
    metric_names = sorted(set(metric_names))
    themes = {
        "latency": "latency",
        "confidence": "confidence",
        "model": "model",
        "inference": "inference",
        "drift": "drift",
        "prediction": "prediction",
    }
    rows = []
    for theme, needle in themes.items():
        matched = [name for name in metric_names if needle in name.lower()]
        rows.append(
            {
                "metric theme": theme,
                "found": "yes" if matched else "no",
                "matching metric names": ", ".join(matched[:6]) if matched else "",
            }
        )
    return {
        "metric_names": metric_names,
        "availability_df": pd.DataFrame(rows),
        "preview_lines": lines[:100],
    }


def render_status_card(title: str, value: str, note: str = "") -> None:
    safe_note = f"<div class='small-muted'>{note}</div>" if note else ""
    st.markdown(
        (
            "<div class='status-card'>"
            f"<p class='status-title'>{title}</p>"
            f"<p class='status-value'>{value}</p>"
            f"{safe_note}</div>"
        ),
        unsafe_allow_html=True,
    )


def render_file_card(filename: str, purpose: str, rel_path: Path) -> None:
    meta = file_metadata(rel_path)
    status = "Available" if meta["exists"] else "Missing"
    render_status_card(
        title=filename,
        value=status,
        note=f"{purpose} | {meta['path']}",
    )
    c1, c2 = st.columns([3, 2])
    c1.write(
        {
            "last_modified": meta["modified_at"] or "Unavailable",
            "size_bytes": meta["size_bytes"] if meta["size_bytes"] is not None else "Unavailable",
        }
    )
    if meta["exists"]:
        resolved = _resolve_path(rel_path)
        try:
            c2.download_button(
                f"Download {filename}",
                resolved.read_bytes(),
                file_name=resolved.name,
                key=f"download-{filename}",
            )
        except OSError as exc:
            LOG.warning("download read failed for %s: %s", resolved, exc)
            c2.warning("File available but unreadable.")


def _drift_threshold(summary: dict[str, Any], params: dict[str, Any]) -> float | None:
    threshold = summary.get("threshold")
    if isinstance(threshold, int | float):
        return float(threshold)
    fallback = params.get("drift", {}).get("drift_threshold_share")
    if isinstance(fallback, int | float):
        return float(fallback)
    return None


def _drift_share(summary: dict[str, Any]) -> float | None:
    for key in ("drift_share_inputs_only", "drift_share_inputs_only_unweighted"):
        val = summary.get(key)
        if isinstance(val, int | float):
            return float(val)
    return None


def _progress_fraction(share: float | None, threshold: float | None) -> float:
    if share is None or threshold is None or threshold <= 0:
        return 0.0
    return max(0.0, min(1.0, share / threshold))


def _status_from_threshold(share: float | None, threshold: float | None) -> str:
    if share is None or threshold is None:
        return "Unavailable"
    if share > threshold:
        return "Above threshold"
    if share >= (0.9 * threshold):
        return "Near threshold"
    return "Below threshold"


def _feature_base_name(name: str) -> str:
    stripped = name
    if "__" in stripped:
        stripped = stripped.split("__", 1)[1]
    for suffix in ("_sin", "_cos"):
        if stripped.endswith(suffix):
            stripped = stripped[: -len(suffix)]
    if "_" in stripped:
        first = stripped.split("_", 1)[0]
        if first in {"season", "weathersit", "weekday", "holiday", "workingday"}:
            stripped = first
    return stripped


def _feature_group(feature: str, numeric: list[str], categorical: list[str]) -> str:
    if feature in numeric:
        return "numeric"
    if feature in categorical:
        return "categorical"
    return "unknown"


def _severity_from_score(score: float | None) -> str:
    if score is None:
        return "Medium"
    if score >= 100:
        return "High"
    return "Medium"


def page_drift_overview() -> None:
    st.subheader("Drift Overview")
    summary_raw = load_drift_summary()
    params_pack = load_params()
    params = params_pack["data"] if params_pack["ok"] else {}
    summary = summary_raw["data"] if summary_raw["ok"] else {}

    share = _drift_share(summary)
    weighted = summary.get("drift_share_weighted_inputs")
    threshold = _drift_threshold(summary, params)
    alert = summary.get("alert")
    drifted = summary.get("drifted_features", [])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Drift Alert", "Active" if alert is True else ("Clear" if alert is False else "Unavailable"))
    c2.metric("Severity", str(summary.get("severity", "Unavailable")))
    c3.metric("Input Drift Share", percent(share))
    c4.metric(
        "Weighted Input Drift Share",
        percent(float(weighted)) if isinstance(weighted, int | float) else "Unavailable",
    )

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Threshold", percent(threshold))
    c6.metric("Drifted Feature Count", str(len(drifted) if isinstance(drifted, list) else "Unavailable"))
    c7.metric("Monitoring Mode", str(summary.get("monitoring_mode", "Unavailable")))
    c8.metric(
        "Synthetic Demo Status",
        str(summary.get("synthetic_demo_report_generated", "Unavailable")),
    )

    if summary_raw["ok"] and alert is True and share is not None and threshold is not None and share > threshold:
        st.error("Action Required: Investigate drift before trusting new predictions.")
    elif summary_raw["ok"] and alert is False:
        st.success("No action required: continue monitoring.")
    else:
        st.warning("Monitoring summary unavailable: run python -m monitoring.run_monitoring.")

    st.markdown("### Drift Share vs Threshold")
    progress = _progress_fraction(share, threshold)
    st.progress(progress)
    st.write(
        {
            "input_drift_share": percent(share),
            "threshold": percent(threshold),
            "status": _status_from_threshold(share, threshold),
        }
    )

    if not summary_raw["ok"]:
        st.info(
            "drift_summary.json is missing or invalid. "
            "Run: python -m monitoring.run_monitoring"
        )


def page_feature_drift_analysis() -> None:
    st.subheader("Feature Drift Analysis")
    summary = load_drift_summary()
    params_pack = load_params()
    params = params_pack["data"] if params_pack["ok"] else {}
    numeric = params.get("numeric_features", []) if isinstance(params, dict) else []
    categorical = params.get("categorical_features", []) if isinstance(params, dict) else []
    perturbed = params.get("drift", {}).get("perturbed_features", [])
    if not isinstance(perturbed, list):
        perturbed = []

    drifted_features = []
    if summary["ok"] and isinstance(summary["data"].get("drifted_features"), list):
        drifted_features = summary["data"]["drifted_features"]

    st.markdown("### Drifted Features Bar Chart")
    if drifted_features:
        drift_df = pd.DataFrame(
            {"feature": drifted_features, "drift_flag": [1] * len(drifted_features)}
        ).set_index("feature")
        st.caption("Drift Flag by Feature (1 = drifted)")
        st.bar_chart(drift_df)
    else:
        st.info("No drifted feature list available.")

    score_pack = load_feature_scores()
    score_df = score_pack["df"] if score_pack["ok"] else pd.DataFrame(columns=["feature", "f_score"])
    score_lookup: dict[str, float] = {}
    if not score_df.empty:
        for _, row in score_df.iterrows():
            base = _feature_base_name(str(row["feature"]))
            score_lookup[base] = max(float(row["f_score"]), score_lookup.get(base, 0.0))

    all_features = sorted(set(numeric + categorical))
    rows = []
    for feature in all_features:
        drifted = feature in drifted_features
        score = score_lookup.get(feature)
        priority = _severity_from_score(score) if drifted else "Normal"
        rows.append(
            {
                "feature": feature,
                "feature_group": _feature_group(feature, numeric, categorical),
                "drifted": "yes" if drifted else "no",
                "importance_score": score if score is not None else "Unavailable",
                "monitoring_priority": priority,
            }
        )
    st.markdown("### Drifted vs Non-Drifted Feature Table")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### Feature Importance vs Drift")
    if not score_df.empty:
        top = score_df.copy()
        top["base_feature"] = top["feature"].apply(_feature_base_name)
        top["drifted"] = top["base_feature"].apply(lambda f: "yes" if f in drifted_features else "no")
        top = top.sort_values("f_score", ascending=False).head(15)
        st.dataframe(top[["feature", "base_feature", "f_score", "drifted"]], use_container_width=True)
        st.bar_chart(top.set_index("feature")[["f_score"]])
    else:
        st.info("Feature score file unavailable: docs/feature_scores.json")

    st.markdown("### Drift Feature Explanation Cards")
    if not drifted_features:
        st.info("No drifted features to explain.")
        return
    for feature in drifted_features:
        group = _feature_group(feature, numeric, categorical)
        perturbed_flag = "yes" if feature in perturbed else "no"
        top_score_flag = "yes" if feature in score_lookup else "no"
        if feature == "hum":
            suggestion = "Check humidity distribution between reference and production windows."
        elif feature == "season":
            suggestion = "Check seasonal split effects or categorical distribution shift."
        elif feature == "weathersit":
            suggestion = "Check weather category distribution and encoding consistency."
        else:
            suggestion = "Inspect distribution shift and upstream feature pipeline consistency."
        render_status_card(
            title=feature,
            value=f"{group} | perturbed/demo={perturbed_flag} | in feature scores={top_score_flag}",
            note=suggestion,
        )


def page_evidently_reports() -> None:
    st.subheader("Evidently Reports")
    for name, purpose in REPORT_PURPOSES.items():
        render_file_card(name, purpose, EVIDENTLY_REPORTS_DIR / name)

    st.markdown("### How to regenerate monitoring reports")
    st.code("python -m monitoring.run_monitoring")

    st.markdown("### Required Artifacts")
    required_artifacts = [
        "data/splits/model.pkl",
        "data/splits/preprocessor.pkl",
        "data/splits/reference.parquet",
        "data/splits/production.parquet",
    ]
    rows = []
    for rel in required_artifacts:
        meta = file_metadata(Path(rel))
        rows.append(
            {
                "artifact": rel,
                "exists": "yes" if meta["exists"] else "no",
                "last_modified": meta["modified_at"] or "Unavailable",
                "size_bytes": meta["size_bytes"] if meta["size_bytes"] is not None else "Unavailable",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def page_prometheus_monitoring() -> None:
    st.subheader("Prometheus Monitoring")
    metrics_res = safe_get_text("/metrics")
    render_status_card("API /metrics endpoint", status_label(metrics_res["ok"]), metrics_res["url"])
    st.markdown(f"[Open Prometheus UI]({PROMETHEUS_URL})")

    if not metrics_res["ok"]:
        st.warning("Metrics endpoint unavailable. API may be down or not reachable.")
        return
    parsed = parse_prometheus_metrics(metrics_res["data"])
    st.markdown("### Metrics Availability Table")
    st.dataframe(parsed["availability_df"], use_container_width=True, hide_index=True)

    with st.expander("Raw Metrics Preview (first 100 lines)"):
        if parsed["preview_lines"]:
            st.code("\n".join(parsed["preview_lines"]))
        else:
            st.info("Metrics text is empty.")


def page_monitoring_runbook() -> None:
    st.subheader("Monitoring Runbook")
    summary = load_drift_summary()
    interpretation = load_text_file(EVIDENTLY_REPORTS_DIR / "interpretation.md")
    params = load_params()

    threshold = _drift_threshold(summary.get("data", {}), params.get("data", {}))
    st.write(
        {
            "policy": summary.get("data", {}).get("policy", "Unavailable"),
            "policy_note": summary.get("data", {}).get("policy_note", "Unavailable"),
            "threshold": percent(threshold),
            "severity": summary.get("data", {}).get("severity", "Unavailable"),
            "excluded_from_share": summary.get("data", {}).get("excluded_from_share", "Unavailable"),
        }
    )

    st.markdown("### Decision Workflow")
    cols = st.columns(3)
    steps = [
        ("1. Detect", "Read drift_summary.json."),
        ("2. Investigate", "Open drift.html and inspect drifted features."),
        ("3. Assess", "Compare drifted features with docs/feature_scores.json."),
        ("4. Decide", "Above threshold once: P2 investigation; two consecutive windows: retrain."),
        ("5. Validate", "Run DVC training and validation gates."),
        ("6. Promote", "Promote only if metrics pass."),
    ]
    for i, (title, note) in enumerate(steps):
        with cols[i % 3]:
            render_status_card(title, "Step", note)

    st.info("Consecutive-window enforcement requires external run history/state.")

    st.markdown("### Runbook Content Preview")
    if interpretation["ok"]:
        st.text_area("interpretation.md", interpretation["text"][:4000], height=320)
    else:
        st.warning("interpretation.md unavailable.")


def page_model_data_context() -> None:
    st.subheader("Model & Data Context")
    params = load_params()
    pdata = params.get("data", {})
    ppaths = params.get("paths", {})
    pdrift = params.get("drift", {})
    pserving = params.get("serving", {})

    st.write(
        {
            "target_variable": pdata.get("target", "Unavailable"),
            "split_column": pdata.get("split_column", "Unavailable"),
            "reference_dataset_path": ppaths.get("reference", "Unavailable"),
            "production_dataset_path": ppaths.get("production", "Unavailable"),
            "model_name": pserving.get("model_name", "Unavailable"),
            "serving_stage": pserving.get("model_stage", "Unavailable"),
        }
    )

    st.markdown("### Feature Group Table")
    rows = []
    for feature in pdata.get("numeric_features", []):
        rows.append({"feature": feature, "group": "numeric"})
    for feature in pdata.get("categorical_features", []):
        rows.append({"feature": feature, "group": "categorical"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### Drift Configuration")
    render_status_card(
        "Drift config",
        "Loaded" if params["ok"] else "Unavailable",
        (
            f"threshold={percent(float(pdrift['drift_threshold_share'])) if isinstance(pdrift.get('drift_threshold_share'), int | float) else 'Unavailable'} | "
            f"perturbed_features={pdrift.get('perturbed_features', 'Unavailable')} | "
            f"synthetic_demo_report_name={pdrift.get('synthetic_demo_report_name', 'Unavailable')}"
        ),
    )

    st.markdown("### Model Performance Context")
    metrics = load_json_file(Path("data/splits/metrics.json"))
    if metrics["ok"]:
        test = metrics["data"].get("test", {})
        gate = metrics["data"].get("gate", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RMSE", f"{test.get('rmse', 'Unavailable')}")
        c2.metric("MAE", f"{test.get('mae', 'Unavailable')}")
        c3.metric("R2", f"{test.get('r2', 'Unavailable')}")
        c4.metric("Validation gate", "Passed" if gate.get("passed") is True else str(gate.get("passed", "Unavailable")))
    else:
        st.warning("Model metrics unavailable: data/splits/metrics.json")

    subgroup = load_json_file(Path("docs/subgroup_metrics.json"))
    if subgroup["ok"]:
        with st.expander("Subgroup metrics snapshot"):
            st.json(subgroup["data"])
    else:
        st.info("Subgroup metrics unavailable: docs/subgroup_metrics.json")


def page_documentation_evidence() -> None:
    st.subheader("Documentation Evidence")
    docs = [
        ("README.md", "Project setup and monitoring usage."),
        ("docs/technical_report.md", "Technical evidence for pipeline and monitoring."),
        ("docs/model_card.md", "Model limitations and intended use."),
        ("docs/data_card.md", "Dataset, splits, and schema context."),
        ("monitoring/evidently_reports/interpretation.md", "Runbook and policy."),
        ("monitoring/evidently_reports/drift_summary.json", "Latest drift alert payload."),
        ("monitoring/evidently_reports/baseline.html", "Baseline report."),
        ("monitoring/evidently_reports/drift.html", "Drift report."),
        ("monitoring/prometheus/prometheus.yml", "Prometheus scrape config."),
        ("configs/params.yaml", "Monitoring and model configuration."),
    ]
    rows = []
    for rel, purpose in docs:
        meta = file_metadata(Path(rel))
        rows.append(
            {
                "file": rel,
                "exists": "yes" if meta["exists"] else "no",
                "purpose": purpose,
                "last_modified": meta["modified_at"] or "Unavailable",
                "size_bytes": meta["size_bytes"] if meta["size_bytes"] is not None else "Unavailable",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    selected = st.selectbox("Download file", [rel for rel, _ in docs])
    meta = file_metadata(Path(selected))
    if meta["exists"]:
        resolved = _resolve_path(Path(selected))
        try:
            st.download_button(
                f"Download {selected}",
                resolved.read_bytes(),
                file_name=resolved.name,
            )
        except OSError:
            st.warning("File exists but could not be read for download.")


def _render_sidebar() -> str:
    st.sidebar.markdown("## BikeOps Monitoring")
    if st.sidebar.button("Refresh monitoring state"):
        st.cache_data.clear()
        st.rerun()
    st.sidebar.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    health = safe_get_json("/health")
    ready = safe_get_json("/ready")
    summary = load_drift_summary()
    alert = summary.get("data", {}).get("alert") if summary["ok"] else None

    st.sidebar.markdown("### Status")
    st.sidebar.write({"API health": status_label(health["ok"])})
    st.sidebar.write({"API readiness": status_label(ready["ok"])})
    st.sidebar.write({"Drift summary": status_label(summary["ok"])})
    st.sidebar.write(
        {"Alert active": "yes" if alert is True else ("no" if alert is False else "Unavailable")}
    )
    st.sidebar.markdown("### Links")
    st.sidebar.markdown(f"- [API health]({API_BASE_URL}/health)")
    st.sidebar.markdown(f"- [API metrics]({API_BASE_URL}/metrics)")
    st.sidebar.markdown(f"- [Prometheus]({PROMETHEUS_URL})")
    st.sidebar.markdown(f"- [MLflow UI]({MLFLOW_UI_URL})")
    return st.sidebar.radio("Navigation", PAGES)


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    _inject_css()
    st.title(PAGE_TITLE)
    st.caption(PAGE_SUBTITLE)
    page = _render_sidebar()
    try:
        if page == "Drift Overview":
            page_drift_overview()
        elif page == "Feature Drift Analysis":
            page_feature_drift_analysis()
        elif page == "Evidently Reports":
            page_evidently_reports()
        elif page == "Prometheus Monitoring":
            page_prometheus_monitoring()
        elif page == "Monitoring Runbook":
            page_monitoring_runbook()
        elif page == "Model & Data Context":
            page_model_data_context()
        elif page == "Documentation Evidence":
            page_documentation_evidence()
    except Exception as exc:  # noqa: BLE001
        LOG.exception("unhandled dashboard section error: %s", exc)
        st.error("Section unavailable right now. Please try again.")


if __name__ == "__main__":
    main()

