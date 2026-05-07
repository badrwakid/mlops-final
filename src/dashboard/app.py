from __future__ import annotations

import json
import logging
import os
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
from mlflow.tracking import MlflowClient

PAGE_TITLE = "Bike Sharing Demand MLflow Dashboard"
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_UI_URL = os.getenv("MLFLOW_UI_URL", "http://localhost:5000")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "bike_share_regressor")
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
HTTP_TIMEOUT = float(os.getenv("DASHBOARD_HTTP_TIMEOUT_SECONDS", "4"))
MAX_BATCH_SIZE = int(os.getenv("DASHBOARD_MAX_BATCH_SIZE", "100"))

PAGES = [
    "Overview",
    "Live Prediction",
    "Batch Prediction",
    "MLflow Tracking",
    "Model Registry",
    "Monitoring & Drift",
    "Documentation Evidence",
]


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


def safe_get_json(path: str) -> dict[str, Any]:
    url = f"{API_BASE_URL}{path}"
    try:
        response = requests.get(url, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        try:
            payload = response.json()
        except json.JSONDecodeError:
            LOG.warning("json decode failed for %s", url)
            return {"ok": False, "error": "Invalid JSON response", "url": url}
        return {"ok": True, "data": payload, "url": url}
    except requests.Timeout:
        LOG.warning("timeout on GET %s", url)
        return {"ok": False, "error": "Request timed out", "url": url}
    except requests.RequestException as exc:
        LOG.warning("request error on GET %s: %s", url, exc)
        return {"ok": False, "error": "Service unavailable", "url": url}


def safe_get_text(path: str) -> dict[str, Any]:
    url = f"{API_BASE_URL}{path}"
    try:
        response = requests.get(url, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        return {"ok": True, "data": response.text, "url": url}
    except requests.Timeout:
        LOG.warning("timeout on GET text %s", url)
        return {"ok": False, "error": "Request timed out", "url": url}
    except requests.RequestException as exc:
        LOG.warning("request error on GET text %s: %s", url, exc)
        return {"ok": False, "error": "Service unavailable", "url": url}


def safe_post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{API_BASE_URL}{path}"
    try:
        response = requests.post(url, json=payload, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        try:
            body = response.json()
        except json.JSONDecodeError:
            LOG.warning("json decode failed for POST %s", url)
            return {"ok": False, "error": "Invalid JSON response", "url": url}
        return {"ok": True, "data": body, "url": url}
    except requests.Timeout:
        LOG.warning("timeout on POST %s", url)
        return {"ok": False, "error": "Request timed out", "url": url}
    except requests.RequestException as exc:
        LOG.warning("request error on POST %s: %s", url, exc)
        return {"ok": False, "error": "Service unavailable", "url": url}


@st.cache_data(ttl=20)
def mlflow_status() -> dict[str, Any]:
    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        experiments = client.search_experiments()
        return {"ok": True, "tracking_uri": MLFLOW_TRACKING_URI, "experiment_count": len(experiments)}
    except Exception as exc:  # noqa: BLE001
        LOG.warning("mlflow status failed: %s", exc)
        return {"ok": False, "error": "MLflow unavailable"}


@st.cache_data(ttl=20)
def mlflow_latest_runs() -> dict[str, Any]:
    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        experiments = client.search_experiments()
        exp_ids = [e.experiment_id for e in experiments]
        runs = client.search_runs(exp_ids, max_results=20, order_by=["start_time DESC"])
        items = []
        for run in runs:
            metrics = run.data.metrics
            primary_metric = None
            for key in ("test_rmse", "rmse", "best_cv_rmse"):
                if key in metrics:
                    primary_metric = (key, metrics[key])
                    break
            items.append(
                {
                    "run_name": run.data.tags.get("mlflow.runName", "unknown"),
                    "run_id": run.info.run_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "duration_s": (
                        (run.info.end_time - run.info.start_time) / 1000.0
                        if run.info.end_time and run.info.start_time
                        else None
                    ),
                    "metric_name": primary_metric[0] if primary_metric else None,
                    "metric_value": primary_metric[1] if primary_metric else None,
                }
            )
        return {"ok": True, "items": items}
    except Exception as exc:  # noqa: BLE001
        LOG.warning("mlflow latest runs failed: %s", exc)
        return {"ok": False, "error": "MLflow runs unavailable", "items": []}


@st.cache_data(ttl=20)
def mlflow_model_registry(model_name: str) -> dict[str, Any]:
    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        versions = client.search_model_versions(f"name='{model_name}'")
        rows = []
        for v in versions:
            rows.append(
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "status": v.status,
                    "run_id": v.run_id,
                    "source": v.source,
                    "aliases": ", ".join(v.aliases) if getattr(v, "aliases", None) else "",
                }
            )
        return {"ok": True, "items": rows, "latest": rows[0] if rows else None}
    except Exception as exc:  # noqa: BLE001
        LOG.warning("mlflow registry failed: %s", exc)
        return {"ok": False, "error": "Model registry unavailable", "items": [], "latest": None}


def _inject_css() -> None:
    st.markdown(
        """
        <style>
            .stMetric {border: 1px solid #d9dde3; border-radius: 8px; padding: 8px;}
            .small-muted {color: #5f6b78; font-size: 12px;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _demo_record() -> dict[str, Any]:
    return {
        "season": 2,
        "mnth": 6,
        "hr": 12,
        "holiday": 0,
        "weekday": 3,
        "workingday": 1,
        "weathersit": 2,
        "temp": 0.62,
        "atemp": 0.60,
        "hum": 0.45,
        "windspeed": 0.20,
    }


def _status_indicator(ok: bool) -> str:
    return "Available" if ok else "Unavailable"


def page_overview() -> None:
    st.subheader("Overview")
    health = safe_get_json("/health")
    ready = safe_get_json("/ready")
    metrics = safe_get_text("/metrics")
    mlf = mlflow_status()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("API health", _status_indicator(health["ok"]))
    col2.metric("Readiness", _status_indicator(ready["ok"]))
    col3.metric("MLflow", _status_indicator(mlf["ok"]))
    col4.metric("Prometheus metrics", _status_indicator(metrics["ok"]))
    if health["ok"]:
        st.write(
            {
                "model_name": health["data"].get("model_name"),
                "model_version": health["data"].get("model_version"),
            }
        )
    runs = mlflow_latest_runs()
    if runs["ok"] and runs["items"]:
        df = pd.DataFrame(runs["items"][:15])
        st.line_chart(df["metric_value"].dropna())
        st.dataframe(df[["run_name", "status", "metric_name", "metric_value"]], use_container_width=True)
    else:
        st.info("Recent runs unavailable.")


def page_live_prediction() -> None:
    st.subheader("Live Prediction")
    if st.button("Load Safe Demo Example"):
        st.session_state["demo_payload"] = _demo_record()
    if st.button("Reset"):
        st.session_state["demo_payload"] = _demo_record()
    payload = st.session_state.get("demo_payload", _demo_record())
    c1, c2, c3 = st.columns(3)
    payload["season"] = c1.selectbox("Season", [1, 2, 3, 4], index=payload["season"] - 1)
    payload["mnth"] = c2.selectbox("Month", list(range(1, 13)), index=payload["mnth"] - 1)
    payload["hr"] = c3.selectbox("Hour", list(range(24)), index=payload["hr"])
    c4, c5, c6, c7 = st.columns(4)
    payload["holiday"] = c4.selectbox("Holiday", [0, 1], index=payload["holiday"])
    payload["weekday"] = c5.selectbox("Weekday", list(range(7)), index=payload["weekday"])
    payload["workingday"] = c6.selectbox("Working day", [0, 1], index=payload["workingday"])
    payload["weathersit"] = c7.selectbox("Weather", [1, 2, 3, 4], index=payload["weathersit"] - 1)
    payload["temp"] = st.slider("Temp (normalized)", 0.0, 1.0, float(payload["temp"]), 0.01)
    payload["atemp"] = st.slider("Atemp (normalized)", 0.0, 1.0, float(payload["atemp"]), 0.01)
    payload["hum"] = st.slider("Humidity (normalized)", 0.0, 1.0, float(payload["hum"]), 0.01)
    payload["windspeed"] = st.slider("Windspeed (normalized)", 0.0, 1.0, float(payload["windspeed"]), 0.01)
    if st.button("Run Prediction"):
        t0 = time.perf_counter()
        result = safe_post_json("/predict", payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if not result["ok"]:
            st.warning(result["error"])
            return
        data = result["data"]
        a, b, c, d = st.columns(4)
        a.metric("Predicted demand", f"{round(float(data.get('prediction', 0)))}")
        b.metric("Confidence", f"{round(float(data.get('confidence', 0))*100)}%")
        c.metric("Response time", f"{round(elapsed_ms)} ms")
        d.metric("Model version", str(data.get("model_version", "n/a")))
        st.progress(max(0.0, min(1.0, float(data.get("confidence", 0)))))
        st.bar_chart(pd.DataFrame({"feature": list(payload.keys())[-4:], "value": list(payload.values())[-4:]}).set_index("feature"))

        hourly_vals = []
        for hr in range(24):
            rec = payload.copy()
            rec["hr"] = hr
            r = safe_post_json("/predict", rec)
            if r["ok"]:
                hourly_vals.append({"hour": hr, "prediction": r["data"].get("prediction", 0)})
        if hourly_vals:
            st.line_chart(pd.DataFrame(hourly_vals).set_index("hour"))
        else:
            st.warning("Predicted demand by hour unavailable.")

        weather_vals = []
        for w in [1, 2, 3, 4]:
            rec = payload.copy()
            rec["weathersit"] = w
            r = safe_post_json("/predict", rec)
            if r["ok"]:
                weather_vals.append({"weather": w, "prediction": r["data"].get("prediction", 0)})
        if weather_vals:
            st.bar_chart(pd.DataFrame(weather_vals).set_index("weather"))
        else:
            st.warning("Predicted demand by weather unavailable.")


def page_batch_prediction() -> None:
    st.subheader("Batch Prediction")
    st.caption(f"Maximum batch size: {MAX_BATCH_SIZE}")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    jsonl_text = st.text_area("JSON-lines input", height=140)
    records: list[dict[str, Any]] = []
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            required = set(_demo_record().keys())
            if not required.issubset(df.columns):
                st.warning("CSV is missing required columns.")
            else:
                records = df[list(required)].to_dict(orient="records")
        except Exception:  # noqa: BLE001
            st.warning("Could not parse CSV.")
    elif jsonl_text.strip():
        for line in jsonl_text.strip().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                st.warning("Invalid JSON-lines input.")
                return
    if st.button("Run batch prediction"):
        if not records:
            st.warning("Batch is empty.")
            return
        if len(records) > MAX_BATCH_SIZE:
            st.warning(f"Batch exceeds max size ({MAX_BATCH_SIZE}).")
            return
        response = safe_post_json("/predict/batch", {"records": records})
        if not response["ok"]:
            st.warning(response["error"])
            return
        preds = response["data"].get("predictions", [])
        values = [float(p.get("prediction", 0)) for p in preds]
        if not values:
            st.warning("No predictions returned.")
            return
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Successful records", str(len(values)))
        c2.metric("Average prediction", f"{round(sum(values)/len(values))}")
        c3.metric("Min prediction", f"{round(min(values))}")
        c4.metric("Max prediction", f"{round(max(values))}")
        st.bar_chart(pd.DataFrame({"row": list(range(1, len(values) + 1)), "prediction": values}).set_index("row"))
        st.dataframe(pd.DataFrame(preds), use_container_width=True)


def page_mlflow_tracking() -> None:
    st.subheader("MLflow Tracking")
    status = mlflow_status()
    st.write({"tracking_uri": MLFLOW_TRACKING_URI, "status": _status_indicator(status["ok"])})
    runs = mlflow_latest_runs()
    if not runs["ok"]:
        st.info("MLflow tracking unavailable.")
        return
    df = pd.DataFrame(runs["items"])
    st.dataframe(df, use_container_width=True)
    if not df.empty and "metric_value" in df.columns:
        chart_df = df[["run_name", "metric_value"]].dropna().set_index("run_name")
        if not chart_df.empty:
            st.bar_chart(chart_df)
    st.markdown(f"[Open MLflow UI]({MLFLOW_UI_URL})")


def page_model_registry() -> None:
    st.subheader("Model Registry")
    reg = mlflow_model_registry(MLFLOW_MODEL_NAME)
    if not reg["ok"]:
        st.info("Model registry unavailable.")
        return
    st.metric("Registered model", MLFLOW_MODEL_NAME)
    if reg["latest"]:
        st.write({"latest_version": reg["latest"]["version"], "stage": reg["latest"]["stage"]})
    st.dataframe(pd.DataFrame(reg["items"]), use_container_width=True)


def _file_status(path: Path) -> dict[str, Any]:
    return {"file": str(path.relative_to(PROJECT_ROOT)), "available": path.exists(), "path": path}


def page_monitoring_drift() -> None:
    st.subheader("Monitoring & Drift")
    metrics = safe_get_text("/metrics")
    st.write({"prometheus_metrics_status": _status_indicator(metrics["ok"])})
    st.markdown(f"[Prometheus UI]({PROMETHEUS_URL})")
    files = [
        PROJECT_ROOT / "monitoring/evidently_reports/drift_summary.json",
        PROJECT_ROOT / "monitoring/evidently_reports/baseline.html",
        PROJECT_ROOT / "monitoring/evidently_reports/drift.html",
        PROJECT_ROOT / "monitoring/evidently_reports/interpretation.md",
    ]
    for item in [_file_status(f) for f in files]:
        st.write({"file": item["file"], "status": _status_indicator(item["available"])})
        if item["available"]:
            st.download_button(
                label=f"Download {item['file']}",
                data=item["path"].read_bytes(),
                file_name=item["path"].name,
                key=f"dl-{item['file']}",
            )


def page_documentation_evidence() -> None:
    st.subheader("Documentation Evidence")
    files = [
        "README.md",
        "docs/technical_report.md",
        "docs/model_card.md",
        "docs/data_card.md",
        "docs/experiment_log.csv",
        "docs/mlflow/export.md",
        "dvc.yaml",
        "dvc.lock",
        "configs/params.yaml",
        ".github/workflows/ci.yml",
    ]
    rows = []
    for rel in files:
        p = PROJECT_ROOT / rel
        rows.append({"file": rel, "status": _status_indicator(p.exists())})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    selected = st.selectbox("Preview file", files)
    p = PROJECT_ROOT / selected
    if p.exists():
        if p.suffix.lower() in {".md", ".yml", ".yaml", ".txt", ".csv"}:
            text = p.read_text(encoding="utf-8", errors="ignore")
            st.text_area("Preview", text[:3000], height=320)
        st.download_button(f"Download {selected}", p.read_bytes(), file_name=p.name)


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    _inject_css()
    st.title(PAGE_TITLE)
    st.caption("Production-style Streamlit dashboard with safe unavailable states.")
    page = st.sidebar.radio("Navigation", PAGES)
    try:
        if page == "Overview":
            page_overview()
        elif page == "Live Prediction":
            page_live_prediction()
        elif page == "Batch Prediction":
            page_batch_prediction()
        elif page == "MLflow Tracking":
            page_mlflow_tracking()
        elif page == "Model Registry":
            page_model_registry()
        elif page == "Monitoring & Drift":
            page_monitoring_drift()
        elif page == "Documentation Evidence":
            page_documentation_evidence()
    except Exception as exc:  # noqa: BLE001
        LOG.exception("unhandled dashboard section error: %s", exc)
        st.error("Section unavailable right now. Please try again.")


if __name__ == "__main__":
    main()

