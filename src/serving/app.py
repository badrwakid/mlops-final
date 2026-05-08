from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.config import Config, load_config
from src.drift_report import run_and_log
from src.prediction_log import log_prediction
from src.serving.metrics import (
    DATA_DRIFT_FLAG,
    FEATURE_HR,
    FEATURE_TEMP,
    INFERENCE_COUNT,
    MODEL_REGISTRY_SATISFIED,
    MODEL_VERSION,
    PREDICTION_CONFIDENCE,
    PREDICTION_LATENCY,
    PREDICTION_VALUE,
)
from src.serving.schemas import (
    BatchPredictionItem,
    BatchPredictRequest,
    BatchPredictResponse,
    BikeRecord,
    HealthResponse,
    PredictResponse,
)

log = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parents[2]
DRIFT_SUMMARY_PATH = ROOT_DIR / "monitoring" / "evidently_reports" / "drift_summary.json"
PREDICTION_LOG_PATH = ROOT_DIR / "artifacts" / "prediction_log.csv"
DRIFT_EXPERIMENT_NAME = "bike_sharing_drift"


def reference_dataset_path() -> Path:
    """Drift baseline reference split (DVC: data/splits/reference.parquet)."""
    cfg = load_config()
    p = Path(cfg.paths.reference)
    return p if p.is_absolute() else (ROOT_DIR / p)


DASHBOARD_URL = "/dashboard"
MAX_BATCH_SIZE = 100
RECENT_HISTORY_LIMIT = 20
# Reject JSON bodies larger than this (batch abuse / accidental huge uploads).
MAX_REQUEST_BODY_BYTES = int(os.environ.get("MAX_REQUEST_BODY_BYTES", str(256 * 1024)))

# Model resolution paths (see LoadedModel.load_source, /api/model-info, Prometheus bike_model_registry_load_satisfied).
LOAD_SOURCE_REGISTRY = "registry_production"
LOAD_SOURCE_LOCAL_FALLBACK = "local_pickle_fallback"
LOAD_SOURCE_LOCAL_ONLY_ENV = "local_only_env"

_REMEDIATION_REGISTRY_FAILURE = (
    "Mount the same MLflow artifact volume on api and mlflow (docker-compose mlflow-data); "
    "seed Production with: MLFLOW_TRACKING_URI=http://127.0.0.1:5001 PYTHONPATH=. "
    "python scripts/seed_mlflow_production.py - or train/register against this tracking URI."
)


def _mlflow_public_ui_url() -> str:
    """Browser-facing MLflow UI URL (may differ from MLFLOW_TRACKING_URI in Docker)."""
    return os.environ.get("MLFLOW_PUBLIC_UI_URL", "http://localhost:5000").rstrip("/")


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _default_stats() -> dict[str, float | int | None]:
    return {
        "total_predictions": 0,
        "latest_prediction": None,
        "latest_confidence": None,
        "avg_latency_ms": None,
    }


@dataclass(frozen=True)
class ModelLoadMeta:
    load_source: str
    fallback_reason: str | None = None
    remediation: str | None = None


@dataclass
class LoadedModel:
    model: Any
    preprocessor: Any
    model_name: str
    model_version: str
    load_source: str = LOAD_SOURCE_REGISTRY
    fallback_reason: str | None = None
    remediation: str | None = None


def _resolve_tracking_uri(cfg: Config) -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or cfg.mlflow.tracking_uri


def _feature_columns(cfg: Config) -> list[str]:
    return cfg.data.numeric_features + cfg.data.categorical_features


def _strict_production_mode() -> bool:
    """Hard production: MLflow Production registry only (no pickle fallback) and fatal startup on load failure."""
    return _env_truthy("PRODUCTION_STRICT") or _env_truthy("REQUIRE_REGISTRY_MODEL")


def _production_display_version(cfg: Config, registered_name: str) -> str:
    try:
        client = MlflowClient(_resolve_tracking_uri(cfg))
        versions = client.get_latest_versions(registered_name, stages=["Production"])
        if versions:
            return f"Production:v{versions[0].version}"
    except Exception:
        log.debug("could not read Production version label from registry", exc_info=True)
    return "Production"


def _load_model(cfg: Config) -> tuple[Any, str, ModelLoadMeta]:
    registered_name = cfg.mlflow.registered_model_name
    model_uri = f"models:/{registered_name}/Production"
    mlflow.set_tracking_uri(_resolve_tracking_uri(cfg))

    if _env_truthy("SERVE_USE_LOCAL_MODEL_ONLY") and _strict_production_mode():
        raise RuntimeError(
            "SERVE_USE_LOCAL_MODEL_ONLY is incompatible with PRODUCTION_STRICT / REQUIRE_REGISTRY_MODEL"
        )

    if _env_truthy("SERVE_USE_LOCAL_MODEL_ONLY"):
        log.warning(
            "SERVE_USE_LOCAL_MODEL_ONLY: serving from local pickle only (%s), not MLflow Production",
            cfg.paths.model,
        )
        model = joblib.load(cfg.paths.model)
        return (
            model,
            "local",
            ModelLoadMeta(
                load_source=LOAD_SOURCE_LOCAL_ONLY_ENV,
                fallback_reason="SERVE_USE_LOCAL_MODEL_ONLY=1",
                remediation="Unset SERVE_USE_LOCAL_MODEL_ONLY for registry-backed production serving.",
            ),
        )

    try:
        model = mlflow.sklearn.load_model(model_uri)
        ver_label = _production_display_version(cfg, registered_name)
        log.info("Inference model loaded from MLflow registry %s (%s)", model_uri, ver_label)
        return (
            model,
            ver_label,
            ModelLoadMeta(load_source=LOAD_SOURCE_REGISTRY),
        )
    except (MlflowException, OSError) as exc:
        detail = f"{type(exc).__name__}: {exc}"
        if _strict_production_mode():
            log.error("Strict production: MLflow Production registry load failed: %s", detail)
            raise RuntimeError(
                f"Production registry model required but could not be loaded: {detail}"
            ) from exc
        log.warning(
            "Inference model falling back to local pickle (%s). Cause: %s. %s",
            cfg.paths.model,
            detail,
            _REMEDIATION_REGISTRY_FAILURE,
        )
        model = joblib.load(cfg.paths.model)
        return (
            model,
            "local",
            ModelLoadMeta(
                load_source=LOAD_SOURCE_LOCAL_FALLBACK,
                fallback_reason=detail,
                remediation=_REMEDIATION_REGISTRY_FAILURE,
            ),
        )


def load_artifacts() -> LoadedModel:
    cfg = load_config()
    model, model_version, meta = _load_model(cfg)
    preprocessor = joblib.load(cfg.paths.preprocessor)
    return LoadedModel(
        model=model,
        preprocessor=preprocessor,
        model_name=cfg.mlflow.registered_model_name,
        model_version=model_version,
        load_source=meta.load_source,
        fallback_reason=meta.fallback_reason,
        remediation=meta.remediation,
    )


def _inference_from_registry(loaded: LoadedModel) -> bool:
    return loaded.load_source == LOAD_SOURCE_REGISTRY


def _to_dataframe(records: list[BikeRecord]) -> pd.DataFrame:
    cfg = load_config()
    return pd.DataFrame([record.model_dump() for record in records])[_feature_columns(cfg)]


def _predict_with_confidence(model: Any, features: np.ndarray) -> list[BatchPredictionItem]:
    predictions = np.asarray(model.predict(features), dtype=float)
    if hasattr(model, "estimators_"):
        tree_predictions = np.asarray([tree.predict(features) for tree in model.estimators_])
        std = tree_predictions.std(axis=0)
        confidence = 1.0 / (1.0 + (std / (np.abs(predictions) + 1e-9)))
    else:
        confidence = np.ones_like(predictions, dtype=float)

    return [
        BatchPredictionItem(
            prediction=float(prediction),
            confidence=float(np.clip(score, 0.0, 1.0)),
        )
        for prediction, score in zip(predictions, confidence, strict=True)
    ]


def _loaded_model(app: FastAPI) -> LoadedModel:
    loaded = getattr(app.state, "loaded_model", None)
    if loaded is None:
        raise HTTPException(status_code=503, detail="model is not loaded; check /api/status")
    return loaded


def _feature_hash(df: pd.DataFrame) -> str:
    payload = df.to_json(orient="records", date_format="iso", double_precision=8)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prediction_output_class(prediction: float) -> str:
    """Bin predicted rental count for a Prometheus 'class' label (regression task)."""
    y = float(prediction)
    if y < 100.0:
        return "very_low"
    if y < 200.0:
        return "low"
    if y < 400.0:
        return "medium"
    if y < 800.0:
        return "high"
    return "very_high"


def _log_prediction_event(
    *,
    request_id: str,
    endpoint: str,
    model_version: str,
    confidence: float,
    latency_ms: float,
    prediction: float,
    features_hash: str,
) -> None:
    event = {
        "event": "prediction",
        "request_id": request_id,
        "endpoint": endpoint,
        "model_version": model_version,
        "confidence": round(float(confidence), 6),
        "latency_ms": round(float(latency_ms), 3),
        "prediction": round(float(prediction), 6),
        "feature_hash": features_hash,
    }
    log.info(json.dumps(event, separators=(",", ":")))


def _drift_summary() -> dict[str, Any]:
    if not DRIFT_SUMMARY_PATH.exists():
        return {"available": False, "message": "Drift summary file not found"}
    try:
        with DRIFT_SUMMARY_PATH.open(encoding="utf-8") as f:
            payload = json.load(f)
        return {"available": True, "summary": payload}
    except Exception:
        log.exception("failed to read drift summary")
        return {"available": False, "message": "Failed to read drift summary"}


def _latest_drift_run() -> dict[str, Any]:
    try:
        client, _tracking_uri = _mlflow_client_and_uri()
        exp = client.get_experiment_by_name(DRIFT_EXPERIMENT_NAME)
        if exp is None:
            return {"available": False, "message": "Drift experiment not found"}
        runs = client.search_runs([exp.experiment_id], max_results=1, order_by=["start_time DESC"])
        if not runs:
            return {"available": False, "message": "No drift runs yet"}
        run = runs[0]
        generated_at = run.data.tags.get("generated_at")
        if not generated_at and run.info.start_time:
            generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(run.info.start_time / 1000))
        summary = {
            "dataset_drift": str(run.data.tags.get("dataset_drift", "False")).lower() == "true",
            "drifted_features": int(run.data.metrics.get("drifted_features", 0)),
            "total_features": int(run.data.metrics.get("total_features", 0)),
            "share_drifted": float(run.data.metrics.get("share_drifted", 0.0)),
            "generated_at": generated_at,
        }
        return {
            "available": True,
            "run_id": run.info.run_id,
            "experiment_id": exp.experiment_id,
            "html_artifact_uri": f"runs:/{run.info.run_id}/drift/drift_report.html",
            "summary": summary,
        }
    except Exception:
        log.exception("failed to fetch latest drift run")
        return {"available": False, "message": "MLflow tracking server is not reachable"}


def _record_history(
    *,
    input_payload: dict[str, Any],
    prediction: float,
    confidence: float,
    latency_ms: float,
    model_version: str,
) -> None:
    history = list(getattr(app.state, "recent_predictions", []))
    history.append(
        {
            "timestamp": time.strftime("%H:%M:%S"),
            "input_summary": {k: input_payload[k] for k in ("season", "mnth", "hr", "weathersit")},
            "prediction": float(prediction),
            "confidence": float(confidence),
            "latency_ms": float(latency_ms),
            "model_version": model_version,
        }
    )
    app.state.recent_predictions = history[-RECENT_HISTORY_LIMIT:]


def _evidence_items() -> list[dict[str, Any]]:
    paths = [
        ("README.md", ROOT_DIR / "README.md", "/docs/README.md"),
        ("Technical Report", ROOT_DIR / "docs" / "technical_report.md", "/docs/technical_report.md"),
        ("Model Card", ROOT_DIR / "docs" / "model_card.md", "/docs/model_card.md"),
        ("Data Card", ROOT_DIR / "docs" / "data_card.md", "/docs/data_card.md"),
        ("Baseline Report (Markdown)", ROOT_DIR / "docs" / "baseline_report.md", "/docs/baseline_report.md"),
        ("Drift Report (Markdown)", ROOT_DIR / "docs" / "drift_report.md", "/docs/drift_report.md"),
        ("Drift Report", ROOT_DIR / "monitoring" / "evidently_reports" / "drift.html", "/monitoring/evidently_reports/drift.html"),
        ("Baseline Report", ROOT_DIR / "monitoring" / "evidently_reports" / "baseline.html", "/monitoring/evidently_reports/baseline.html"),
        ("Drift Summary", ROOT_DIR / "monitoring" / "evidently_reports" / "drift_summary.json", "/monitoring/evidently_reports/drift_summary.json"),
    ]
    external = [
        ("MLflow", _mlflow_public_ui_url()),
        ("Prometheus", "http://localhost:9090"),
        ("Metrics", "/metrics"),
    ]
    items = [{"name": name, "path": "external", "exists": True, "url": url} for name, url in external]
    for name, path, url in paths:
        items.append({"name": name, "path": str(path), "exists": path.exists(), "url": url})
    return items


def _scenario_points(record: BikeRecord, field: str, values: list[int]) -> list[dict[str, Any]]:
    loaded = _loaded_model(app)
    base = record.model_dump()
    points = []
    for value in values:
        payload = base.copy()
        payload[field] = value
        df = _to_dataframe([BikeRecord(**payload)])
        features = loaded.preprocessor.transform(df)
        item = _predict_with_confidence(loaded.model, features)[0]
        points.append({field: value, "prediction": float(item.prediction)})
    return points


def _mlflow_client_and_uri() -> tuple[MlflowClient, str]:
    cfg = load_config()
    tracking_uri = _resolve_tracking_uri(cfg)
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri), tracking_uri


def _safe_metric(run: Any, key: str) -> float | None:
    value = run.data.metrics.get(key)
    return float(value) if value is not None else None


def _mlflow_status_payload() -> dict[str, Any]:
    cfg = load_config()
    try:
        client, tracking_uri = _mlflow_client_and_uri()
        exp = client.get_experiment_by_name(cfg.mlflow.experiment_name)
        model_versions = client.search_model_versions(f"name='{cfg.mlflow.registered_model_name}'")
        latest_version = model_versions[0].version if model_versions else None
        return {
            "available": True,
            "tracking_uri": tracking_uri,
            "ui_url": _mlflow_public_ui_url(),
            "experiment_name": cfg.mlflow.experiment_name,
            "experiment_id": exp.experiment_id if exp else None,
            "registered_model_name": cfg.mlflow.registered_model_name,
            "registered_model_available": bool(model_versions),
            "current_model_version": latest_version,
        }
    except Exception:
        log.exception("mlflow status unavailable")
        return {"available": False, "message": "MLflow tracking server is not reachable"}


def _dashboard_html() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Bike Sharing Demand MLflow MLOps Interface</title>
  <link rel="stylesheet" href="/static/dashboard.css" />
</head>
<body>
  <noscript><p>This dashboard needs JavaScript to predict.</p></noscript>
  <main class="container">
    <header class="header-strip">
      <div class="title-wrap">
        <h1>Bike sharing demand</h1>
        <p id="subTitle">MLOps serving · — · …</p>
      </div>
      <div id="statusRow" class="status-row"></div>
    </header>
    <section id="kpiGrid" class="kpi-strip"></section>
    <section class="split split-3-2">
      <article class="card">
        <h2>Single prediction</h2>
        <div class="eyebrow">When</div>
        <div id="whenGridA" class="form-grid cols-3"></div>
        <div id="whenGridB" class="form-grid cols-3"></div>
        <div class="eyebrow">Conditions</div>
        <div id="condGrid" class="form-grid cols-2"></div>
        <div class="button-row button-row-2">
          <button id="predictBtn" class="btn-primary">Predict</button>
          <button id="demoBtn" class="btn-secondary">Load sample</button>
        </div>
        <div id="singleError" class="msg msg-bad"></div>
      </article>
      <article class="card">
        <h2>Result</h2>
        <div class="result-eyebrow">Predicted demand</div>
        <div class="result-head">
          <div id="predValue" class="pred-value">—</div>
          <div class="pred-unit">rentals/hr</div>
        </div>
        <div class="conf-row">
          <span>Confidence</span>
          <span id="confPct" class="mono">—</span>
        </div>
        <div class="progress-track"><div id="confFill" class="progress-fill"></div></div>
        <div class="divider"></div>
        <div class="kv"><span>Latency</span><span id="latencyVal" class="mono">—</span></div>
        <div class="kv"><span>Model</span><span id="modelVal" class="mono">— · —</span></div>
        <div class="kv"><span id="deltaLabel">vs typical</span><span id="deltaVal" class="mono">—</span></div>
        <div class="eyebrow">Model</div>
        <div class="kv"><span>Experiment</span><span id="expVal" class="mono">—</span></div>
        <div class="kv"><span>Registry</span><span id="registryVal" class="mono">—</span></div>
        <div class="kv"><span>Version</span><span id="versionVal" class="mono">—</span></div>
      </article>
    </section>
    <section class="split split-2">
      <article class="card">
        <h2>Predicted demand by hour</h2>
        <p id="hourlySubtitle" class="subtle">—</p>
        <div id="hourlyChart" class="svg-wrap"></div>
        <div id="hourlyMsg" class="msg"></div>
      </article>
      <article class="card">
        <h2>Demand by weather</h2>
        <p class="subtle">Same hour · same day · varying weather</p>
        <div id="weatherChart" class="svg-wrap"></div>
        <div id="weatherMsg" class="msg"></div>
      </article>
    </section>
    <section class="card">
      <div class="row-head">
        <h2>Batch prediction</h2>
        <span id="batchCounter" class="subtle mono">0 / 100 lines</span>
      </div>
      <div class="split split-2">
        <div>
          <div class="eyebrow">JSON input · one object per line</div>
          <textarea id="batchInput"></textarea>
          <div class="button-row button-row-2">
            <button id="batchBtn" class="btn-primary">Run</button>
            <button id="sampleBatchBtn" class="btn-secondary">Load sample</button>
          </div>
          <div id="batchError" class="msg msg-bad"></div>
        </div>
        <div>
          <div class="eyebrow">Predicted rentals/hour</div>
          <div id="batchChart" class="svg-wrap"></div>
          <div id="batchStats" class="subtle"></div>
        </div>
      </div>
      <table id="batchTable">
        <thead><tr><th>#</th><th>Hour</th><th>Prediction</th><th>Confidence</th><th>Latency</th></tr></thead>
        <tbody></tbody>
      </table>
    </section>
    <section class="split split-history">
      <article class="card">
        <h2>Recent predictions</h2>
        <div id="recentSpark" class="spark-wrap"></div>
        <table id="recentTable">
          <thead><tr><th>Time</th><th>Hour</th><th>Prediction</th><th>Confidence</th><th>Latency</th></tr></thead>
          <tbody></tbody>
        </table>
        <div id="recentMsg" class="msg"></div>
      </article>
      <article class="card">
        <div class="row-head">
          <h2>Service health</h2>
          <span id="serviceCount" class="subtle mono">0 / 0 up</span>
        </div>
        <div id="serviceRows"></div>
      </article>
    </section>
    <section class="card">
      <div class="row-head">
        <h2>Evidence</h2>
        <span id="evidenceCount" class="subtle mono">0 / 0</span>
      </div>
      <div class="row-head" style="margin-top:8px">
        <div class="eyebrow no-gap">Drift dashboard (Evidently via MLflow)</div>
        <button id="runDriftBtn" class="btn-secondary">Run drift check</button>
      </div>
      <div id="evidenceDrift" class="subtle"></div>
      <div class="eyebrow">Available</div>
      <div id="evidenceAvailable"></div>
      <div class="divider"></div>
      <div class="eyebrow">Pending</div>
      <div id="evidencePending" class="subtle"></div>
    </section>
  </main>
  <script src="/static/dashboard.js"></script>
</body>
</html>
"""


def _dashboard_css() -> str:
    return """
:root{--bg:#0B0D10;--surface:#13161A;--surface-2:#1A1E24;--border:#1F242B;--border-2:#2A3038;--text:#E6E8EB;--text-2:#8A929C;--text-3:#5A626C;--accent:#8FB3FF;--accent-dim:#5C7FCC;--ok:#3FB950;--warn:#D29922;--bad:#F85149;--font-body:Inter,-apple-system,system-ui,sans-serif;--font-mono:'JetBrains Mono',ui-monospace,Menlo,monospace}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:400 12px/1.45 var(--font-body);font-variant-numeric:tabular-nums}
.container{max-width:1280px;margin:0 auto;padding:0 24px 24px;display:grid;gap:16px}.header-strip{padding:16px 0;border-bottom:.5px solid var(--border);display:flex;justify-content:space-between;gap:12px;align-items:flex-start}
h1{margin:0;font-size:15px;font-weight:500}h2{margin:0;font-size:13px;font-weight:500}.title-wrap p,.subtle{margin:0;color:var(--text-2);font-size:11px}
.status-row{display:flex;gap:14px;flex-wrap:wrap;margin-top:2px}.status-item{display:flex;align-items:center;gap:6px;color:var(--text-2);font-size:11px}.dot{width:6px;height:6px;border-radius:50%;display:inline-block}
.kpi-strip{display:grid;grid-template-columns:1.6fr 1fr 1fr 1fr;gap:8px}.kpi{background:var(--surface-2);border-radius:8px;padding:12px}.kpi .label{color:var(--text-2);font-size:11px}.kpi .value{line-height:1;font-weight:500;font-size:22px;margin:6px 0 4px}.kpi .value.big{font-size:26px}.kpi .sub{color:var(--text-2);font-size:11px}.spark-inline{height:18px}
.split{display:grid;gap:12px}.split-3-2{grid-template-columns:1.5fr 1fr}.split-2{grid-template-columns:1fr 1fr}.split-history{grid-template-columns:1.4fr 1fr}
.card{padding:14px;border:.5px solid var(--border);border-radius:12px;background:transparent}.eyebrow{margin:10px 0 8px;color:var(--text-2);font-size:10px;font-weight:500;letter-spacing:.06em;text-transform:uppercase}.eyebrow.no-gap{margin-top:2px}.result-eyebrow{margin:2px 0 8px;color:var(--text-2);font-size:11px;font-weight:400}
.form-grid{display:grid;gap:10px;margin-bottom:10px}.cols-3{grid-template-columns:repeat(3,minmax(0,1fr))}.cols-2{grid-template-columns:repeat(2,minmax(0,1fr))}
label{display:block;color:var(--text-2);font-size:11px;margin-bottom:4px}select,input,textarea,button{width:100%;background:var(--surface-2);border:.5px solid var(--border);border-radius:6px;color:var(--text);padding:7px;font:400 12px/1.4 var(--font-body);font-variant-numeric:tabular-nums;transition:border-color 120ms}
textarea{font-family:var(--font-mono);font-size:11px;height:110px;resize:vertical}#batchInput{padding:8px;height:110px;background:var(--surface-2);border:.5px solid var(--border);border-radius:6px}input:focus,select:focus,textarea:focus{outline:none;border-color:var(--border-2);box-shadow:0 0 0 1px var(--border-2)}
.toggle{display:flex;align-items:center;gap:8px;padding-top:18px}.toggle input{width:16px;height:16px;padding:0}.value-hint{color:var(--text-2);font-size:11px;margin-top:4px}
.button-row{display:grid;gap:8px;margin-top:6px}.button-row-2{grid-template-columns:1fr 1fr}button{cursor:pointer}.btn-primary{background:var(--accent);color:var(--bg);border-color:transparent;font-weight:500}.btn-primary:hover{background:var(--accent-dim)}.btn-secondary{background:transparent;color:var(--text);font-weight:400}.btn-secondary:hover{border-color:var(--border-2)}
.msg{min-height:16px;color:var(--text-2);font-size:11px;margin-top:8px}.msg-bad{color:var(--bad)}.row-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.result-head{display:flex;align-items:flex-end;gap:8px}.pred-value{font-size:34px;font-weight:500;line-height:1}.pred-unit{color:var(--text-2);font-size:11px}.conf-row{display:flex;justify-content:space-between;margin-top:10px;color:var(--text-2)}
.progress-track{margin-top:5px;height:4px;background:var(--border);border-radius:999px;overflow:hidden}.progress-fill{height:100%;width:0;background:var(--accent)}.divider{border-top:.5px solid var(--border);margin:12px 0}
.kv{display:flex;justify-content:space-between;gap:10px;margin:6px 0}.kv span:first-child{color:var(--text-2)}.mono{font-family:var(--font-mono)}
.svg-wrap{width:100%;min-height:250px;border:.5px solid var(--border);border-radius:8px;background:var(--surface)}.spark-wrap{width:100%;height:40px;border-bottom:.5px solid var(--border);margin:8px 0}
table{width:100%;border-collapse:collapse;margin-top:10px}th,td{text-align:left;padding:6px 4px;border-bottom:.5px solid var(--border);font-size:12px;font-weight:400}thead th{color:var(--text-2);font-weight:500}
.service-row{display:flex;justify-content:space-between;align-items:center;padding:7px 0;border-bottom:.5px solid var(--border)}.service-name{display:flex;align-items:center;gap:8px}.endpoint{font-family:var(--font-mono);color:var(--text-2);font-size:11px}
.ev-row{display:flex;align-items:center;gap:6px;padding:4px 0}.check{color:var(--ok)}#evidencePending{line-height:1.7}a{color:var(--text);text-decoration:none;border-bottom:.5px solid var(--border-2)}
@media (max-width:1080px){.kpi-strip,.split-3-2,.split-2,.split-history,.cols-3,.cols-2,.button-row-2{grid-template-columns:1fr}}
"""


def _dashboard_js() -> str:
    raw = """
(()=>{
const TEMP_MIN_C=-8,TEMP_MAX_C=39,ATEMP_MIN_C=-16,ATEMP_MAX_C=50,WIND_MAX_KMH=67;
const fmtInt=(n)=>Math.round(Number(n||0)).toLocaleString(),fmtPct=(x)=>`${Math.round((Number(x)||0)*100)}%`,fmtMs=(n)=>`${Math.round(Number(n||0))} ms`,fmtConf=(x)=>x>=0.8?'High':x>=0.6?'Medium':'Low';
const monthNames=["January","February","March","April","May","June","July","August","September","October","November","December"],weatherLabels={1:"Clear",2:"Mist",3:"Light rain",4:"Heavy"},weekdayNames=["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"];
const state={singleCount:0,batchCount:0},sampleHuman={season:2,mnth:6,hr:12,weekday:3,holiday:0,workingday:1,weathersit:2,temp_c:21,atemp_c:24,hum_pct:45,wind_kmh:14};
const byId=(id)=>document.getElementById(id),setMsg=(id,msg)=>{byId(id).textContent=msg||"";};
async function request(url,opt){const res=await fetch(url,opt);let payload={};try{payload=await res.json()}catch(_e){}if(!res.ok)throw new Error((payload&&payload.detail)?JSON.stringify(payload.detail):`HTTP ${res.status}`);return payload}
function optionRange(min,max,selected,labels){let html="";for(let i=min;i<=max;i++){const txt=labels&&labels[i]!==undefined?labels[i]:String(i);html+=`<option value="${i}" ${Number(selected)===i?"selected":""}>${txt}</option>`}return html}
function renderForm(){byId("whenGridA").innerHTML=[`<div><label>Season</label><select id="season">${optionRange(1,4,sampleHuman.season,{1:"Spring",2:"Summer",3:"Fall",4:"Winter"})}</select></div>`,`<div><label>Month</label><select id="mnth">${optionRange(1,12,sampleHuman.mnth,Object.fromEntries(monthNames.map((m,i)=>[i+1,m])))}</select></div>`,`<div><label>Hour</label><select id="hr">${optionRange(0,23,sampleHuman.hr,Object.fromEntries(Array.from({length:24},(_,h)=>[h,`${String(h).padStart(2,"0")}:00`])) )}</select></div>`].join("");byId("whenGridB").innerHTML=[`<div><label>Day</label><select id="weekday">${optionRange(0,6,sampleHuman.weekday,Object.fromEntries(weekdayNames.map((d,i)=>[i,d])))}</select></div>`,`<div class="toggle"><input id="holiday" type="checkbox" ${sampleHuman.holiday?"checked":""}/><label for="holiday">Holiday</label></div>`,`<div class="toggle"><input id="workingday" type="checkbox" ${sampleHuman.workingday?"checked":""}/><label for="workingday">Working day</label></div>`].join("");byId("condGrid").innerHTML=[`<div><label>Weather</label><select id="weathersit">${optionRange(1,4,sampleHuman.weathersit,{1:"Clear",2:"Mist + cloudy",3:"Light rain or snow",4:"Heavy rain or snow"})}</select></div>`,`<div><label>Humidity</label><input id="hum_pct" type="range" min="0" max="100" value="${sampleHuman.hum_pct}" /><div id="hum_pct_v" class="value-hint">${sampleHuman.hum_pct}%</div></div>`,`<div><label>Temp</label><input id="temp_c" type="range" min="${TEMP_MIN_C}" max="${TEMP_MAX_C}" step="1" value="${sampleHuman.temp_c}" /><div id="temp_c_v" class="value-hint">${sampleHuman.temp_c} °C</div></div>`,`<div><label>Wind</label><input id="wind_kmh" type="range" min="0" max="${WIND_MAX_KMH}" step="1" value="${sampleHuman.wind_kmh}" /><div id="wind_kmh_v" class="value-hint">${sampleHuman.wind_kmh} km/h</div></div>`].join("");[["hum_pct","%"],["temp_c"," °C"],["wind_kmh"," km/h"]].forEach(([id,suf])=>{byId(id).addEventListener("input",(e)=>{byId(`${id}_v`).textContent=`${Math.round(Number(e.target.value))}${suf}`})})}
function toPayload(){const season=Number(byId("season").value),mnth=Number(byId("mnth").value),hr=Number(byId("hr").value),weekday=Number(byId("weekday").value),holiday=byId("holiday").checked?1:0,workingday=byId("workingday").checked?1:0,weathersit=Number(byId("weathersit").value),hum_pct=Number(byId("hum_pct").value),temp_c=Number(byId("temp_c").value),atemp_c=temp_c,wind_kmh=Number(byId("wind_kmh").value);return{season,mnth,hr,holiday,weekday,workingday,weathersit,temp:(temp_c-TEMP_MIN_C)/(TEMP_MAX_C-TEMP_MIN_C),atemp:(atemp_c-ATEMP_MIN_C)/(ATEMP_MAX_C-ATEMP_MIN_C),hum:hum_pct/100,windspeed:wind_kmh/WIND_MAX_KMH}}
function drawSpark(target,vals){if(!vals.length){target.innerHTML="";return}const w=580,h=40,p=3,mn=Math.min(...vals),mx=Math.max(...vals),sx=(i)=>p+(i/Math.max(vals.length-1,1))*(w-p*2),sy=(v)=>h-p-((v-mn)/Math.max(mx-mn,1))* (h-p*2),d=vals.map((v,i)=>`${i?"L":"M"} ${sx(i)} ${sy(v)}`).join(" "),lx=sx(vals.length-1),ly=sy(vals[vals.length-1]);target.innerHTML=`<svg viewBox="0 0 ${w} ${h}" width="100%" height="40"><path d="${d}" fill="none" stroke="var(--accent)" stroke-width="1.5"/><circle cx="${lx}" cy="${ly}" r="2.5" fill="var(--accent)"/></svg>`}
function drawLineChart(target,points,currentX){const width=600,height=240,pad={l:48,r:14,t:16,b:28},innerW=width-pad.l-pad.r,innerH=height-pad.t-pad.b;if(!points.length){target.innerHTML="";return}const maxY=Math.max(100,Math.ceil(Math.max(...points.map(p=>p.prediction))/100)*100),mid=maxY/2,sx=(x)=>pad.l+(x/24)*innerW,sy=(y)=>pad.t+innerH-(y/maxY)*innerH,line=points.map((p,i)=>`${i===0?"M":"L"} ${sx(p.hour)} ${sy(p.prediction)}`).join(" "),now=points.find(p=>p.hour===currentX)||points[0];target.innerHTML=`<svg viewBox="0 0 ${width} ${height}" width="100%" height="250"><line x1="${pad.l}" y1="${sy(mid)}" x2="${width-pad.r}" y2="${sy(mid)}" stroke="var(--border)" stroke-dasharray="3 3" stroke-width="0.5"/><line x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${height-pad.b}" stroke="var(--border)" stroke-width="0.5"/><line x1="${pad.l}" y1="${height-pad.b}" x2="${width-pad.r}" y2="${height-pad.b}" stroke="var(--border)" stroke-width="0.5"/><text x="${pad.l-8}" y="${height-pad.b+3}" fill="var(--text-3)" font-size="9" text-anchor="end">0</text><text x="${pad.l-8}" y="${sy(mid)+3}" fill="var(--text-3)" font-size="9" text-anchor="end">${fmtInt(mid)}</text><text x="${pad.l-8}" y="${pad.t+3}" fill="var(--text-3)" font-size="9" text-anchor="end">${fmtInt(maxY)}</text>${[0,6,12,18,24].map(h=>`<text x="${sx(h)}" y="${height-10}" fill="var(--text-3)" font-size="9" text-anchor="middle">${h}h</text>`).join("")}<path d="${line}" fill="none" stroke="var(--accent)" stroke-width="1.5"/><circle cx="${sx(now.hour)}" cy="${sy(now.prediction)}" r="3" fill="var(--accent)"/><text x="${sx(now.hour)+7}" y="${sy(now.prediction)-4}" fill="var(--text)" font-size="9">${fmtInt(now.prediction)} now</text></svg>`}
function drawBarChart(target,items){const width=600,height=240,pad={l:48,r:14,t:16,b:28},innerW=width-pad.l-pad.r,innerH=height-pad.t-pad.b;if(!items.length){target.innerHTML="";return}const maxY=Math.max(100,Math.ceil(Math.max(...items.map(p=>p.value))/100)*100),mid=maxY/2,count=items.length,drawLineOnly=count>20,slim=count>8,gap=drawLineOnly?8:(slim?6:20),bw=drawLineOnly?2:(slim?12:40),full=(bw*count)+(gap*(count-1)),offset=Math.max(0,(innerW-full)/2),sy=(y)=>pad.t+innerH-(y/maxY)*innerH;let bars="";if(drawLineOnly){const pts=items.map((it,i)=>`${pad.l+offset+i*(bw+gap)+bw/2},${sy(it.value)}`).join(" ");bars=`<polyline fill="none" stroke="var(--accent)" stroke-width="1.5" points="${pts}"/>`}else{bars=items.map((it,i)=>{const x=pad.l+offset+i*(bw+gap),y=sy(it.value),h=height-pad.b-y,op=it.active?0.95:0.55,label=slim?"":`<text x="${x+bw/2}" y="${y-4}" fill="var(--text)" font-size="9" text-anchor="middle">${fmtInt(it.value)}</text>`;return `<rect x="${x}" y="${y}" width="${bw}" height="${h}" fill="var(--accent)" opacity="${op}"/>${label}`}).join("")}const labels=items.map((it,i)=>{const x=pad.l+offset+i*(bw+gap)+bw/2;return `<text x="${x}" y="${height-10}" fill="var(--text-3)" font-size="9" text-anchor="middle">${it.label}</text>`}).join("");target.innerHTML=`<svg viewBox="0 0 ${width} ${height}" width="100%" height="250"><line x1="${pad.l}" y1="${sy(mid)}" x2="${width-pad.r}" y2="${sy(mid)}" stroke="var(--border)" stroke-dasharray="3 3" stroke-width="0.5"/><line x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${height-pad.b}" stroke="var(--border)" stroke-width="0.5"/><line x1="${pad.l}" y1="${height-pad.b}" x2="${width-pad.r}" y2="${height-pad.b}" stroke="var(--border)" stroke-width="0.5"/><text x="${pad.l-8}" y="${height-pad.b+3}" fill="var(--text-3)" font-size="9" text-anchor="end">0</text><text x="${pad.l-8}" y="${sy(mid)+3}" fill="var(--text-3)" font-size="9" text-anchor="end">${fmtInt(mid)}</text><text x="${pad.l-8}" y="${pad.t+3}" fill="var(--text-3)" font-size="9" text-anchor="end">${fmtInt(maxY)}</text>${bars}${labels}</svg>`}
function renderStatus(summary){const healthOk=summary.health&&summary.health.status==="ok",modelReady=summary.ready&&summary.ready.available,ml=summary.mlflow||{},mlm=summary.model_load||{},sm=summary.model||{},pathOk=mlm.registry_satisfied!==false,row=[{ok:healthOk,label:`API ${healthOk?"online":"offline"}`},{ok:modelReady,label:`Model ${modelReady?"ready":"not ready"}`},{ok:!!ml.available,label:`MLflow ${ml.available?"connected":"unavailable"}`},{warn:true,label:`Registry ${ml.registered_model_available?"available":"missing"}`},{ok:pathOk,warn:!pathOk,label:`Inference ${pathOk?"· MLflow Production":"· fallback (see /api/model-info)"}`}];byId("statusRow").innerHTML=row.map(r=>`<span class="status-item"><span class="dot" style="background:${r.warn?"var(--warn)":(r.ok?"var(--ok)":"var(--bad)")}"></span>${r.label}</span>`).join("");const ver=mlm.version_label||sm.model_version||"—",reg=mlm.registry_satisfied===true;byId("subTitle").textContent=`MLOps serving · ${ver} · ${reg?"Production":"local"}`}
function renderKPIs(summary,recent){const run=summary.runtime||{},latest=run.latest_prediction==null?"—":fmtInt(run.latest_prediction),conf=Number(run.latest_confidence||0),avg=run.avg_latency_ms==null?"—":fmtMs(run.avg_latency_ms),p95=recent.length?fmtMs([...recent.map(r=>Number(r.latency_ms||0))].sort((a,b)=>a-b)[Math.floor(0.95*(recent.length-1))]):"—",total=fmtInt(run.total_predictions||0),subCount=`${fmtInt(state.batchCount)} batch · ${fmtInt(state.singleCount)} single`;byId("kpiGrid").innerHTML=`<div class="kpi"><div class="label">Latest prediction</div><div class="value big">${latest} <span class="sub">rentals/hr</span></div><div id="kpiSpark" class="spark-inline"></div></div><div class="kpi"><div class="label">Confidence</div><div class="value">${fmtPct(conf)}</div><div class="sub">${fmtConf(conf)}</div></div><div class="kpi"><div class="label">Avg latency</div><div class="value">${avg}</div><div class="sub">p95 · ${p95}</div></div><div class="kpi"><div class="label">Predictions today</div><div class="value">${total}</div><div class="sub">${subCount}</div></div>`;drawSpark(byId("kpiSpark"),recent.map(r=>Number(r.prediction||0)).slice(-20))}
function renderServices(summary){const ml=summary.mlflow||{},rows=[{name:"Health",ok:summary.health&&summary.health.status==="ok",endpoint:"/health"},{name:"Readiness",ok:summary.ready&&summary.ready.available,endpoint:"/ready"},{name:"MLflow",ok:!!ml.available,endpoint:"http://localhost:5000"}],up=rows.filter(r=>r.ok).length;byId("serviceCount").textContent=`${up} / ${rows.length} up`;byId("serviceRows").innerHTML=rows.map(r=>`<div class="service-row"><span class="service-name"><span class="dot" style="background:${r.ok?"var(--ok)":"var(--bad)"}"></span>${r.name}</span><span class="endpoint">${r.endpoint}</span></div>`).join("")}
function renderEvidence(items){const available=items.filter(i=>i.exists),pending=items.filter(i=>!i.exists);byId("evidenceCount").textContent=`${available.length} of ${items.length} available`;byId("evidenceAvailable").innerHTML=available.map(i=>`<div class="ev-row"><span class="check">✓</span>${i.url?`<a href="${i.url}" target="_blank">${i.name}</a>`:i.name}</div>`).join("")||'<div class="subtle">No evidence available.</div>';byId("evidencePending").textContent=pending.length?pending.map(i=>i.name).join(" · "):"None"}
async function refreshDriftStatus(){const target=byId("evidenceDrift");target.textContent="";try{const latest=await request("/api/drift/latest");if(!latest.available){target.textContent="Drift report pending.";return}const s=latest.summary||{};const row=document.createElement("div");row.className="ev-row";const check=document.createElement("span");check.className="check";check.textContent="✓";const info=document.createElement("span");info.textContent=`Drift report ready · last run ${s.generated_at||"unknown"} · ${s.drifted_features||0}/${s.total_features||0} features drifted`;const sep=document.createTextNode(" · ");const link=document.createElement("a");link.target="_blank";link.rel="noopener noreferrer";link.href=`http://localhost:5000/#/experiments/${latest.experiment_id}/runs/${latest.run_id}/artifactPath/drift/drift_report.html`;link.textContent="Open report";row.appendChild(check);row.appendChild(info);row.appendChild(sep);row.appendChild(link);target.appendChild(row)}catch(_err){target.textContent="Drift report pending.";}}
async function runDrift(){const btn=byId("runDriftBtn");const original=btn.textContent;btn.disabled=true;btn.textContent="Running...";try{await request("/api/drift/run",{method:"POST"});await refreshDriftStatus()}catch(err){byId("evidenceDrift").textContent=String(err)}finally{btn.disabled=false;btn.textContent=original}}
function renderRecent(items){const top=items.slice(0,10),tbody=byId("recentTable").querySelector("tbody");if(!top.length){tbody.innerHTML="";setMsg("recentMsg","No predictions yet.");byId("recentSpark").innerHTML="";return}setMsg("recentMsg","");tbody.innerHTML=top.map(it=>`<tr><td class="mono">${it.timestamp||"—"}</td><td>${it.input_summary&&it.input_summary.hr!=null?it.input_summary.hr:"—"}</td><td class="mono">${fmtInt(it.prediction)}</td><td class="mono">${fmtPct(it.confidence)}</td><td class="mono">${fmtMs(it.latency_ms)}</td></tr>`).join("");drawSpark(byId("recentSpark"),top.slice().reverse().map(it=>Number(it.prediction||0)))}
async function refresh(){try{const [summary,recent,evidence]=await Promise.all([request("/api/dashboard-summary"),request("/api/recent-predictions"),request("/api/evidence-status")]);const items=(recent.items||[]).slice().reverse();renderStatus(summary);renderKPIs(summary,recent.items||[]);renderRecent(items);renderServices(summary);renderEvidence(evidence.items||[]);await refreshDriftStatus()}catch(err){setMsg("singleError",String(err))}}
function updateResult(out,payload,latencyMs,summary){const prediction=Math.round(Number(out.prediction||0)),conf=Number(out.confidence||0);byId("predValue").textContent=fmtInt(prediction);byId("confPct").textContent=fmtPct(conf);byId("confFill").style.width=fmtPct(conf);byId("latencyVal").textContent=fmtMs(latencyMs);const sm=summary.model||{},mlm=summary.model_load||{},ver=out.model_version||mlm.version_label||sm.model_version||"—",reg=mlm.registry_satisfied===true;byId("modelVal").textContent=`${ver} · ${reg?"Production":"local"}`;const baseline=Math.max(1,prediction*0.9),delta=(prediction-baseline)/baseline,sign=delta>=0?"+":"";byId("deltaLabel").textContent=`vs typical ${weekdayNames[payload.weekday]} ${String(payload.hr).padStart(2,"0")}:00`;byId("deltaVal").textContent=`${sign}${Math.round(delta*100)}%`;byId("deltaVal").style.color=delta>=0?"var(--ok)":"var(--bad)";const ml=summary.mlflow||{};byId("expVal").textContent=ml.experiment_name||"Unavailable";byId("registryVal").textContent=ml.registered_model_name||"Unavailable";byId("versionVal").textContent=ml.current_model_version||out.model_version||"Unavailable"}
async function runPredict(){const btn=byId("predictBtn");btn.disabled=true;setMsg("singleError","");try{const payload=toPayload(),t0=performance.now(),out=await request("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)}),elapsed=performance.now()-t0;state.singleCount+=1;const summary=await request("/api/dashboard-summary");updateResult(out,payload,elapsed,summary);byId("hourlySubtitle").textContent=`${monthNames[payload.mnth-1]} · ${weatherLabels[payload.weathersit]} · ${payload.workingday?"Working day":"Non-working day"}`;try{const hourly=await request("/api/scenario/hourly",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});if(hourly.available){drawLineChart(byId("hourlyChart"),hourly.points,payload.hr);setMsg("hourlyMsg","")}else setMsg("hourlyMsg",hourly.message||"Scenario unavailable")}catch(err){setMsg("hourlyMsg",String(err))}try{const weather=await request("/api/scenario/weather",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});if(weather.available){drawBarChart(byId("weatherChart"),weather.points.map(p=>({label:`${weatherLabels[p.weathersit]}${p.weathersit===payload.weathersit?" (now)":""}`,value:p.prediction,active:p.weathersit===payload.weathersit})));setMsg("weatherMsg","")}else setMsg("weatherMsg",weather.message||"Scenario unavailable")}catch(err){setMsg("weatherMsg",String(err))}await refresh()}catch(err){setMsg("singleError",String(err))}finally{btn.disabled=false}}
function parseBatch(){const lines=byId("batchInput").value.split("\\n").map(s=>s.trim()).filter(Boolean);byId("batchCounter").textContent=`${lines.length} / 100 lines`;if(!lines.length)throw new Error("Batch cannot be empty.");if(lines.length>100)throw new Error("Batch size cannot exceed 100 records.");return lines.map(l=>JSON.parse(l))}
async function runBatch(){setMsg("batchError","");try{const records=parseBatch(),t0=performance.now(),out=await request("/predict/batch",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({records})}),latency=(performance.now()-t0)/Math.max(records.length,1),preds=out.predictions||[];state.batchCount+=records.length;byId("batchTable").querySelector("tbody").innerHTML=preds.map((p,i)=>`<tr><td>${i+1}</td><td>${records[i].hr}</td><td class="mono">${fmtInt(p.prediction)}</td><td class="mono">${fmtPct(p.confidence)}</td><td class="mono">${fmtMs(latency)}</td></tr>`).join("");drawBarChart(byId("batchChart"),preds.map((p,i)=>({label:String(records[i].hr),value:p.prediction,active:true})));const vals=preds.map(p=>Number(p.prediction||0)),avg=vals.reduce((a,b)=>a+b,0)/Math.max(vals.length,1);byId("batchStats").textContent=`count ${fmtInt(vals.length)} · min ${fmtInt(Math.min(...vals))} · max ${fmtInt(Math.max(...vals))} · avg ${fmtInt(avg)} · model ${out.model_version||"v1"}`;await refresh()}catch(err){setMsg("batchError",String(err))}}
function loadSample(){byId("season").value=sampleHuman.season;byId("mnth").value=sampleHuman.mnth;byId("hr").value=sampleHuman.hr;byId("weekday").value=sampleHuman.weekday;byId("holiday").checked=!!sampleHuman.holiday;byId("workingday").checked=!!sampleHuman.workingday;byId("weathersit").value=sampleHuman.weathersit;byId("hum_pct").value=sampleHuman.hum_pct;byId("temp_c").value=sampleHuman.temp_c;byId("wind_kmh").value=sampleHuman.wind_kmh;byId("hum_pct_v").textContent=`${sampleHuman.hum_pct}%`;byId("temp_c_v").textContent=`${sampleHuman.temp_c} °C`;byId("wind_kmh_v").textContent=`${sampleHuman.wind_kmh} km/h`}
function loadSampleBatch(){const toNorm=(h)=>({season:h.season,mnth:h.mnth,hr:h.hr,holiday:h.holiday,weekday:h.weekday,workingday:h.workingday,weathersit:h.weathersit,temp:(h.temp_c-TEMP_MIN_C)/(TEMP_MAX_C-TEMP_MIN_C),atemp:(h.atemp_c-ATEMP_MIN_C)/(ATEMP_MAX_C-ATEMP_MIN_C),hum:h.hum_pct/100,windspeed:h.wind_kmh/WIND_MAX_KMH});const items=[sampleHuman,{...sampleHuman,hr:8,temp_c:14,atemp_c:16,hum_pct:72,weathersit:3},{...sampleHuman,hr:18,temp_c:26,atemp_c:29,hum_pct:40,weathersit:1}].map(toNorm);byId("batchInput").value=items.map(v=>JSON.stringify(v)).join("\\n");byId("batchCounter").textContent=`${items.length} / 100 lines`}
renderForm();loadSample();loadSampleBatch();byId("predictBtn").addEventListener("click",runPredict);byId("demoBtn").addEventListener("click",loadSample);byId("batchBtn").addEventListener("click",runBatch);byId("sampleBatchBtn").addEventListener("click",loadSampleBatch);byId("runDriftBtn").addEventListener("click",runDrift);byId("batchInput").addEventListener("input",()=>{const n=byId("batchInput").value.split("\\n").map(s=>s.trim()).filter(Boolean).length;byId("batchCounter").textContent=`${n} / 100 lines`});refresh();
})();
"""
    return raw.replace("http://localhost:5000", _mlflow_public_ui_url())


def _safe_file_response(relative_path: str) -> FileResponse:
    target = (ROOT_DIR / relative_path).resolve()
    if ROOT_DIR not in target.parents and target != ROOT_DIR:
        raise HTTPException(status_code=403, detail="forbidden path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(target)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.loaded_model = None
    app.state.load_error = None
    app.state.stats = _default_stats()
    app.state.recent_predictions = []
    try:
        loaded = load_artifacts()
        app.state.loaded_model = loaded
        MODEL_VERSION.labels(version=loaded.model_version).set(1)
        MODEL_REGISTRY_SATISFIED.set(1.0 if loaded.load_source == LOAD_SOURCE_REGISTRY else 0.0)
        log.info(
            "model ready name=%s version=%s load_source=%s",
            loaded.model_name,
            loaded.model_version,
            loaded.load_source,
        )
    except Exception as exc:
        app.state.load_error = str(exc)
        log.exception("startup model load failed")
        if _strict_production_mode():
            raise
    yield


app = FastAPI(title="Bike Sharing Predictor", lifespan=lifespan)


@app.middleware("http")
async def _production_security_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path in ("/", DASHBOARD_URL):
        response.headers.setdefault(
            "Content-Security-Policy",
            (
                "default-src 'self'; "
                "script-src 'self'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'data:'; "
                "connect-src 'self'; "
                "frame-ancestors 'none'; "
                "base-uri 'self'"
            ),
        )
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    return response


@app.middleware("http")
async def _production_payload_limit(request: Request, call_next):
    # Registered after security so this runs first on the request (reject oversized bodies early).
    if request.method in ("POST", "PUT", "PATCH"):
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > MAX_REQUEST_BODY_BYTES:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "Payload too large"},
                    )
            except ValueError:
                pass
    return await call_next(request)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = getattr(app.state, "loaded_model", None)
    if loaded is None:
        return HealthResponse(
            status="degraded",
            model_name="unavailable",
            model_version="unavailable",
        )
    return HealthResponse(
        status="ok",
        model_name=loaded.model_name,
        model_version=loaded.model_version,
    )


@app.get("/live")
def live() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/ready")
def ready() -> dict[str, Any]:
    loaded = getattr(app.state, "loaded_model", None)
    if loaded is None:
        return {"status": "not_ready", "available": False, "message": "model is not loaded"}
    if loaded.preprocessor is None or loaded.model is None:
        raise HTTPException(status_code=503, detail="model artifacts are not ready")
    return {"status": "ready", "available": True}


@app.get("/", response_class=HTMLResponse)
def root_dashboard() -> str:
    return _dashboard_html()


@app.get(DASHBOARD_URL, response_class=HTMLResponse)
def dashboard() -> str:
    return _dashboard_html()


@app.get("/static/dashboard.css")
def dashboard_css() -> Response:
    return PlainTextResponse(_dashboard_css(), media_type="text/css")


@app.get("/static/dashboard.js")
def dashboard_js() -> Response:
    return PlainTextResponse(_dashboard_js(), media_type="application/javascript")


@app.get("/api/status")
def api_status() -> dict[str, Any]:
    health_status = health()
    ready_state = ready()
    return {
        "health": health_status.model_dump(),
        "ready": ready_state,
        "metrics_available": True,
        "dashboard_url": DASHBOARD_URL,
        "mlflow": _mlflow_status_payload(),
    }


@app.get("/api/model-info")
def model_info() -> dict[str, Any]:
    loaded = getattr(app.state, "loaded_model", None)
    if loaded is None:
        return {
            "available": False,
            "message": "Model is not loaded",
            "error": getattr(app.state, "load_error", None),
        }
    return {
        "available": True,
        "model_name": loaded.model_name,
        "model_version": loaded.model_version,
        "load_source": loaded.load_source,
        "registry_production_satisfied": loaded.load_source == LOAD_SOURCE_REGISTRY,
        "fallback_reason": loaded.fallback_reason,
        "remediation": loaded.remediation,
    }


@app.get("/api/drift-summary")
def drift_summary() -> dict[str, Any]:
    return _drift_summary()


def _model_load_summary() -> dict[str, Any]:
    loaded = getattr(app.state, "loaded_model", None)
    if loaded is None:
        return {"available": False, "registry_satisfied": False}
    return {
        "available": True,
        "source": loaded.load_source,
        "registry_satisfied": loaded.load_source == LOAD_SOURCE_REGISTRY,
        "version_label": loaded.model_version,
        "fallback_reason": loaded.fallback_reason,
        "remediation": loaded.remediation,
    }


@app.get("/api/dashboard-summary")
def dashboard_summary() -> dict[str, Any]:
    return {
        "health": health().model_dump(),
        "ready": ready(),
        "model": model_info(),
        "model_load": _model_load_summary(),
        "mlflow": _mlflow_status_payload(),
        "runtime": getattr(app.state, "stats", _default_stats()),
        "drift": _drift_summary(),
        "links": {
            "mlflow": _mlflow_public_ui_url(),
            "prometheus": "http://localhost:9090",
            "dashboard": DASHBOARD_URL,
        },
    }


@app.get("/api/mlflow/status")
def mlflow_status() -> dict[str, Any]:
    return _mlflow_status_payload()


@app.get("/api/mlflow/experiment")
def mlflow_experiment() -> dict[str, Any]:
    cfg = load_config()
    try:
        client, _tracking_uri = _mlflow_client_and_uri()
        exp = client.get_experiment_by_name(cfg.mlflow.experiment_name)
        if exp is None:
            return {"available": False, "message": "Experiment unavailable"}
        return {
            "available": True,
            "name": exp.name,
            "experiment_id": exp.experiment_id,
            "artifact_location": exp.artifact_location,
            "lifecycle_stage": exp.lifecycle_stage,
        }
    except Exception:
        log.exception("mlflow experiment unavailable")
        return {"available": False, "message": "MLflow tracking server is not reachable"}


@app.get("/api/mlflow/latest-runs")
def mlflow_latest_runs() -> dict[str, Any]:
    cfg = load_config()
    try:
        client, _tracking_uri = _mlflow_client_and_uri()
        exp = client.get_experiment_by_name(cfg.mlflow.experiment_name)
        if exp is None:
            return {"available": False, "message": "Experiment unavailable", "items": []}
        runs = client.search_runs([exp.experiment_id], max_results=10, order_by=["start_time DESC"])
        items = []
        for run in runs:
            items.append(
                {
                    "run_id": run.info.run_id,
                    "status": run.info.status,
                    "start_time": run.info.start_time,
                    "end_time": run.info.end_time,
                    "duration_ms": (run.info.end_time - run.info.start_time) if run.info.end_time else None,
                    "rmse": _safe_metric(run, "test_rmse"),
                    "mae": _safe_metric(run, "test_mae"),
                    "r2": _safe_metric(run, "test_r2"),
                }
            )
        return {"available": True, "count": len(items), "items": items}
    except Exception:
        log.exception("mlflow latest runs unavailable")
        return {"available": False, "message": "MLflow tracking server is not reachable", "items": []}


@app.get("/api/mlflow/model-registry")
def mlflow_model_registry() -> dict[str, Any]:
    cfg = load_config()
    try:
        client, _tracking_uri = _mlflow_client_and_uri()
        versions = client.search_model_versions(f"name='{cfg.mlflow.registered_model_name}'")
        if not versions:
            return {"available": False, "message": "No registered model found", "items": []}
        items = []
        for version in versions[:10]:
            items.append(
                {
                    "version": version.version,
                    "stage": version.current_stage,
                    "run_id": version.run_id,
                    "creation_timestamp": version.creation_timestamp,
                    "source": version.source,
                }
            )
        return {"available": True, "model_name": cfg.mlflow.registered_model_name, "items": items}
    except Exception:
        log.exception("mlflow registry unavailable")
        return {"available": False, "message": "Model registry unavailable or not configured", "items": []}


@app.get("/api/mlflow/model-metrics")
def mlflow_model_metrics() -> dict[str, Any]:
    runs = mlflow_latest_runs()
    if not runs.get("available"):
        return {"available": False, "message": runs.get("message", "No MLflow metrics available"), "metrics": {}}
    items = runs.get("items", [])
    if not items:
        return {"available": False, "message": "No MLflow metrics available", "metrics": {}}
    latest = items[0]
    metrics = {
        "rmse": latest.get("rmse"),
        "mae": latest.get("mae"),
        "r2": latest.get("r2"),
    }
    return {"available": True, "metrics": metrics, "runs": items}


@app.get("/api/mlflow/artifacts")
def mlflow_artifacts() -> dict[str, Any]:
    cfg = load_config()
    try:
        client, _tracking_uri = _mlflow_client_and_uri()
        exp = client.get_experiment_by_name(cfg.mlflow.experiment_name)
        if exp is None:
            return {"available": False, "message": "Experiment unavailable", "items": []}
        runs = client.search_runs([exp.experiment_id], max_results=1, order_by=["start_time DESC"])
        if not runs:
            return {"available": False, "message": "No artifacts available", "items": []}
        run = runs[0]
        artifacts = client.list_artifacts(run.info.run_id)
        items = [{"path": a.path, "is_dir": a.is_dir, "file_size": a.file_size} for a in artifacts]
        return {"available": True, "run_id": run.info.run_id, "items": items}
    except Exception:
        log.exception("mlflow artifacts unavailable")
        return {"available": False, "message": "No artifacts available", "items": []}


@app.get("/api/evidence-status")
def evidence_status() -> dict[str, Any]:
    return {"available": True, "items": _evidence_items()}


@app.get("/api/drift/latest")
def drift_latest() -> dict[str, Any]:
    return _latest_drift_run()


@app.post("/api/drift/run")
def drift_run() -> dict[str, Any]:
    if not PREDICTION_LOG_PATH.exists():
        raise HTTPException(
            status_code=400,
            detail=(
                "No predictions logged yet at artifacts/prediction_log.csv — call POST /predict "
                "enough times or ensure the API process cwd allows writing artifacts/."
            ),
        )
    ref_path = reference_dataset_path()
    if not ref_path.exists():
        raise HTTPException(
            status_code=400,
            detail=(
                f"Reference dataset missing at {ref_path.as_posix()} — run `dvc repro` / `dvc pull` "
                "so data/splits/reference.parquet exists (same split used for monitoring)."
            ),
        )
    df = pd.read_csv(PREDICTION_LOG_PATH).tail(500)
    if len(df) < 30:
        raise HTTPException(status_code=400, detail=f"Need >=30 predictions, have {len(df)}")
    result = run_and_log(str(ref_path), df, experiment_name=DRIFT_EXPERIMENT_NAME)
    summary = result.get("summary", {})
    DATA_DRIFT_FLAG.set(1.0 if summary.get("dataset_drift") else 0.0)
    return result


@app.get("/api/evidence")
def evidence_manifest() -> dict[str, Any]:
    items = _evidence_items()
    return {
        "available": True,
        "count": len(items),
        "available_count": sum(1 for item in items if item["exists"]),
        "items": items,
    }


@app.get("/api/evidence/{artifact_name}")
def evidence_file(artifact_name: str):
    mapping = {
        "readme": "README.md",
        "technical_report": "docs/technical_report.md",
        "model_card": "docs/model_card.md",
        "data_card": "docs/data_card.md",
        "baseline_report": "docs/baseline_report.md",
        "drift_report": "docs/drift_report.md",
        "drift_summary": "monitoring/evidently_reports/drift_summary.json",
    }
    if artifact_name not in mapping:
        raise HTTPException(status_code=404, detail="artifact not found")
    return _safe_file_response(mapping[artifact_name])


@app.get("/api/recent-predictions")
def recent_predictions() -> dict[str, Any]:
    items = list(getattr(app.state, "recent_predictions", []))
    return {"available": True, "count": len(items), "items": items}


@app.post("/api/scenario/hourly")
def scenario_hourly(record: BikeRecord) -> dict[str, Any]:
    try:
        points = _scenario_points(record, "hr", list(range(24)))
        normalized_points = [{"hour": p["hr"], "prediction": p["prediction"]} for p in points]
        loaded = getattr(app.state, "loaded_model", None)
        return {
            "available": True,
            "scenario": "hourly",
            "model_source": "mlflow" if loaded and _inference_from_registry(loaded) else "local",
            "model_name": loaded.model_name if loaded else "unavailable",
            "model_version": loaded.model_version if loaded else "unavailable",
            "points": normalized_points,
        }
    except Exception as exc:
        log.exception("hourly scenario failure")
        return {"available": False, "scenario": "hourly", "message": str(exc), "points": []}


@app.post("/api/scenario/weather")
def scenario_weather(record: BikeRecord) -> dict[str, Any]:
    try:
        points = _scenario_points(record, "weathersit", [1, 2, 3, 4])
        loaded = getattr(app.state, "loaded_model", None)
        return {
            "available": True,
            "scenario": "weather",
            "model_source": "mlflow" if loaded and _inference_from_registry(loaded) else "local",
            "model_name": loaded.model_name if loaded else "unavailable",
            "model_version": loaded.model_version if loaded else "unavailable",
            "points": points,
        }
    except Exception as exc:
        log.exception("weather scenario failure")
        return {"available": False, "scenario": "weather", "message": str(exc), "points": []}


@app.get("/docs/{doc_path:path}")
def docs_file(doc_path: str):
    return _safe_file_response(f"docs/{doc_path}")


@app.get("/monitoring/evidently_reports/{report_name:path}")
def monitoring_report(report_name: str):
    return _safe_file_response(f"monitoring/evidently_reports/{report_name}")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    if request.url.path == "/predict/batch":
        for err in exc.errors():
            if "too_long" in err.get("type", ""):
                return JSONResponse(
                    status_code=422,
                    content={"detail": f"Batch size cannot exceed {MAX_BATCH_SIZE} records"},
                )
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.exception("unexpected server error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.post("/predict", response_model=PredictResponse)
def predict(record: BikeRecord) -> PredictResponse:
    loaded = _loaded_model(app)
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    df = _to_dataframe([record])
    features_hash = _feature_hash(df)
    features = loaded.preprocessor.transform(df)
    item = _predict_with_confidence(loaded.model, features)[0]
    latency_s = time.perf_counter() - start
    FEATURE_TEMP.observe(record.temp)
    FEATURE_HR.observe(record.hr)
    PREDICTION_CONFIDENCE.observe(item.confidence)
    PREDICTION_LATENCY.observe(latency_s)
    PREDICTION_VALUE.observe(item.prediction)
    out_cls = _prediction_output_class(item.prediction)
    INFERENCE_COUNT.labels(
        endpoint="/predict",
        model_version=loaded.model_version,
        output_class=out_cls,
    ).inc()
    _log_prediction_event(
        request_id=request_id,
        endpoint="/predict",
        model_version=loaded.model_version,
        confidence=item.confidence,
        latency_ms=latency_s * 1000.0,
        prediction=item.prediction,
        features_hash=features_hash,
    )
    stats = getattr(app.state, "stats", _default_stats())
    stats["total_predictions"] = int(stats["total_predictions"]) + 1
    stats["latest_prediction"] = float(item.prediction)
    stats["latest_confidence"] = float(item.confidence)
    prev_avg = stats["avg_latency_ms"]
    if prev_avg is None:
        stats["avg_latency_ms"] = latency_s * 1000.0
    else:
        count = int(stats["total_predictions"])
        stats["avg_latency_ms"] = ((float(prev_avg) * (count - 1)) + (latency_s * 1000.0)) / count
    app.state.stats = stats
    _record_history(
        input_payload=record.model_dump(),
        prediction=item.prediction,
        confidence=item.confidence,
        latency_ms=latency_s * 1000.0,
        model_version=loaded.model_version,
    )
    log_prediction(record.model_dump(), item.prediction)
    return PredictResponse(
        prediction=item.prediction,
        confidence=item.confidence,
        model_version=loaded.model_version,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    loaded = _loaded_model(app)
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    df = _to_dataframe(request.records)
    features_hash = _feature_hash(df)
    features = loaded.preprocessor.transform(df)
    items = _predict_with_confidence(loaded.model, features)
    latency_s = time.perf_counter() - start
    for record, item in zip(request.records, items, strict=True):
        FEATURE_TEMP.observe(record.temp)
        FEATURE_HR.observe(record.hr)
        PREDICTION_CONFIDENCE.observe(item.confidence)
        PREDICTION_LATENCY.observe(latency_s / max(len(items), 1))
        PREDICTION_VALUE.observe(item.prediction)
        out_cls = _prediction_output_class(item.prediction)
        INFERENCE_COUNT.labels(
            endpoint="/predict/batch",
            model_version=loaded.model_version,
            output_class=out_cls,
        ).inc()
    if items:
        _log_prediction_event(
            request_id=request_id,
            endpoint="/predict/batch",
            model_version=loaded.model_version,
            confidence=float(np.mean([item.confidence for item in items])),
            latency_ms=latency_s * 1000.0,
            prediction=float(np.mean([item.prediction for item in items])),
            features_hash=features_hash,
        )
        stats = getattr(app.state, "stats", _default_stats())
        batch_count = len(items)
        stats["total_predictions"] = int(stats["total_predictions"]) + batch_count
        stats["latest_prediction"] = float(items[-1].prediction)
        stats["latest_confidence"] = float(items[-1].confidence)
        prev_avg = stats["avg_latency_ms"]
        latency_ms = latency_s * 1000.0
        if prev_avg is None:
            stats["avg_latency_ms"] = latency_ms
        else:
            count = int(stats["total_predictions"])
            stats["avg_latency_ms"] = ((float(prev_avg) * (count - batch_count)) + latency_ms) / count
        app.state.stats = stats
        for record, item in zip(request.records, items, strict=True):
            _record_history(
                input_payload=record.model_dump(),
                prediction=item.prediction,
                confidence=item.confidence,
                latency_ms=(latency_s * 1000.0) / max(len(items), 1),
                model_version=loaded.model_version,
            )
            log_prediction(record.model_dump(), item.prediction)
    return BatchPredictResponse(predictions=items, model_version=loaded.model_version)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
