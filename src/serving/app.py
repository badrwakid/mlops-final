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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from mlflow.exceptions import MlflowException
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.config import Config, load_config
from src.serving.metrics import (
    FEATURE_HR,
    FEATURE_TEMP,
    INFERENCE_COUNT,
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
DASHBOARD_URL = "/dashboard"
MAX_BATCH_SIZE = 100


def _default_stats() -> dict[str, float | int | None]:
    return {
        "total_predictions": 0,
        "latest_prediction": None,
        "latest_confidence": None,
        "avg_latency_ms": None,
    }


@dataclass
class LoadedModel:
    model: Any
    preprocessor: Any
    model_name: str
    model_version: str


def _resolve_tracking_uri(cfg: Config) -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or cfg.mlflow.tracking_uri


def _feature_columns(cfg: Config) -> list[str]:
    return cfg.data.numeric_features + cfg.data.categorical_features


def _load_model(cfg: Config) -> tuple[Any, str]:
    model_name = cfg.mlflow.registered_model_name
    model_uri = f"models:/{model_name}/Production"
    mlflow.set_tracking_uri(_resolve_tracking_uri(cfg))
    try:
        return mlflow.sklearn.load_model(model_uri), "Production"
    except MlflowException as exc:
        log.warning("MLflow registry load failed; falling back to local pickle: %s", exc)
        return joblib.load(cfg.paths.model), "local"


def load_artifacts() -> LoadedModel:
    cfg = load_config()
    model, model_version = _load_model(cfg)
    preprocessor = joblib.load(cfg.paths.preprocessor)
    return LoadedModel(
        model=model,
        preprocessor=preprocessor,
        model_name=cfg.mlflow.registered_model_name,
        model_version=model_version,
    )


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


def _dashboard_html() -> str:
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Bike Sharing Demand MLOps Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }
    .wrap { max-width: 1200px; margin: 0 auto; padding: 24px; }
    h1, h2 { margin: 0 0 12px; }
    .grid { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }
    .card { background: #111827; border: 1px solid #334155; border-radius: 10px; padding: 14px; }
    label { display:block; font-size: 12px; margin-bottom: 4px; color:#93c5fd; }
    input, select, button, textarea { width:100%; padding:8px; border-radius: 6px; border:1px solid #475569; background:#0b1220; color:#e2e8f0; }
    button { cursor:pointer; background:#1d4ed8; border:none; margin-top:6px; }
    button.secondary { background: #334155; }
    .row { display:grid; gap:10px; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
    table { width:100%; border-collapse: collapse; margin-top:8px; }
    td,th { border:1px solid #334155; padding:6px; font-size:12px; }
    .small { font-size: 12px; color: #94a3b8; }
    .ok { color:#22c55e; } .bad { color:#ef4444; }
    a { color: #93c5fd; }
    pre { white-space: pre-wrap; word-break: break-word; font-size: 12px; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Bike Sharing Demand MLOps Dashboard</h1>
    <p class="small">Professor-proof serving interface for predictions, status, and evidence.</p>

    <h2>System Status</h2>
    <div id="statusCards" class="grid"></div>

    <h2>Single Prediction</h2>
    <div class="card">
      <div id="singleForm" class="row"></div>
      <div class="row">
        <button id="predictBtn">Predict</button>
        <button class="secondary" id="demoBtn">Load Safe Demo Example</button>
        <button class="secondary" id="resetBtn">Reset</button>
      </div>
      <pre id="singleResult"></pre>
    </div>

    <h2>Batch Prediction</h2>
    <div class="card">
      <p class="small">One JSON record per line (max 100). You can paste multiple lines.</p>
      <textarea id="batchInput" rows="7"></textarea>
      <button id="batchBtn">Run Batch Prediction</button>
      <pre id="batchResult"></pre>
    </div>

    <h2>Project Evidence Links</h2>
    <div class="card">
      <a href="/api/dashboard-summary" target="_blank">Dashboard Summary JSON</a><br/>
      <a href="http://localhost:5000" target="_blank">MLflow</a><br/>
      <a href="http://localhost:9090" target="_blank">Prometheus</a><br/>
      <a href="/docs/README.md">README.md</a><br/>
      <a href="/docs/technical_report.md">docs/technical_report.md</a><br/>
      <a href="/docs/model_card.md">docs/model_card.md</a><br/>
      <a href="/docs/data_card.md">docs/data_card.md</a><br/>
      <a href="/monitoring/evidently_reports/drift.html">monitoring/evidently_reports/drift.html</a><br/>
      <a href="/monitoring/evidently_reports/baseline.html">monitoring/evidently_reports/baseline.html</a>
    </div>
  </div>
  <script>
    const fields = [
      ["season",1,4],["mnth",1,12],["hr",0,23],["holiday",0,1],["weekday",0,6],
      ["workingday",0,1],["weathersit",1,4],["temp",0,1],["atemp",0,1],["hum",0,1],["windspeed",0,1]
    ];
    const defaults = {season:2,mnth:6,hr:12,holiday:0,weekday:3,workingday:1,weathersit:2,temp:0.62,atemp:0.6,hum:0.45,windspeed:0.2};
    const form = document.getElementById("singleForm");
    function buildForm() {
      form.innerHTML = "";
      for (const [name,min,max] of fields) {
        const wrap = document.createElement("div");
        const label = document.createElement("label");
        label.innerText = name;
        const input = document.createElement("input");
        input.id = name; input.name = name;
        input.type = (name==="temp"||name==="atemp"||name==="hum"||name==="windspeed") ? "number" : "number";
        input.step = (input.type==="number" && max===1) ? "0.01" : "1";
        input.min = min; input.max = max; input.required = true;
        input.value = defaults[name];
        wrap.appendChild(label); wrap.appendChild(input); form.appendChild(wrap);
      }
      document.getElementById("batchInput").value = JSON.stringify(defaults);
    }
    function getRecord() {
      const r = {};
      for (const [name,min,max] of fields) {
        const raw = document.getElementById(name).value;
        if (raw === "") throw new Error(`Missing ${name}`);
        const val = Number(raw);
        if (Number.isNaN(val) || val < min || val > max) throw new Error(`Invalid ${name}: expected ${min}-${max}`);
        r[name] = val;
      }
      return r;
    }
    async function refreshStatus() {
      const el = document.getElementById("statusCards");
      try {
        const resp = await fetch("/api/dashboard-summary");
        const data = await resp.json();
        el.innerHTML = `
          <div class="card"><b>Health</b><div class="${data.health.status==='ok'?'ok':'bad'}">${data.health.status}</div></div>
          <div class="card"><b>Ready</b><div class="${data.ready.available?'ok':'bad'}">${data.ready.available ? "ready" : "not ready"}</div></div>
          <div class="card"><b>Model</b><div>${data.model.model_name || "n/a"} v${data.model.model_version || "n/a"}</div></div>
          <div class="card"><b>Latest Prediction</b><div>${data.runtime.latest_prediction ?? "n/a"}</div></div>
          <div class="card"><b>Latest Confidence</b><div>${data.runtime.latest_confidence ?? "n/a"}</div></div>
          <div class="card"><b>Total Predictions</b><div>${data.runtime.total_predictions}</div></div>
          <div class="card"><b>Avg Latency (ms)</b><div>${data.runtime.avg_latency_ms ?? "n/a"}</div></div>
          <div class="card"><b>Drift Summary</b><div class="${data.drift.available?'ok':'bad'}">${data.drift.available ? "available" : data.drift.message}</div></div>
        `;
      } catch (e) {
        el.innerHTML = `<div class="card bad">Status unavailable: ${String(e)}</div>`;
      }
    }
    document.getElementById("demoBtn").onclick = () => buildForm();
    document.getElementById("resetBtn").onclick = () => buildForm();
    document.getElementById("predictBtn").onclick = async () => {
      const btn = document.getElementById("predictBtn");
      const out = document.getElementById("singleResult");
      out.textContent = "";
      try {
        const payload = getRecord();
        btn.disabled = true;
        const t0 = performance.now();
        const resp = await fetch("/predict", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(payload)});
        const body = await resp.json();
        const elapsed = (performance.now() - t0).toFixed(1);
        out.textContent = JSON.stringify({status: resp.status, response_time_ms: elapsed, ...body}, null, 2);
      } catch (e) {
        out.textContent = JSON.stringify({status: "error", detail: String(e)}, null, 2);
      } finally {
        btn.disabled = false;
        refreshStatus();
      }
    };
    document.getElementById("batchBtn").onclick = async () => {
      const out = document.getElementById("batchResult");
      try {
        const lines = document.getElementById("batchInput").value.split("\\n").map(s => s.trim()).filter(Boolean);
        const records = lines.map(line => JSON.parse(line));
        const resp = await fetch("/predict/batch", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({records})});
        const body = await resp.json();
        out.textContent = JSON.stringify({status: resp.status, ...body}, null, 2);
      } catch (e) {
        out.textContent = JSON.stringify({status: "error", detail: String(e)}, null, 2);
      } finally {
        refreshStatus();
      }
    };
    buildForm();
    refreshStatus();
  </script>
</body>
</html>
"""


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
    try:
        loaded = load_artifacts()
        app.state.loaded_model = loaded
        MODEL_VERSION.labels(version=loaded.model_version).set(1)
        log.info("model loaded name=%s version=%s", loaded.model_name, loaded.model_version)
    except Exception as exc:
        app.state.load_error = str(exc)
        log.exception("startup model load failed")
    yield


app = FastAPI(title="Bike Sharing Predictor", lifespan=lifespan)


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


@app.get("/api/status")
def api_status() -> dict[str, Any]:
    health_status = health()
    ready_state = ready()
    return {
        "health": health_status.model_dump(),
        "ready": ready_state,
        "metrics_available": True,
        "dashboard_url": DASHBOARD_URL,
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
    }


@app.get("/api/drift-summary")
def drift_summary() -> dict[str, Any]:
    return _drift_summary()


@app.get("/api/dashboard-summary")
def dashboard_summary() -> dict[str, Any]:
    return {
        "health": health().model_dump(),
        "ready": ready(),
        "model": model_info(),
        "runtime": getattr(app.state, "stats", _default_stats()),
        "drift": _drift_summary(),
        "links": {
            "mlflow": "http://localhost:5000",
            "prometheus": "http://localhost:9090",
            "dashboard": DASHBOARD_URL,
        },
    }


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
    return BatchPredictResponse(predictions=items, model_version=loaded.model_version)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
