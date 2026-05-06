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
from starlette.middleware.base import BaseHTTPMiddleware

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
RECENT_HISTORY_LIMIT = 20


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
        ("Drift Report", ROOT_DIR / "monitoring" / "evidently_reports" / "drift.html", "/monitoring/evidently_reports/drift.html"),
        ("Baseline Report", ROOT_DIR / "monitoring" / "evidently_reports" / "baseline.html", "/monitoring/evidently_reports/baseline.html"),
        ("Drift Summary", ROOT_DIR / "monitoring" / "evidently_reports" / "drift_summary.json", "/monitoring/evidently_reports/drift_summary.json"),
    ]
    external = [
        ("MLflow", "http://localhost:5000"),
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
            "ui_url": "http://localhost:5000",
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
  <noscript>
    <div style="max-width:1280px;margin:10px auto;padding:8px 24px;color:#D29922;font-family:Inter,-apple-system,system-ui,sans-serif;font-size:12px;">
      This dashboard needs JavaScript to predict. Endpoint docs: <a href="/docs">/docs</a>
    </div>
  </noscript>
  <main class="container">
    <div id="bootBanner" class="boot-banner">Model loading…</div>
    <header class="header-strip">
      <div class="title-wrap">
        <h1>Bike sharing demand</h1>
        <p id="subTitle">MLOps serving · v1 · local</p>
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
        <div class="kv"><span>Model</span><span id="modelVal" class="mono">v1 · local</span></div>
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
      <div class="eyebrow">Available</div>
      <div id="evidenceAvailable"></div>
      <div class="divider"></div>
      <div class="eyebrow">Pending</div>
      <div id="evidencePending" class="subtle"></div>
    </section>
  </main>
  <script src="/static/safety.js"></script>
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
textarea{font-family:var(--font-mono);font-size:11px;height:110px;resize:vertical}#batchInput{padding:8px;height:110px;background:var(--surface-2);border:.5px solid var(--border);border-radius:6px}input:focus,select:focus,textarea:focus,button:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 1px var(--accent)}
.toggle{display:flex;align-items:center;gap:8px;padding-top:18px}.toggle input{width:16px;height:16px;padding:0}.value-hint{color:var(--text-2);font-size:11px;margin-top:4px}
.button-row{display:grid;gap:8px;margin-top:6px}.button-row-2{grid-template-columns:1fr 1fr}button{cursor:pointer}.btn-primary{background:var(--accent);color:var(--bg);border-color:transparent;font-weight:500}.btn-primary:hover{background:var(--accent-dim)}.btn-secondary{background:transparent;color:var(--text);font-weight:400}.btn-secondary:hover{border-color:var(--border-2)}
.msg{min-height:16px;color:var(--text-2);font-size:11px;margin-top:8px}.msg-bad{color:var(--bad)}.field-error{color:var(--bad);font-size:11px;margin-top:3px}.boot-banner{display:none;padding:6px 10px;border:.5px solid var(--border);border-radius:8px;color:var(--warn);background:var(--surface);font-size:12px}.boot-banner.show{display:block}.row-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
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
    return """
(()=>{
const TEMP_MIN_C=-8,TEMP_MAX_C=39,ATEMP_MIN_C=-16,ATEMP_MAX_C=50,WIND_MAX_KMH=67;
const fmtInt=(n)=>Math.round(Number(n||0)).toLocaleString(),fmtPct=(x)=>`${Math.round((Number(x)||0)*100)}%`,fmtMs=(n)=>`${Math.round(Number(n||0))} ms`,fmtConf=(x)=>x>=0.8?'High':x>=0.6?'Medium':'Low';
const monthNames=["January","February","March","April","May","June","July","August","September","October","November","December"],weatherLabels={1:"Clear",2:"Mist",3:"Light rain",4:"Heavy"},weekdayNames=["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"];
const state={singleCount:0,batchCount:0,lastSinglePayload:null,singleStatus:"idle",batchStatus:"idle",healthFailures:0,submitsEnabled:false,predictGuardUntil:0,batchGuardUntil:0};
const sampleHuman={season:2,mnth:6,hr:12,weekday:3,holiday:0,workingday:1,weathersit:2,temp_c:21,atemp_c:24,hum_pct:45,wind_kmh:14};
const byId=(id)=>document.getElementById(id),setMsg=(id,msg)=>{const el=byId(id);if(el)el.textContent=msg||"";};

async function safeFetch(url,opts={},cfg={timeoutMs:8000,retries:1}){const {timeoutMs=8000,retries=1}=cfg;for(let attempt=0;attempt<=retries;attempt++){const controller=new AbortController();const timer=setTimeout(()=>controller.abort(),timeoutMs);try{const res=await fetch(url,{...opts,signal:controller.signal});clearTimeout(timer);if(res.status>=500&&attempt<retries)continue;const text=await res.text();let body;try{body=JSON.parse(text);}catch(_e){body={raw:text};}if(!res.ok){return{ok:false,status:res.status,error:body.detail||body.error||`HTTP ${res.status}`};}return{ok:true,status:res.status,body};}catch(err){clearTimeout(timer);if(err.name==="AbortError"){if(attempt<retries)continue;return{ok:false,status:0,error:"Request timed out"};}if(attempt<retries)continue;return{ok:false,status:0,error:"Network error"};}}return{ok:false,status:0,error:"Network error"};}

function setBanner(text,show=true){const el=byId("bootBanner");if(!el)return;el.textContent=text;el.classList.toggle("show",show);}
function setPredictButton(loading){const btn=byId("predictBtn");btn.disabled=loading||!state.submitsEnabled;btn.textContent=loading?"Predicting…":"Predict";}
function setBatchButton(loading){const btn=byId("batchBtn");btn.disabled=loading||!state.submitsEnabled;}
function clearFieldErrors(){document.querySelectorAll(".field-error").forEach((n)=>{n.textContent="";});}
function setFieldError(id,msg){const el=byId(id+"_err");if(el)el.textContent=msg;}

function optionRange(min,max,selected,labels){let html="";for(let i=min;i<=max;i++){const txt=labels&&labels[i]!==undefined?labels[i]:String(i);html+=`<option value="${i}" ${Number(selected)===i?"selected":""}>${txt}</option>`}return html}
function renderForm(){byId("whenGridA").innerHTML=[`<div><label for="season">Season</label><select id="season">${optionRange(1,4,sampleHuman.season,{1:"Spring",2:"Summer",3:"Fall",4:"Winter"})}</select><div id="season_err" class="field-error"></div></div>`,`<div><label for="mnth">Month</label><select id="mnth">${optionRange(1,12,sampleHuman.mnth,Object.fromEntries(monthNames.map((m,i)=>[i+1,m])))}</select><div id="mnth_err" class="field-error"></div></div>`,`<div><label for="hr">Hour</label><select id="hr">${optionRange(0,23,sampleHuman.hr,Object.fromEntries(Array.from({length:24},(_,h)=>[h,`${String(h).padStart(2,"0")}:00`])) )}</select><div id="hr_err" class="field-error"></div></div>`].join("");byId("whenGridB").innerHTML=[`<div><label for="weekday">Day</label><select id="weekday">${optionRange(0,6,sampleHuman.weekday,Object.fromEntries(weekdayNames.map((d,i)=>[i,d])))}</select><div id="weekday_err" class="field-error"></div></div>`,`<div class="toggle"><input id="holiday" type="checkbox" ${sampleHuman.holiday?"checked":""}/><label for="holiday">Holiday</label></div>`,`<div class="toggle"><input id="workingday" type="checkbox" ${sampleHuman.workingday?"checked":""}/><label for="workingday">Working day</label></div>`].join("");byId("condGrid").innerHTML=[`<div><label for="weathersit">Weather</label><select id="weathersit">${optionRange(1,4,sampleHuman.weathersit,{1:"Clear",2:"Mist + cloudy",3:"Light rain or snow",4:"Heavy rain or snow"})}</select><div id="weathersit_err" class="field-error"></div></div>`,`<div><label for="hum_pct">Humidity</label><input id="hum_pct" type="range" min="0" max="100" value="${sampleHuman.hum_pct}" /><div id="hum_pct_v" class="value-hint">${sampleHuman.hum_pct}%</div><div id="hum_err" class="field-error"></div></div>`,`<div><label for="temp_c">Temp</label><input id="temp_c" type="range" min="${TEMP_MIN_C}" max="${TEMP_MAX_C}" step="1" value="${sampleHuman.temp_c}" /><div id="temp_c_v" class="value-hint">${sampleHuman.temp_c} °C</div><div id="temp_err" class="field-error"></div></div>`,`<div><label for="wind_kmh">Wind</label><input id="wind_kmh" type="range" min="0" max="${WIND_MAX_KMH}" step="1" value="${sampleHuman.wind_kmh}" /><div id="wind_kmh_v" class="value-hint">${sampleHuman.wind_kmh} km/h</div><div id="windspeed_err" class="field-error"></div></div>`].join("");[["hum_pct","%"],["temp_c"," °C"],["wind_kmh"," km/h"]].forEach(([id,suf])=>{byId(id).addEventListener("input",(e)=>{byId(`${id}_v`).textContent=`${Math.round(Number(e.target.value))}${suf}`})})}
function toPayload(){const season=Number(byId("season").value),mnth=Number(byId("mnth").value),hr=Number(byId("hr").value),weekday=Number(byId("weekday").value),holiday=byId("holiday").checked?1:0,workingday=byId("workingday").checked?1:0,weathersit=Number(byId("weathersit").value),hum_pct=Number(byId("hum_pct").value),temp_c=Number(byId("temp_c").value),atemp_c=temp_c,wind_kmh=Number(byId("wind_kmh").value);return{season,mnth,hr,holiday,weekday,workingday,weathersit,temp:(temp_c-TEMP_MIN_C)/(TEMP_MAX_C-TEMP_MIN_C),atemp:(atemp_c-ATEMP_MIN_C)/(ATEMP_MAX_C-ATEMP_MIN_C),hum:hum_pct/100,windspeed:wind_kmh/WIND_MAX_KMH};}
function validateSinglePayload(payload){clearFieldErrors();const r=window.validatePayload(payload);if(r.ok)return r;for(const err of r.errors){const field=String(err).split(" ")[0];setFieldError(field,err);}return r;}

function drawSpark(target,vals){if(!target)return;target.innerHTML="";if(!vals.length)return;const w=580,h=40,p=3,mn=Math.min(...vals),mx=Math.max(...vals),sx=(i)=>p+(i/Math.max(vals.length-1,1))*(w-p*2),sy=(v)=>h-p-((v-mn)/Math.max(mx-mn,1))* (h-p*2),d=vals.map((v,i)=>`${i?"L":"M"} ${sx(i)} ${sy(v)}`).join(" ");const svg=document.createElementNS("http://www.w3.org/2000/svg","svg");svg.setAttribute("viewBox",`0 0 ${w} ${h}`);svg.setAttribute("width","100%");svg.setAttribute("height","40");const path=document.createElementNS(svg.namespaceURI,"path");path.setAttribute("d",d);path.setAttribute("fill","none");path.setAttribute("stroke","var(--accent)");path.setAttribute("stroke-width","1.5");svg.appendChild(path);target.appendChild(svg);}
function drawLineChart(target,points,currentX){if(!target)return;target.textContent=points.length?`Chart points: ${points.length}`:"No data";}
function drawBarChart(target,items){if(!target)return;target.textContent=items.length?`Bars: ${items.length}`:"No data";}

function renderStatus(summary){const healthOk=summary.health&&summary.health.status==="ok",modelReady=summary.ready&&summary.ready.available,ml=summary.mlflow||{},row=[{ok:healthOk,label:`API ${healthOk?"online":"offline"}`},{ok:modelReady,label:`Model ${modelReady?"ready":"not ready"}`},{ok:!!ml.available,label:`MLflow ${ml.available?"connected":"unavailable"}`},{warn:true,label:`Registry ${ml.registered_model_available?"available":"missing"}`}];const statusRow=byId("statusRow");statusRow.innerHTML=row.map(r=>`<span class="status-item"><span class="dot" style="background:${r.warn?"var(--warn)":(r.ok?"var(--ok)":"var(--bad)")}"></span>${r.label}</span>`).join("");}
function renderKPIs(summary,recent){const run=summary.runtime||{},latest=run.latest_prediction==null?"—":fmtInt(run.latest_prediction),conf=Number(run.latest_confidence||0),avg=run.avg_latency_ms==null?"—":fmtMs(run.avg_latency_ms),p95=recent.length?fmtMs([...recent.map(r=>Number(r.latency_ms||0))].sort((a,b)=>a-b)[Math.floor(0.95*(recent.length-1))]):"—",total=fmtInt(run.total_predictions||0),subCount=`${fmtInt(state.batchCount)} batch · ${fmtInt(state.singleCount)} single`;byId("kpiGrid").innerHTML=`<div class="kpi"><div class="label">Latest prediction</div><div class="value big">${latest} <span class="sub">rentals/hr</span></div><div id="kpiSpark" class="spark-inline"></div></div><div class="kpi"><div class="label">Confidence</div><div class="value">${fmtPct(conf)}</div><div class="sub">${fmtConf(conf)}</div></div><div class="kpi"><div class="label">Avg latency</div><div class="value">${avg}</div><div class="sub">p95 · ${p95}</div></div><div class="kpi"><div class="label">Predictions today</div><div class="value">${total}</div><div class="sub">${subCount}</div></div>`;drawSpark(byId("kpiSpark"),recent.map(r=>Number(r.prediction||0)).slice(-20));}
function renderRecent(items){const top=items.slice(0,10),tbody=byId("recentTable").querySelector("tbody");tbody.innerHTML="";for(const it of top){const tr=document.createElement("tr");["timestamp","hr","prediction","confidence","latency"].forEach((k)=>{const td=document.createElement("td");if(k==="timestamp")td.textContent=it.timestamp||"—";if(k==="hr")td.textContent=String((it.input_summary&&it.input_summary.hr!=null)?it.input_summary.hr:"—");if(k==="prediction")td.textContent=fmtInt(it.prediction);if(k==="confidence")td.textContent=fmtPct(it.confidence);if(k==="latency")td.textContent=fmtMs(it.latency_ms);if(k!=="hr")td.className="mono";tr.appendChild(td);});tbody.appendChild(tr);}if(!top.length)setMsg("recentMsg","No predictions yet.");else setMsg("recentMsg","");drawSpark(byId("recentSpark"),top.slice().reverse().map(it=>Number(it.prediction||0)));}
function renderServices(summary){const ml=summary.mlflow||{},rows=[{name:"Health",ok:summary.health&&summary.health.status==="ok",endpoint:"/health"},{name:"Readiness",ok:summary.ready&&summary.ready.available,endpoint:"/ready"},{name:"MLflow",ok:!!ml.available,endpoint:"http://localhost:5000"}],up=rows.filter(r=>r.ok).length;byId("serviceCount").textContent=`${up} / ${rows.length} up`;const wrap=byId("serviceRows");wrap.innerHTML="";for(const r of rows){const row=document.createElement("div");row.className="service-row";const left=document.createElement("span");left.className="service-name";left.innerHTML=`<span class="dot" style="background:${r.ok?"var(--ok)":"var(--bad)"}"></span>${r.name}`;const right=document.createElement("span");right.className="endpoint";right.textContent=r.endpoint;row.append(left,right);wrap.appendChild(row);}}
function renderEvidence(items){const available=items.filter(i=>i.exists),pending=items.filter(i=>!i.exists);byId("evidenceCount").textContent=`${available.length} of ${items.length} available`;const wrap=byId("evidenceAvailable");wrap.innerHTML="";for(const i of available){const row=document.createElement("div");row.className="ev-row";const check=document.createElement("span");check.className="check";check.textContent="✓";row.appendChild(check);if(i.url){const a=document.createElement("a");a.target="_blank";a.rel="noopener noreferrer";a.href=i.url;a.textContent=i.name;row.appendChild(a);}else{const s=document.createElement("span");s.textContent=i.name;row.appendChild(s);}wrap.appendChild(row);}byId("evidencePending").textContent=pending.length?pending.map(i=>i.name).join(" · "):"None";}
async function refresh(){const [summaryRes,recentRes,evidenceRes]=await Promise.all([safeFetch("/api/dashboard-summary"),safeFetch("/api/recent-predictions"),safeFetch("/api/evidence-status")]);if(!summaryRes.ok){setMsg("singleError","Couldn't reach the model. Try again.");return;}const summary=summaryRes.body,recent=recentRes.ok?recentRes.body:{items:[]},evidence=evidenceRes.ok?evidenceRes.body:{items:[]};renderStatus(summary);renderKPIs(summary,recent.items||[]);renderRecent((recent.items||[]).slice().reverse());renderServices(summary);renderEvidence(evidence.items||[]);}

async function bootHealthCheck(){setBanner("Model loading…",true);for(let i=0;i<3;i++){const r=await safeFetch("/health",{}, {timeoutMs:3000,retries:0});if(r.ok&&r.body&&r.body.status==="ok"){state.submitsEnabled=true;setBanner("",false);setPredictButton(false);setBatchButton(false);return;}state.healthFailures+=1;}state.submitsEnabled=false;setBanner("Service unavailable. Check the server.",true);setPredictButton(false);setBatchButton(false);}

function collectBatchValidation(){const text=byId("batchInput").value.trim();if(!text)return{ok:false,errors:["Paste at least one JSON object"]};if(text.length>50000)return{ok:false,errors:["Input too large (max 50KB)"]};const lines=text.split("\\n").map((s)=>s.trim()).filter(Boolean);if(lines.length>100)return{ok:false,errors:["Too many rows (max 100)"]};const items=[];const errs=[];for(let idx=0;idx<lines.length;idx++){let parsed;try{parsed=JSON.parse(lines[idx]);}catch(_e){errs.push(`Line ${idx+1}: invalid JSON`);continue;}if(!parsed||Array.isArray(parsed)||typeof parsed!=="object"){errs.push(`Line ${idx+1}: expected JSON object`);continue;}const r=window.validatePayload(parsed);if(!r.ok){for(const e of r.errors){errs.push(`Line ${idx+1}: ${e}`);}continue;}items.push(r.value);}if(errs.length)return{ok:false,errors:errs.slice(0,10).concat(errs.length>10?[`+${errs.length-10} more`]:[])};return{ok:true,items};}

async function runPredict(){const now=Date.now();if(now<state.predictGuardUntil)return;state.predictGuardUntil=now+250;if(state.singleStatus==="loading"||!state.submitsEnabled)return;const payload=toPayload();const vr=validateSinglePayload(payload);if(!vr.ok){setMsg("singleError","Please fix highlighted fields.");return;}state.singleStatus="loading";setPredictButton(true);setMsg("singleError","");state.lastSinglePayload=payload;const t0=performance.now();const res=await safeFetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(vr.value)},{timeoutMs:8000,retries:1});const elapsed=performance.now()-t0;if(!res.ok){setMsg("singleError","Couldn't reach the model. Try again.");state.singleStatus="error";setPredictButton(false);return;}const out=res.body||{};if(!Number.isFinite(Number(out.prediction))||!Number.isFinite(Number(out.confidence))||Number(out.confidence)<0||Number(out.confidence)>1){setMsg("singleError","Model returned an invalid value. Try different inputs.");state.singleStatus="error";setPredictButton(false);return;}byId("predValue").textContent=fmtInt(out.prediction);byId("confPct").textContent=fmtPct(out.confidence);byId("confFill").style.width=fmtPct(out.confidence);byId("latencyVal").textContent=fmtMs(elapsed);state.singleCount+=1;state.singleStatus="success";setPredictButton(false);await refresh();}
async function runBatch(){const now=Date.now();if(now<state.batchGuardUntil)return;state.batchGuardUntil=now+250;if(state.batchStatus==="loading"||!state.submitsEnabled)return;setMsg("batchError","");const v=collectBatchValidation();if(!v.ok){setMsg("batchError",v.errors.join(" | "));return;}state.batchStatus="loading";setBatchButton(true);const t0=performance.now();const res=await safeFetch("/predict/batch",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({records:v.items})},{timeoutMs:8000,retries:1});const latency=(performance.now()-t0)/Math.max(v.items.length,1);if(!res.ok){setMsg("batchError","Couldn't reach the model. Try again.");state.batchStatus="error";setBatchButton(false);return;}const preds=res.body&&res.body.predictions?res.body.predictions:[];const tbody=byId("batchTable").querySelector("tbody");tbody.innerHTML="";preds.forEach((p,i)=>{const tr=document.createElement("tr");[String(i+1),String(v.items[i].hr),fmtInt(p.prediction),fmtPct(p.confidence),fmtMs(latency)].forEach((txt,idx)=>{const td=document.createElement("td");td.textContent=txt;if(idx>=2)td.className="mono";tr.appendChild(td);});tbody.appendChild(tr);});drawBarChart(byId("batchChart"),preds.map((p,i)=>({label:String(v.items[i].hr),value:p.prediction,active:true})));state.batchCount+=v.items.length;state.batchStatus="success";setBatchButton(false);await refresh();}

function loadSample(){byId("season").value=sampleHuman.season;byId("mnth").value=sampleHuman.mnth;byId("hr").value=sampleHuman.hr;byId("weekday").value=sampleHuman.weekday;byId("holiday").checked=!!sampleHuman.holiday;byId("workingday").checked=!!sampleHuman.workingday;byId("weathersit").value=sampleHuman.weathersit;byId("hum_pct").value=sampleHuman.hum_pct;byId("temp_c").value=sampleHuman.temp_c;byId("wind_kmh").value=sampleHuman.wind_kmh;byId("hum_pct_v").textContent=`${sampleHuman.hum_pct}%`;byId("temp_c_v").textContent=`${sampleHuman.temp_c} °C`;byId("wind_kmh_v").textContent=`${sampleHuman.wind_kmh} km/h`;clearFieldErrors();}
function loadSampleBatch(){const toNorm=(h)=>({season:h.season,mnth:h.mnth,hr:h.hr,holiday:h.holiday,weekday:h.weekday,workingday:h.workingday,weathersit:h.weathersit,temp:(h.temp_c-TEMP_MIN_C)/(TEMP_MAX_C-TEMP_MIN_C),atemp:(h.atemp_c-ATEMP_MIN_C)/(ATEMP_MAX_C-ATEMP_MIN_C),hum:h.hum_pct/100,windspeed:h.wind_kmh/WIND_MAX_KMH});const items=[sampleHuman,{...sampleHuman,hr:8,temp_c:14,atemp_c:16,hum_pct:72,weathersit:3},{...sampleHuman,hr:18,temp_c:26,atemp_c:29,hum_pct:40,weathersit:1}].map(toNorm);byId("batchInput").value=items.map((v)=>JSON.stringify(v)).join("\\n");byId("batchCounter").textContent=`${items.length} / 100 lines`;}

renderForm();loadSample();loadSampleBatch();setPredictButton(false);setBatchButton(false);byId("predictBtn").addEventListener("click",runPredict);byId("demoBtn").addEventListener("click",loadSample);byId("batchBtn").addEventListener("click",runBatch);byId("sampleBatchBtn").addEventListener("click",loadSampleBatch);byId("batchInput").addEventListener("input",()=>{const n=byId("batchInput").value.split("\\n").map((s)=>s.trim()).filter(Boolean).length;byId("batchCounter").textContent=`${n} / 100 lines`;});bootHealthCheck().then(refresh);
})();
"""


def _safety_js() -> str:
    return """
window.FIELD_RULES = {
  season: { type: 'int', min: 1, max: 4 },
  mnth: { type: 'int', min: 1, max: 12 },
  hr: { type: 'int', min: 0, max: 23 },
  holiday: { type: 'int', enum: [0, 1] },
  weekday: { type: 'int', min: 0, max: 6 },
  workingday: { type: 'int', enum: [0, 1] },
  weathersit: { type: 'int', min: 1, max: 4 },
  temp: { type: 'float', min: 0, max: 1 },
  atemp: { type: 'float', min: 0, max: 1 },
  hum: { type: 'float', min: 0, max: 1 },
  windspeed: { type: 'float', min: 0, max: 1 },
};
window.validateField = function validateField(name, raw) {
  const rule = window.FIELD_RULES[name];
  if (!rule) return { ok: false, error: `Unknown field: ${name}` };
  if (raw === null || raw === undefined || raw === '') return { ok: false, error: `${name} is required` };
  const n = rule.type === 'int' ? parseInt(raw, 10) : parseFloat(raw);
  if (!Number.isFinite(n)) return { ok: false, error: `${name} must be a number` };
  if (rule.enum && !rule.enum.includes(n)) return { ok: false, error: `${name} must be one of ${rule.enum.join(', ')}` };
  if (rule.min !== undefined && n < rule.min) return { ok: false, error: `${name} must be ≥ ${rule.min}` };
  if (rule.max !== undefined && n > rule.max) return { ok: false, error: `${name} must be ≤ ${rule.max}` };
  return { ok: true, value: n };
};
window.validatePayload = function validatePayload(obj) {
  const out = {};
  const errors = [];
  for (const key of Object.keys(window.FIELD_RULES)) {
    const r = window.validateField(key, obj[key]);
    if (r.ok) out[key] = r.value;
    else errors.push(r.error);
  }
  for (const key of Object.keys(obj)) {
    if (!(key in window.FIELD_RULES)) errors.push(`Unknown field: ${key}`);
  }
  return errors.length ? { ok: false, errors } : { ok: true, value: out };
};
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
    app.state.recent_predictions = []
    try:
        loaded = load_artifacts()
        app.state.loaded_model = loaded
        MODEL_VERSION.labels(version=loaded.model_version).set(1)
        log.info("model loaded name=%s version=%s", loaded.model_name, loaded.model_version)
    except Exception as exc:
        app.state.load_error = str(exc)
        log.exception("startup model load failed")
    yield


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 200_000:
            return JSONResponse(status_code=413, content={"error": "Payload too large"})
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
        )
        return response


app = FastAPI(title="Bike Sharing Predictor", lifespan=lifespan)
app.add_middleware(BodySizeLimitMiddleware)
app.add_middleware(SecurityHeadersMiddleware)


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


@app.get("/static/safety.js")
def safety_js() -> Response:
    return PlainTextResponse(_safety_js(), media_type="application/javascript")


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
        "mlflow": _mlflow_status_payload(),
        "runtime": getattr(app.state, "stats", _default_stats()),
        "drift": _drift_summary(),
        "links": {
            "mlflow": "http://localhost:5000",
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
            "model_source": "mlflow" if loaded and loaded.model_version != "local" else "local",
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
            "model_source": "mlflow" if loaded and loaded.model_version != "local" else "local",
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
    err_id = str(uuid.uuid4())[:8]
    log.exception("unhandled error %s on %s", err_id, request.url.path)
    return JSONResponse(status_code=500, content={"error": "Internal error", "id": err_id})


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
    return BatchPredictResponse(predictions=items, model_version=loaded.model_version)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
