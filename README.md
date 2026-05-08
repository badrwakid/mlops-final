# Bike Sharing Demand — End-to-End MLOps Pipeline

DDSC611 Final Project · Spring 2026 · ESLSCA University

Predicts hourly bike rental counts using the [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset). Covers all eight mandatory pipeline components plus both bonus components (Docker + Prefect orchestration).

---

## Quickstart (3 commands)

```bash
git clone https://github.com/badrwakid/mlops-final.git && cd mlops-final
docker compose up --build -d
curl http://localhost:8000/health
```

The stack starts automatically in the correct order:
`mlflow` → `seed-mlflow` (registers Production model) → `api` → `prometheus` → `grafana`

---

## Architecture

```
Raw Data (UCI CSV)
      │
      ▼
  DVC Pipeline ──────────────────────────────────────────────────────┐
  dvc.yaml stages:                                                    │
    prepare → preprocess → featurize → train                         │
      │               │           │         │                        │
  hour.csv      splits/        preprocessor  model.pkl  metrics.json │
  (DagsHub)    parquets         .pkl                                 │
                                                                     │
                        MLflow Tracking Server ◄───────────────────────┘
                        (Docker :5001)
                              │
                    Model Registry (Production)
                              │
                        FastAPI :8000
                        /predict  /health  /metrics
                              │
                 ┌────────────┼────────────┐
                 │            │            │
           Prometheus    Grafana       Evidence
            :9090         :3000       Dashboard
                                     (API HTML)
                              │
                      Evidently Drift Reports
                      monitoring/evidently_reports/
```

---

## Services

| Service | URL | Credentials |
|---|---|---|
| API + HTML dashboard | http://localhost:8000 | — |
| MLflow UI | http://localhost:5001 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |

---

## Pipeline Components

### 1 · Data Versioning (DVC)

```bash
dvc repro          # reproduce all pipeline stages deterministically
dvc push           # push artifacts to DagsHub remote
dvc pull           # restore artifacts on a fresh clone
dvc dag            # visualise pipeline DAG
```

Remote: DagsHub (`https://dagshub.com/badrwakid/mlops-final.dvc`). See `.dvc/config`.

Tracked artifacts: `data/raw/hour.csv`, `data/processed/`, `data/splits/` (preprocessor, model, splits, metrics).

### 2 · Preprocessing Pipeline

scikit-learn `Pipeline` with imputation, scaling, and categorical encoding. All parameters in `configs/params.yaml` — no hardcoded values. Saved as `data/splits/preprocessor.pkl` (DVC-tracked).

```bash
python -m src.data.prepare        # prepare raw → clean
python -m src.features.featurize  # fit and save preprocessor
```

### 3 · Experiment Tracking & Model Registry

MLflow tracking server runs as a Docker service. 60 Optuna HPO trials logged per training run. Best model registered and promoted to Production via API.

```bash
# train with Optuna HPO and log to local MLflow
MLFLOW_TRACKING_URI=http://127.0.0.1:5001 PYTHONPATH=. python -m src.training.train

# export experiment log
python scripts/export_experiment_log.py
```

MLflow UI: http://localhost:5001

### 4 · Model Serving (FastAPI)

```bash
# health check
curl http://localhost:8000/health

# single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"season":2,"mnth":6,"hr":12,"holiday":0,"weekday":3,"workingday":1,
       "weathersit":1,"temp":0.5,"atemp":0.5,"hum":0.5,"windspeed":0.2}'

# batch prediction (up to 100 records)
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"records":[{"season":2,"mnth":6,"hr":8,"holiday":0,"weekday":1,
       "workingday":1,"weathersit":1,"temp":0.5,"atemp":0.5,"hum":0.5,"windspeed":0.2}]}'
```

Key endpoints: `GET /health`, `GET /ready`, `POST /predict`, `POST /predict/batch`, `GET /metrics`, `GET /api/evidence-status`.

Interactive dashboard: http://localhost:8000

### 5 · CI/CD (GitHub Actions)

Six-job pipeline on every push to `main`:

```
lint → test (coverage ≥70%) → data-validation → model-validation → compose-validate → monitoring-validation
```

See `.github/workflows/ci.yml`. Artifacts pulled from DagsHub via `DVC_REMOTE_URL` + `DAGSHUB_TOKEN` secrets; falls back to UCI download + `dvc repro` if secrets are absent.

### 6 · Monitoring & Drift Detection

```bash
# generate both Evidently reports
python -m monitoring.run_monitoring
```

Outputs written to `monitoring/evidently_reports/`:
- `baseline.html` — reference vs clean held-out set (minimal drift expected)
- `drift.html` — reference vs production set with feature drift
- `drift_summary.json` — machine-readable drift alert payload
- `interpretation.md` — decision runbook

Drift threshold: **20% of input features**. Alert triggers `P2` investigation; two consecutive windows trigger retraining.

Five custom Prometheus metrics:
- `bike_prediction_confidence` — confidence histogram
- `bike_feature_temp` — temperature feature histogram
- `bike_feature_hr` — hour feature histogram
- `bike_model_version_info` — currently loaded model version (gauge)
- `bike_inference_total` — inference count by endpoint and output class

Grafana dashboard at http://localhost:3000 — pre-provisioned, no setup required (login: `admin` / `admin`).

### 7 · Documentation

| File | Purpose |
|---|---|
| `docs/model_card.md` | Model description, metrics, limitations, ethical considerations |
| `docs/data_card.md` | Dataset source, schema, preprocessing, biases, licensing |
| `docs/technical_report.md` | Pipeline evidence: DVC DAG, HPO results, CI screenshots |
| `docs/experiment_log.csv` | All MLflow runs exported (parameters + metrics) |

### 8 · Project Setup & Reproducibility

```bash
# install dependencies
pip install -r requirements.txt

# verify the full stack locally (matches CI)
powershell -ExecutionPolicy Bypass -File scripts/run_full_ci_local.ps1
```

All pipeline parameters are in `configs/params.yaml`. No hardcoded values in source files. `.gitignore` excludes all data, model, and log files — tracked via DVC.

---

## Bonus A · Docker Containerisation

Full multi-service Docker Compose stack:

```bash
docker compose up --build          # start everything
docker compose down                # stop everything
docker compose logs api --follow   # tail API logs
```

Services: `mlflow`, `seed-mlflow` (init container), `api`, `prometheus`, `grafana`.

All inter-service communication uses Docker internal networking (service names as hostnames).

## Bonus B · Pipeline Orchestration (Prefect)

Five-task Prefect flow encoding the full training pipeline:

```bash
# run the full training flow locally (no Prefect server required)
PREFECT_API_URL="" MLFLOW_TRACKING_URI=http://127.0.0.1:5001 PYTHONPATH=. \
  python flows/training_flow.py
```

Tasks: `validate_data` → `preprocess` → `train` → `evaluate` → `register_model`.

Flow definition: `flows/training_flow.py`. Prefect config: `prefect.yaml`.

---

## Repository Structure

```
.github/workflows/ci.yml          # CI/CD pipeline (6 jobs)
configs/params.yaml               # all pipeline parameters
data/
  raw/                            # tracked by DVC (not in git)
  processed/                      # tracked by DVC
  splits/                         # tracked by DVC (model.pkl, preprocessor.pkl, *.parquet)
docker/
  api.Dockerfile
  mlflow.Dockerfile
docker-compose.yml
docs/
  model_card.md
  data_card.md
  technical_report.md
  experiment_log.csv
dvc.yaml                          # pipeline definition
dvc.lock                          # deterministic hash lock
flows/training_flow.py            # Prefect orchestration
monitoring/
  run_monitoring.py               # Evidently drift pipeline
  evidently_reports/              # generated HTML reports + JSON summary
  grafana/                        # provisioned Grafana dashboards
  prometheus/prometheus.yml       # Prometheus scrape config
requirements.txt                  # pinned dependencies
src/
  data/                           # loading, validation, preprocessing
  features/                       # featurization
  training/                       # HPO, training, registry
  evaluation/                     # metrics
  serving/                        # FastAPI app + Prometheus metrics
tests/
  unit/
  integration/
  data/
```

---

## Drift → Retrain Loop

1. Run drift check: `python -m monitoring.run_monitoring`
2. If `drift_share_inputs_only > 0.20` → alert fires
3. Retrain: `MLFLOW_TRACKING_URI=http://127.0.0.1:5001 PYTHONPATH=. python -m src.training.train`
4. New model auto-registered in MLflow; promote to Production via UI or API
5. API reloads model on next startup — no fallback, registry only

---

## DVC Remote (DagsHub)

The project uses DagsHub as the DVC remote. On a fresh clone with credentials:

```bash
dvc remote modify --local localremote password <DAGSHUB_TOKEN>
dvc pull
```

Token is stored locally only (`~/.dvc/config.local`) — never committed to git.
