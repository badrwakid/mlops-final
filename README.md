# Bike Sharing Demand — End-to-End MLOps Pipeline

DDSC611 Final Project · Spring 2026 · ESLSCA University

Predicts hourly bike rental counts using the [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset). Covers all eight mandatory pipeline components plus both bonus components (Docker + Prefect orchestration).

---

## Prerequisites

- **Git**
- **Python 3.11+** (see `requirements.txt`; a virtual environment is recommended)
- **Docker Engine + Docker Compose** (for `docker compose`)

On Windows, prefer **`python -m dvc`** if the **`dvc`** command is not on your `PATH`.

---

## How to run (recommended order)

Do these steps **in sequence** once per machine. Skip steps you have already finished (for example after a successful **`dvc pull`**, you do not need **`dvc repro`** before Docker).

### Step 1 — Clone the repo

```bash
git clone https://github.com/badrwakid/mlops-final.git
cd mlops-final
```

Keep **`main`** up to date (**`git pull`**) before **`dvc pull`** so **`dvc.lock`** matches the remote artifacts.

### Step 2 — Python environment

```bash
python -m venv .venv
```

Activate it (POSIX): **`source .venv/bin/activate`** · (Windows PowerShell): **`.venv\Scripts\Activate.ps1`**.

```bash
pip install -r requirements.txt
```

### Step 3 — Restore data & trained artifacts (`model.pkl`, etc.)

The API image **`COPY`**’s **`data/splits/model.pkl`**, **`preprocessor.pkl`**, **`reference.parquet`** (baseline is also in Git). You must populate those files **before** building the **`api`** service.

Pick **one** path:

**Path A — DagsHub (`dvc pull`)**

For **you on another machine**, **TA**, or **teammates** who can **`git clone`** this repo:

1. In DagsHub, open **account / developer settings** and create a **personal access token** for an account that has **read** access to the **`mlops-final`** dataset repo (**`badrwakid`** on DagsHub must invite **other users** under that repository’s access settings).
2. **`user`** must be **that account’s DagsHub username** — not your Windows/macOS/Linux login — e.g. your own **`yourname`** even when the repo owner is **`badrwakid`**.
3. Run (**`.dvc/config.local` only**; never commit these lines):

```bash
python -m dvc remote modify --local localremote user '<YOUR_DAGSHUB_USERNAME>'
python -m dvc remote modify --local localremote password '<YOUR_DAGSHUB_TOKEN>'
python -m dvc pull
```

**Path B — No remote / pull failed (same as CI)**

Rebuild artifacts from **`dvc.lock`** (downloads UCI `hour.csv`, runs the full pipeline; uses **`file:./mlruns`** unless you export **`MLFLOW_TRACKING_URI`** yourself):

```bash
python scripts/bootstrap_dvc_workspace.py
```

Publishing for others (optional): on a machine with a full **`dvc repro`**, **`python -m dvc push`** uploads blobs so Path A works elsewhere.

### Step 4 — Confirm files exist before Docker

You should see real files with non-zero size, especially:

- **`data/splits/model.pkl`**
- **`data/splits/preprocessor.pkl`**
- **`data/splits/reference.parquet`** (also tracked in Git on **`main`**)

POSIX:

```bash
ls -l data/splits/model.pkl data/splits/preprocessor.pkl data/splits/reference.parquet
```

Windows PowerShell:

```powershell
Get-Item data\splits\model.pkl, data\splits\preprocessor.pkl, data\splits\reference.parquet | Format-Table Name, Length
```

### Step 5 — Build and start the stack

From the **`mlops-final`** root:

```bash
docker compose build
docker compose up -d
```

Compose starts **`mlflow`** → **`seed-mlflow`** (registers a Production model) → **`api`** → **`prometheus`** → **`grafana`** (exact service set may match your **`docker-compose.yml`**).

Stop when finished:

```bash
docker compose down
```

Tail API logs (optional):

```bash
docker compose logs api --follow
```

### Step 6 — Smoke test

```bash
curl http://localhost:8000/health
```

Open in a browser:

- Dashboard + API · **http://localhost:8000**
- MLflow · **http://localhost:5001**
- Grafana · **http://localhost:3000** (`admin` / `admin`)

### Step 7 — Optional next steps

- **Operational drift checks:** `PYTHONPATH=. python -m monitoring.run_monitoring`
- **Local CI parity:** `powershell -ExecutionPolicy Bypass -File scripts/run_full_ci_local.ps1` (Windows; see script header on other platforms)
- **Training again (outside Docker stack):** with MLflow reachable, **`MLFLOW_TRACKING_URI=http://127.0.0.1:5001 PYTHONPATH=. python -m src.training.train`** (often after Compose is **`up`**).

You only need **`python -m dvc repro`** when you change pipeline code/data and want fresh artifacts—not before every Compose run if **`dvc pull`** already restored **`model.pkl`**.

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

Remote: DagsHub (`https://dagshub.com/badrwakid/mlops-final.dvc`). See `.dvc/config`.

- **Stored on remote after `push`:** raw `hour.csv`, processed parquet, split parquet outputs, `preprocessor.pkl`, `model.pkl`.
- **In Git (`cache: false` or small baseline):** `data/splits/metrics.json`, `data/splits/reference.parquet`.

**Fresh clone:** see **Step 3** in **[How to run (recommended order)](#how-to-run-recommended-order)** (`dvc pull` vs **`bootstrap_dvc_workspace.py`**).

Common commands:

```bash
python -m dvc repro           # rerun pipeline stages vs dvc.lock
python -m dvc push            # publish blobs after repro
python -m dvc pull             # restore on another clone (requires auth + successful prior push)
python -m dvc dag              # show pipeline DAG
```

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

See `.github/workflows/ci.yml`. If repository secrets **`DVC_REMOTE_URL`** (same URL as `.dvc/config`, e.g. `https://dagshub.com/badrwakid/mlops-final.dvc`) and **`DAGSHUB_TOKEN`** are set, CI runs `dvc pull` for heavy artifacts. If they are absent, CI uses UCI download + `dvc repro` (same flow as `scripts/bootstrap_dvc_workspace.py`).

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

### 8 · Reference

**Install + stack order:** **[How to run (recommended order)](#how-to-run-recommended-order)**.

All pipeline tuning lives in **`configs/params.yaml`**. Large binaries are restored via DVC (see **[DVC Remote (DagsHub)](#dvc-remote-dagshub)**); **`metrics.json`** and **`reference.parquet`** are intentionally in Git (see **§ 1 · Data versioning (DVC)** under **Pipeline Components** below).

---

## Bonus A · Docker notes

Compose uses internal service DNS (e.g. **`mlflow:5000`**). For the full build-start-smoke sequence, use **Steps 4–6** above; shorthand:

```bash
docker compose up --build -d
```

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
  splits/                         # mostly DVC; metrics.json + reference.parquet also in Git
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

| Action | Commands |
|---|---|
| **Publish** (so others can `pull`) | `python -m dvc repro` then `python -m dvc push` |
| **Consume** (fresh clone; token only in `.dvc/config.local`) | See **Step 3 — Path A** |
| **No remote blobs yet** | **Step 3 — Path B** (`bootstrap_dvc_workspace.py`) |

CI can **`dvc pull`** when **`DVC_REMOTE_URL`** + **`DAGSHUB_TOKEN`** are set; otherwise it mirrors Path B (`docs/github-ci-gate.md`).
