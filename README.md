# MLOps Final Project — Bike Sharing Demand

End-to-end MLOps pipeline for predicting hourly bike rental counts (UCI Bike Sharing dataset).

## Architecture overview

- **Training and reproducibility:** DVC orchestrates data prep, training, and artifact generation (`python -m dvc repro`).
- **Serving:** FastAPI app serves predictions and health/metrics endpoints on port `8000`.
- **Tracking and observability:** MLflow tracks experiments; Prometheus scrapes service metrics.

## Quickstart (rubric path)

Prerequisite (fresh clone): create and activate a virtual environment first (see [Troubleshooting and detailed setup](#troubleshooting-and-detailed-setup) for exact setup steps).
Run from the repository root:

```powershell
python -m pip install -r requirements.txt
python -m dvc repro
python -m uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

Fresh clone (self-contained):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m dvc repro
python -m uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m dvc repro
python -m uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

Health check:

```powershell
Invoke-WebRequest http://localhost:8000/health -UseBasicParsing
```

```bash
curl http://localhost:8000/health
```

## Troubleshooting and detailed setup

Use a virtual environment (recommended). From the repository root:

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**`ERROR: cannot import name '_DIR_MARK' from 'pathspec...'`** — **pathspec 1.x** removes that symbol; **DVC 3.51.x** requires **pathspec 0.11.x** (pinned in `requirements.txt`). Often an old **`pathspec==1.x`** stays on disk until you remove it explicitly:

```powershell
pip uninstall pathspec -y
pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import pathspec; from pathspec.patterns.gitwildmatch import _DIR_MARK; print(pathspec.__version__, pathspec.__file__)"
python -m dvc repro
```

Confirm `pathspec.__file__` is under **`.\.venv\Lib\site-packages`** and the version prints **`0.11.2`**.

**If PowerShell says `dvc : The term 'dvc' is not recognized`** — either activate the venv first (`.\.venv\Scripts\Activate.ps1`) so `Scripts\dvc.exe` is on `PATH`, or run DVC via Python without activating:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt   # once
.\.venv\Scripts\python.exe -m dvc repro
```

**Windows: `Permission denied` on `.venv\...\python.exe`, or `WinError 32` / “being used by another process” during `pip install`:**  
Something is **locking** files under `.venv` (common: **Cursor/VS Code** using that interpreter, **Windows Defender** scanning, or **OneDrive** syncing `Downloads`). That leaves `pip` half-finished so **`dvc` never installs** (`No module named dvc`).

1. **Close** every terminal and notebook using this repo; in Cursor/VS Code switch the Python interpreter to **system** Python (not `.venv`) or close the IDE briefly.  
2. In **Task Manager**, end stray **`python.exe`** processes if safe.  
3. **Delete the broken venv** and recreate:

```powershell
cd c:\Users\bigbo\Downloads\MLOPS_Project
Remove-Item -Recurse -Force .venv
py -3.12 -m venv .venv   # or: python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m dvc --version
```

4. If locks persist: add **`MLOPS_Project`** (or move the repo to e.g. **`C:\dev\MLOPS_Project`**, outside **Downloads/OneDrive**), or temporarily **exclude** `.venv` from real-time antivirus scanning.  
5. Prefer a **3.11+** interpreter consistent with **`docker/api.Dockerfile`** (`python:3.11.9-slim`) when possible.


**MLflow and Docker (two different stories):**

- **`dvc repro` / training does *not* require Docker.** If `configs/params.yaml` still points at `http://localhost:5000` and you have **no** server listening, either start MLflow below **or** set a file-backed store before repro (see **`MLFLOW_TRACKING_URI=file:./mlruns`** earlier in this README).
- **Bonus A / hosted tracking UI:** start only the MLflow service when you want the stack or the browser UI:

```bash
docker compose up mlflow
```

UI from the host: [http://localhost:5000](http://localhost:5000). In **`docker-compose.yml`**, the MLflow service uses Docker named volume **`mlflow-data`** mounted at **`/mlflow`** in the container. SQLite **`sqlite:///mlflow.db`** therefore lives **inside that mount** (not as **`./mlflow.db`** next to your repo on the host). **`MLFLOW_TRACKING_URI`** overrides the configured URI when set.

**Optional — Ridge baseline** (same experiment, different algorithm; no model registry promotion):

```bash
python -m src.training.train_baseline
```

Run after `dvc repro` has produced `data/splits/preprocessor.pkl` and splits.

**Then reproduce data artifacts (and optionally the full Docker stack)**

With **`.venv` activated** (so **`dvc`** / **`python`** point at installed deps):

```powershell
python -m dvc repro   # if `dvc` not on PATH
dvc push              # optional: only if you use the DVC remote
```

**Docker is optional until you need Bonus A (`docker compose up --build`).** Same commands on bash if `dvc` is on PATH.

`dvc repro` runs the full DVC pipeline locally. **`dvc push` copies outputs to the configured local DVC remote** so your cache matches the project layout (run it after `dvc repro` whenever you want to persist artifacts under the remote). If someone else has already populated the remote directory, you can run **`dvc pull`** before `dvc repro` to avoid rebuilding large artifacts.

**Bonus A & B (grading):** See **`## DDSC611 Bonus A & B — grading readiness`** below for pickles/Git, Docker checklist, Prefect UI evidence, and rubric snapshot.

**Prefect quick reminders:** `python flows/training_flow.py` (one-off); `python flows/training_flow.py serve` (cron worker); `prefect.yaml` + **`prefect deploy --prefect-file prefect.yaml`** after creating **`default-process-pool`** (details in that section).

**DVC storage:** The default remote `localremote` targets **`../dvc-storage`** (one directory **above** the repo — see `.dvc/config`; that folder is gitignored). Create `dvc-storage` there if missing, then run `dvc pull` / `dvc push` / `dvc repro` as needed. To point the remote at something else (for example `./dvc-storage` inside the repo):  
`dvc remote modify localremote url .\dvc-storage` (PowerShell) or adjust the URL to match your layout.

Check **API health** (with the stack up): **PowerShell** `Invoke-WebRequest http://localhost:8000/health -UseBasicParsing` or **bash** `curl http://localhost:8000/health`.

**Missing `data/raw/hour.csv`:** Only `hour.csv.dvc` is in Git, so **`dvc repro`** fails until the real file exists. Options: **`dvc pull`** (if your DVC remote/cache has it), or fetch from UCI:

- **Linux / macOS / GitHub Actions:** `python scripts/fetch_uci_hour_csv.py` (same MD5 check as the PowerShell script).
- **Windows PowerShell 5.x** (no `pwsh` required):

```powershell
cd <path-to-this-repo>
powershell -ExecutionPolicy Bypass -File .\scripts\fetch_uci_hour_csv.ps1
# or, if your session allows scripts:
.\scripts\fetch_uci_hour_csv.ps1
```

**`dvc repro` / `train`: MLflow connection refused (`localhost:5000`):** If **Docker** is installed, start MLflow in another terminal: `docker compose up mlflow`. If **`docker` is not recognized**, skip Docker and use a **file-backed** store (no server):

```powershell
$env:MLFLOW_TRACKING_URI = "file:./mlruns"
python -m dvc repro
```

**`dvc` not recognized:** Use the venv interpreter: **`.\.venv\Scripts\python.exe -m dvc repro`** (after `pip install -r requirements.txt`). Or activate first: **`.\.venv\Scripts\Activate.ps1`**, then **`dvc repro`**.

**`dvc` stages use the `python` on your PATH** — activate **`.venv`** *before* **`dvc repro`** so `python -m src...` uses the same deps as your venv.

**If you edited `dvc.yaml` deps** (for example adding a file under `train:`): run **`python -m dvc repro`** in that same venv and **commit `dvc.lock`** so reproducibility stays honest.

After **`dvc repro`** completes, **`data/splits/model.pkl`** exists and you can `git add data/splits/model.pkl data/splits/preprocessor.pkl`.

## Monitoring

- Start observability stack: `docker compose up --build` (API `:8000`, MLflow `:5000`, Prometheus `:9090`).
- Generate drift reports: `python -m monitoring.run_monitoring`.
- Scheduled drift job: `.github/workflows/monitoring-drift.yml` runs weekly (and on manual dispatch). It **downloads `hour.csv`**, restores a **cache** of `data/processed`, `data/splits`, and `.dvc/cache` (keyed by `dvc.lock` / `dvc.yaml`), runs **`python -m dvc repro` only on cache miss** (first run or after pipeline changes), then **`MLFLOW_TRACKING_URI=file:./mlruns`** for training without Docker.
- Read outputs:
  - `monitoring/evidently_reports/baseline.html`
  - `monitoring/evidently_reports/drift.html`
  - `monitoring/evidently_reports/drift_summary.json`
  - `monitoring/evidently_reports/interpretation.md` (decision policy/runbook)
- Alert meaning:
  - `drift_share_inputs_only > 0.20` indicates significant covariate drift.
  - Policy: investigate immediately (`P2`), retrain if the threshold is exceeded in 2 consecutive windows.
- Batch drift metric note:
  - `bike_feature_drift_psi` is generated by the batch monitoring run; use
    `monitoring/evidently_reports/drift_summary.json` as the canonical latest drift state.
  - In production, use Pushgateway/shared export if you need batch metrics to appear live on API `/metrics`.
- Serving endpoints:
  - `GET /health` liveness
  - `GET /ready` readiness (model/preprocessor loaded)
  - `GET /metrics` Prometheus scrape

## MLflow Dashboard & Visualization Interface

- Dashboard URL: `http://localhost:8000/dashboard` (also `http://localhost:8000/`)
- Prediction endpoints:
  - `POST /predict`
  - `POST /predict/batch`
- System endpoints:
  - `GET /health`
  - `GET /ready`
  - `GET /live`
  - `GET /metrics`
- Platform links:
  - MLflow: `http://localhost:5000`
  - Prometheus: `http://localhost:9090`

Included visualizations:

- Confidence gauge
- Normalized input feature bars (`temp`, `atemp`, `hum`, `windspeed`)
- Hourly scenario chart (predicted demand by `hr`)
- Batch prediction bar chart (+ min/max/avg stats)
- Recent predictions trend chart (runtime in-memory history)

MLflow-focused dashboard features:

- MLflow server status card and connectivity badge
- Experiment tracking overview endpoint integration
- Latest runs and model metrics API integration
- Model registry status/version integration
- Artifact listing endpoint integration

Crash-proofing highlights:

- Startup is resilient: if model loading fails, the API remains alive and dashboard still loads.
- Prediction endpoints return clean `503` when model/preprocessor is unavailable.
- Validation protects malformed payloads and out-of-range values with clean `422` responses.
- Batch predictions enforce a max of 100 records (`Batch size cannot exceed 100 records`).
- Drift summary and dashboard helper endpoints return safe JSON even when files are missing.
- See `docs/dashboard_crash_test_checklist.md` for live-demo robustness checks.

## Documentation

- `docs/technical_report.md` — pipeline evidence (includes DVC DAG figure)
- `docs/model_card.md` and `docs/data_card.md` — model and data details
- **MLflow run export for graders:** [`docs/mlflow/export.md`](docs/mlflow/export.md) — produces [`docs/experiment_log.csv`](docs/experiment_log.csv) after runs exist.

## Submission audit bundle (§6 rubric alignment)

Internal evidence used to chase **120/120** (course rubric):

- Single-command submission audit (includes docs/repro checks + docs contract tests for components 7/8 readiness):
  - `python scripts/verify_docs_repro.py`

- [`docs/plan/2026-05-05-complete-project-audit.md`](docs/plan/2026-05-05-complete-project-audit.md) — phased audit + rubric checklist plan
- [`docs/audits/2026-05-05-project-checklist.md`](docs/audits/2026-05-05-project-checklist.md) — actionable §6 rubric checkboxes + latest command outcomes
- [`docs/audits/2026-05-05-scorecard.md`](docs/audits/2026-05-05-scorecard.md) — numeric tracking
- [`docs/audits/2026-05-05-critical-issues.md`](docs/audits/2026-05-05-critical-issues.md) — blocking items before claiming full marks
- Code/config file inventory + auto sections: [`docs/audits/2026-05-05-file-by-file-audit.md`](docs/audits/2026-05-05-file-by-file-audit.md) (from [`docs/audits/2026-05-05-inventory-code.txt`](docs/audits/2026-05-05-inventory-code.txt))

### Screenshot ↔ rubric mapping

| Rubric area | Commit under `docs/screenshots/` |
|-------------|----------------------------------|
| CI/CD (coverage gate) | `pytest_coverage_report.png` |
| CI/CD (branch protection) | `branch_protection_main.png` |
| DVC deterministic repro | `dvc_repro_deterministic.png` |
| Clean install quickstart | `quickstart_clean_install.png` |
| Bonus A — compose / health | `docker_compose_ps_healthy.png`, `api_health_200.png` |
| Bonus B — Prefect success / failure | `prefect_flow_run_success.png`, `prefect_flow_run_failed_halt.png` |

## Verification Evidence (strict grading)

- Pytest + coverage (local strict venv): `docs/screenshots/pytest_coverage_report.png`
- Clean-install quickstart verification screenshot target: `docs/screenshots/quickstart_clean_install.png`
- DVC deterministic repro screenshot target: `docs/screenshots/dvc_repro_deterministic.png`
- Branch protection screenshot target: `docs/screenshots/branch_protection_main.png`

### Final evidence checklist

- Required grading screenshot names are listed in `docs/screenshots/README.md`.

---

## DDSC611 Bonus A & B — grading readiness

### 1) Bonus A — Git + pickles (clone must build Docker)

**What must be committed for “no undocumented steps” before `docker compose up --build`:**  
After `python -m dvc repro`, commit **`data/splits/model.pkl`** and **`data/splits/preprocessor.pkl`**. **`docker/api.Dockerfile`** copies both; if either file is missing, **`docker compose build`** fails.

**Nested `.gitignore` rule:** Patterns under **`data/splits/.gitignore`** apply *after* the repo root ignores. Putting **`/model.pkl`** here used to block **`git add`** even when root `.gitignore` had **`!data/splits/model.pkl`**. That entry is removed; **`data/splits/.gitignore`** now only carries comments — do not re-add ignores for those pickles here.

**Root `.gitignore` (confirmed):** `data/splits/*` plus **`!data/splits/model.pkl`** and **`!data/splits/preprocessor.pkl`**, and again **`!data/splits/model.pkl`** / **`!data/splits/preprocessor.pkl`** after **`*.pkl`**, so both files can be tracked.

**Verify before commit:**

```powershell
Test-Path .\data\splits\model.pkl, .\data\splits\preprocessor.pkl   # both True after dvc repro
git add -n data/splits/model.pkl data/splits/preprocessor.pkl       # both "add …" without -f
```

### 2) Bonus A — Docker checklist (compose + API image + grader commands)

**`docker-compose.yml` (verified in repo):**

| Item | Evidence |
|------|----------|
| ≥2 services + serving app | **`api`** (FastAPI), **`mlflow`**, **`prometheus`** |
| Compose wiring | **`api`** `depends_on` **`mlflow`** with **`condition: service_healthy`** |
| Inter-container MLflow URI | **`MLFLOW_TRACKING_URI: http://mlflow:5000`** — uses **service name**, not **`localhost`** between containers |
| Prometheus scrape | **`monitoring/prometheus/prometheus.yml`** → **`targets: ["api:8000"]`** |

**Using `localhost` is OK inside a container’s own `HEALTHCHECK`** (same network namespace). It must **not** be how **`api`** reaches **MLflow** from another container (here it correctly uses **`mlflow:5000`**).

**`docker/api.Dockerfile` (verified):** **`FROM python:3.11.9-slim`** (pinned), **`USER app`**, **`HEALTHCHECK`** on **`localhost:8000/health`** (intra-container), **`CMD`** uvicorn on **`0.0.0.0:8000`**.

**Exact commands — Windows (PowerShell, repo root):**

```powershell
docker compose build
docker compose up --build -d
Invoke-WebRequest -Uri http://localhost:8000/health -UseBasicParsing | Select-Object StatusCode, Content
```

**Linux / macOS (bash, repo root):**

```bash
docker compose build
docker compose up --build -d
curl -sS http://localhost:8000/health
```

**“Zero manual steps” tradeoff:** The **smallest** path that matches the rubric **without** a multi-stage Dockerfile that runs **`dvc repro`** inside Docker is **committing both pickles**. Alternatives (build stage that trains, or artifact download URL) add complexity and CI size; committing artifacts is intentional for graders who only run Compose.

### 3) Bonus B — Prefect + UI evidence + deploy

**Flow & tasks (`flows/training_flow.py`):** Five Prefect **`@task`**s — **`validate_data`**, **`preprocess`**, **`train`**, **`evaluate`**, **`register_model`** — call real code (`load_raw`, `build_splits`, `fit_preprocessor`, **`train_main`**, **`promote_version_to_production`**, etc.), not placeholders.

**Schedulable in-repo:** Root **`prefect.yaml`** deployment **`bike-share-training`** — **`cron: "0 3 * * *"`**, **`timezone: UTC`**. Also **`python flows/training_flow.py serve`** invokes **`training_flow.serve(..., cron="0 3 * * *", timezone="UTC")`**.

**One-time work pool + deploy (CLI):**

```powershell
prefect work-pool create default-process-pool --type process
cd <path-to-repo>
prefect deploy --prefect-file prefect.yaml
```

Start UI / API: **`prefect server start`** (separate terminal), open the Prefect dashboard, trigger or watch scheduled runs.

**Screenshots are not gitignored** (root `.gitignore` has no `*.png` / `docs/screenshots` rule) — add them and `git add docs/screenshots/*.png`.

**Recommended screenshot filenames (commit under `docs/screenshots/`):**

| File | Purpose |
|------|---------|
| `docker_compose_ps_healthy.png` | Containers up / healthy |
| `api_health_200.png` | Browser or terminal showing **`GET /health`** success |
| `prefect_flow_run_success.png` | Prefect UI: full DAG green / completed **`register_model`** |
| `prefect_flow_run_failed_halt.png` | Prefect UI: upstream task failed; **`evaluate`** / **`register_model`** skipped or marked failed |

**How to capture a deliberate failure:** Temporarily raise **`configs/params.yaml`** → **`validation.min_test_r2`** above your measured test R² (e.g. **0.99**), run **`python flows/training_flow.py`**, **`evaluate`** should **`raise ValueError`** and downstream **`register_model`** should not succeed. Revert **`min_test_r2`** afterward.

### 4) Rubric snapshot (repo evidence only — run Docker/Prefect to prove live)

| Criterion | Verdict | Evidence |
|-----------|---------|----------|
| **A:** Serving Dockerfile, best practices (pin / non-root / minimal-ish) | **Pass** | `docker/api.Dockerfile` |
| **A:** `docker-compose.yml`, serving + MLflow or monitoring | **Pass** | `docker-compose.yml` + Prometheus |
| **A:** `/health` implemented | **Pass** | `src/serving/app.py` |
| **A:** Inter-service hostnames (`mlflow`, `api`) | **Pass** | compose env + `prometheus.yml` |
| **A:** **`docker compose up --build`** with zero extra steps | **Partial** | Requires **`model.pkl`** + **`preprocessor.pkl`** in tree (commit after `dvc repro`) |
| **A:** Build/run + `/health` live | **Insufficient data** | Needs machine with Docker + your run logs/screenshots |
| **B:** Airflow DAG *or* Prefect flow, ≥5 tasks, named pipeline | **Pass** | `flows/training_flow.py` |
| **B:** Full pipeline (real `train_main`, registry) | **Pass** | imports + calls in same file |
| **B:** Schedulable | **Pass** | `prefect.yaml` + `training_flow.py serve` |
| **B:** UI run + failure evidence | **Insufficient data** unless PNGs committed | Add files under **`docs/screenshots/`** |

### Suggested submission commit message

```
docs: DDSC611 bonus grading — README pickles/compose/Prefect evidence; splits .gitignore safe for Docker

chore: add serving artifacts for Bonus A Docker clone build (model.pkl, preprocessor.pkl)
```

Use the second line only when you **`git add`** the pickles and are ready for that commit.
