# MLOps Final Project — Bike Sharing Demand

End-to-end MLOps pipeline for predicting hourly bike rental counts (UCI Bike Sharing dataset).

## Quickstart

Use a virtual environment (recommended). From the repository root:

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**MLflow tracking server (Docker)** — point training at it via `MLFLOW_TRACKING_URI` (Compose publishes **`5001`** on the host):

```bash
docker compose up -d mlflow
```

UI (Compose maps host **`5001`** → container `5000`): [http://localhost:5001](http://localhost:5001). Artifacts live under the **`mlflow-data`** Docker volume (see `docker-compose.yml`). Use **`MLFLOW_TRACKING_URI`** to point training at this server.

**Optional — Ridge baseline** (same experiment, different algorithm; no model registry promotion):

```bash
python -m src.training.train_baseline
```

Run after `dvc repro` has produced `data/splits/preprocessor.pkl` and splits.

**Then reproduce data artifacts and run the API/monitoring stack**

```bash
dvc repro
dvc push
docker compose up --build
```

`dvc repro` runs the full DVC pipeline locally. **`dvc push` copies outputs to the configured local DVC remote** so your cache matches the project layout (run it after `dvc repro` whenever you want to persist artifacts under the remote). If someone else has already populated the remote directory, you can run **`dvc pull`** before `dvc repro` to avoid rebuilding large artifacts.

**DVC storage:** The default remote `localremote` targets **`dvc-storage/` at the repository root** (see `.dvc/config`; path is gitignored). On a fresh clone the folder may be missing—create it if needed (`mkdir dvc-storage` on Unix, `mkdir dvc-storage` in PowerShell), then run `dvc repro` and `dvc push`. To use another directory:  
`dvc remote modify localremote url <absolute-or-relative-path>`.

Then open `curl http://localhost:8000/health`.

## Production MLflow registry (serving)

Docker Compose enables **`PRODUCTION_STRICT=true`** on the API: **only** the MLflow **Production** registry model is allowed (no local pickle fallback), and any startup failure loading model **or** preprocessor **exits the process** (no degraded “empty” API).

**Bring-up order (required):**

1. Start MLflow and wait until healthy: `docker compose up -d mlflow`
2. **Seed Production** once (host maps MLflow to **`5001`**):

   `MLFLOW_TRACKING_URI=http://127.0.0.1:5001 PYTHONPATH=. python scripts/seed_mlflow_production.py`

   Uses experiment **`bike_sharing`** from `configs/params.yaml` unless you pass `--experiment`.

3. Start the rest: `docker compose up -d api prometheus`

Shared **`mlflow-data`** volume is mounted on **both** `mlflow` and `api` so registry artifact paths resolve inside the API container.

**Outside Docker**, unset **`PRODUCTION_STRICT`** for local development if you need pickle fallback; **`REQUIRE_REGISTRY_MODEL`** alone still forbids fallback but historically allowed a running app without a model when load threw elsewhere — prefer **`PRODUCTION_STRICT`** for “fail closed.” **`SERVE_USE_LOCAL_MODEL_ONLY`** is incompatible with strict production.

**Observability:** **`GET /api/model-info`** (`load_source`, `fallback_reason`), **`GET /api/dashboard-summary`** (`model_load`), Prometheus **`bike_model_registry_load_satisfied`** (1 = registry Production).

## Monitoring

- Start observability stack: `docker compose up --build` (API `:8000`, MLflow UI **`5001`** on localhost, Prometheus `:9090`).
- Generate drift reports: `python -m monitoring.run_monitoring`.
- Required monitoring artifacts must exist before running:
  - `data/splits/model.pkl`
  - `data/splits/preprocessor.pkl`
  - `data/splits/reference.parquet`
  - `data/splits/production.parquet`
  - Prepare them with `dvc pull` (preferred) or `dvc repro`.
- Operational drift alerting uses real production data from `data/splits/production.parquet`.
- Synthetic perturbation is demo-only and written to
  `monitoring/evidently_reports/drift_synthetic_demo.html`.
- Default configuration is operational-only (`generate_synthetic_demo_report: false`); enable demo
  explicitly when needed.
- Scheduled drift job: `.github/workflows/monitoring-drift.yml` runs weekly (and on manual dispatch).
- The drift workflow is fail-loud: missing artifacts or runtime errors fail the job while still
  uploading diagnostics (`drift_summary.json`, reports, runbook) as artifacts.
- In GitHub Actions, missing artifacts are pulled from DVC when `DVC_REMOTE_URL` is configured as
  a repository secret; otherwise jobs fail with explicit remediation messages.
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

## CI parity (test like production locally)

GitHub Actions runs **lint → pytest + coverage (≥70%) → Pandera data tests → MLflow Production `validate_model.py`**.

From the repo root (PowerShell):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_full_ci_local.ps1
```

By default this matches CI end-to-end: it checks required artifacts (including **`data/raw/hour.csv`**), starts a local MLflow server on **`127.0.0.1:5000`** with the same SQLite layout as the **`model-validation`** job, runs **`scripts/seed_mlflow_production.py`**, then **`scripts/validate_model.py`** with **`REQUIRE_LOCAL_MODEL_ARTIFACTS=1`**.

If you already run MLflow elsewhere (e.g. Docker on **`5001`**), set **`MLFLOW_TRACKING_URI`** before the script so step 5 seeds and validates against that server instead of spawning port **5000**.

The script clears **`SKIP_MLFLOW_REGISTRY`** for the validation step so a developer shell cannot accidentally run a partial gate (full load of **`models:/…/Production`** is always exercised).

For a faster loop without the registry gate (not full parity): **`$env:SKIP_MODEL_VALIDATION='1'`** before the script.

## Runtime Hardening Defaults

- Docker services now bind to `127.0.0.1` by default (local-only exposure for safety).
- `restart: unless-stopped` is enabled for `mlflow`, `api`, and `prometheus`.
- Prometheus data is persisted in a named volume `prometheus-data`.

## Documentation

- `docs/technical_report.md` — pipeline evidence (includes DVC DAG figure)
- `docs/model_card.md` and `docs/data_card.md` — model and data details
- `docs/github-ci-gate.md` — GitHub Actions secrets, DVC remote for CI, branch protection, team workflow

## Verification Evidence (strict grading)

- Pytest + coverage (local strict venv): `docs/screenshots/pytest_coverage_report.png`
- Clean-install quickstart verification screenshot target: `docs/screenshots/quickstart_clean_install.png`
- DVC deterministic repro screenshot target: `docs/screenshots/dvc_repro_deterministic.png`
- Branch protection screenshot target: `docs/screenshots/branch_protection_main.png`
