# Project checklist — DDSC611 §6.1 rubric (target 120/120)

**Legend:** `[x]` verified in audit session on 2026-05-05 · `[ ]` needs you (data/MLflow/Docker/UI)

**Automated sweep (audit session)**

| Command | Result |
|---------|--------|
| `python -m ruff check src/ tests/` | PASS |
| `python -m pytest tests/ --cov=src --cov-fail-under=70` | PASS — total coverage **≈81%** (`src/` subset; see CI for canonical run) |
| `python -m dvc dag` | **FAIL** on host Python due to **`pathspec` / `_DIR_MARK`** — use **project venv** + `requirements.txt` (see `README.md`). Re-run locally and tick §1 below. |

---

## §6.1 Rubric component 1 — Data versioning (10)

- [ ] `python -m dvc repro` (or `dvc repro`) completes without error (**requires** `data/raw/hour.csv` — see README fetch / `dvc pull`)
- [x] Three+ distinct reproducible artifacts defined in pipeline (example names for graders): `data/processed/bike_clean.parquet`, `data/splits/preprocessor.pkl`, `data/splits/model.pkl` — stages in `dvc.yaml`
- [x] Randomness / deterministic story documented — `configs/params.yaml` (`data.random_state`, etc.)
- [x] DVC remote configured & documented — `.dvc/config` → `localremote`, `README.md` § “DVC storage” / `../dvc-storage`

## §6.1 Rubric component 2 — Preprocessing pipeline (12)

- [x] sklearn `Pipeline` serialized to `data/splits/preprocessor.pkl` — stage `featurize` in `dvc.yaml`
- [x] Parameters driven from `configs/params.yaml` — inspect `src/features/preprocessor.py` + config keys
- [x] DVC-tracked output path referenced in pipeline
- [x] ≥3 focused unit tests (spot check):  
  `pytest tests/unit/test_preprocessor.py tests/unit/test_featurize_main.py tests/unit/test_prepare_main.py -q`

## §6.1 Rubric component 3 — Experiments & registry (15)

- [ ] MLflow: ≥3 distinguishable logged runs (**capture** screenshots or regenerate `python scripts/export_runs.py` → `docs/experiment_log.csv`; see [`docs/mlflow/export.md`](../mlflow/export.md))
- [x] HPO module present (`src/training/hpo.py`) wired from training (`src/training/train.py`)
- [x] Registry promotion API path implemented (`src/training/registry.py`, used from training / Prefect flow)
- **Note:** CI sets `SKIP_MLFLOW_REGISTRY: "1"` for `model-validation` — local or Docker MLflow needed for live registry demo.

## §6.1 Rubric component 4 — Serving (15)

- [x] `/health` + `/predict` (+ `/ready`, `/metrics`) — `src/serving/app.py`
- [x] Pydantic validation — `src/serving/schemas.py`
- [x] Tests: `pytest tests/test_api.py tests/unit/test_serving.py -q` (included in full suite GREEN)

## §6.1 Rubric component 5 — CI/CD (13)

- [x] Four jobs in `.github/workflows/ci.yml`: `lint`, `test`, `data-validation`, `model-validation`
- [x] Coverage gate `--cov-fail-under=70` in CI
- [ ] Branch protection screenshot committed as `docs/screenshots/branch_protection_main.png`
- [ ] Green CI screenshot (**Actions UI** PNG or **`docs/screenshots/pytest_coverage_report.png`** as local adjunct — align with grader wording)

## §6.1 Rubric component 6 — Monitoring & drift (15)

- [x] Two Evidently HTML artifacts in repo paths: `monitoring/evidently_reports/baseline.html`, `drift.html`
- [x] Drift threshold logic + interpretation — `monitoring/drift_logic.py`, `monitoring/evidently_reports/interpretation.md`, `README.md` Monitoring section
- [x] Prometheus scrape target `api:8000` — `monitoring/prometheus/prometheus.yml` ↔ `docker-compose.yml`

## §6.1 Rubric component 7 — Documentation (10)

- [x] README Quickstart (venv → install → DVC/train notes)
- [x] Model card — `docs/model_card.md`
- [x] Data card — `docs/data_card.md`
- [ ] MLflow export artifact committed (**after runs exist**) — `python scripts/export_runs.py` → `docs/experiment_log.csv`; procedure in [`docs/mlflow/export.md`](../mlflow/export.md)

## §6.1 Rubric component 8 — Reproducibility (10)

- [x] Pinned deps — `requirements.txt`
- [x] Tunables centralized — `configs/params.yaml`
- [x] `.gitignore` policy documented for pickles (`README.md` Bonus A section)

---

## Bonus A — Docker (+10)

- [ ] `docker compose build api` succeeds on submitter machine
- [ ] `docker compose up` → `Invoke-WebRequest http://localhost:8000/health` or `curl` — capture `docs/screenshots/api_health_200.png`

## Bonus B — Orchestration (+10)

- [x] ≥5 Prefect tasks — `flows/training_flow.py`
- [x] Schedulable — `prefect.yaml` + `python flows/training_flow.py serve`
- [ ] Failure-handling screenshot — `docs/screenshots/prefect_flow_run_failed_halt.png` (**README** documents how via `validation.min_test_r2`)
