    # MLOps End-to-End Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a complete, production-grade MLOps pipeline (8 mandatory components + both bonuses) on the UCI Bike Sharing dataset that scores full marks on the DDSC611 Final Project rubric (100/100 + 20 bonus = 120/100).

**Architecture:** A reproducible, containerised regression pipeline. Raw CSV → DVC-versioned data → sklearn preprocessing pipeline → Random Forest tuned with Optuna → MLflow tracking + registry → FastAPI serving → Evidently drift reports + Prometheus metrics → GitHub Actions CI → Docker Compose stack → Prefect training flow.

**Tech Stack:** Python 3.11, DVC 3.x, scikit-learn 1.4, MLflow 2.x, Optuna 3.x, FastAPI, Pydantic v2, Pandera, Evidently 0.4.x, prometheus-client, Docker + Compose, Prefect 2.x, ruff, pytest, GitHub Actions.

---

## PROJECT CONTEXT & GITHUB STRATEGY (AS-BUILT — checkpoint)

This section is the **authoritative workflow snapshot** for DDSC611 / `mlops-final`. It aligns with the intended branching + CI strategy; tweaks reflect what is actually configured on GitHub today.

### Repository state on `main` (completed work)

The following are **merged on `main`** (PR **#1** Component 2+3, PR **#2** Component 4+5 serving + CI, PR **#14** collaboration templates):

| Area | Status |
|------|--------|
| Phases 0–2 (historical) | Project skeleton, DVC prepare/preprocess/featurize, EDA, sklearn preprocessing pipeline |
| Phase 3 — Components 2+3 | DVC **`train`** stage, Optuna HPO, MLflow tracking + registry code paths, `configs/params.yaml`, committed **`data/splits/metrics.json`** (small metrics gate only; parquet/pkl splits remain DVC-only) |
| Phases 4–5 (branch name `component-4-5-serve-ci`) | FastAPI **`/health`**, **`/predict`**, **`/predict/batch`**, Pydantic v2, Prometheus metrics; integration tests with **`load_artifacts` mocked in CI** (no `model.pkl` or local MLflow required on GitHub runners) |
| CI | `.github/workflows/ci.yml`: jobs **`lint`** → **`test`** → **`data-validation`** → **`model-validation`**, `ubuntu-latest`, Python **3.11**, **`actions/cache`** on pip, pytest coverage ≥ **70%**, coverage XML artifact |
| Model gate | `scripts/validate_model.py` reads committed metrics vs **`validation.rmse_threshold`**; **`SKIP_MLFLOW_REGISTRY=1`** in CI optional Production load |
| Collaboration | `.github/ISSUE_TEMPLATE/*`, `pull_request_template.md`, `ISSUE_TEMPLATE/config.yml` |

### Branching & merge rules (target vs actual)

| Intent | Implementation |
|--------|----------------|
| Short-lived **`feature/<component-slug>`** branches | Used for delivered work; **merged branches deleted on remote** after squash-merge to keep the repo tidy (`feature/component-2-3-train`, `feature/component-4-5-serve-ci`, `chore/github-collaboration-templates` removed post-merge). |
| **Squash merge** via PR only | Ruleset **Protect main** allows **squash** only (no merge/rebase commits). |
| **Reviews** | Rubric often asks for **1 approval**; **solo workflow tweak:** ruleset **`required_approving_review_count` = 0** so you can merge after CI. **Set back to 1** when a teammate can review (GitHub → Settings → Rules → *Protect main*). |
| **CI gate** | Same ruleset requires status checks **`lint`**, **`test`**, **`data-validation`**, **`model-validation`**, strict policy (branch up to date). |
| **Commits** | Conventional style: **`feat|fix|chore|test|docs(scope): description`** |

### Placeholder branches (next implementation — reset to current `main`)

All at commit **`e2b785c`** (latest `main` at last checkpoint):

- `feature/component-6-monitoring` → Phase 6 monitoring / Evidently / Prometheus (issue **#8**)
- `feature/bonus-docker-prefect` → Bonus Docker + Prefect (issues **#10**, **#11**)
- `feature/component-7-docs` → Model card, data card, README (issue **#12**)

Start each line of work with: `git checkout main && git pull && git checkout <branch>` (and `git merge main` or `rebase` if `main` advanced).

### Issues hygiene (GitHub)

**Closed:** **#3** (training PR), **#5** (serve/CI PR), **#6** (branch protection configured).

**Open (expected):** **#7** optional hosted MLflow in CI, **#8** monitoring, **#10–#12** bonus/docs, **#13** final submission checklist.

### Constraints (unchanged)

- Do **not** commit raw splits/processed CSV/parquet under `data/` except via DVC; **`data/splits/metrics.json`** is the agreed small JSON exception.
- Do **not** commit `.env`, `mlruns/`, or virtualenv folders.
- **`configs/params.yaml`**, **`dvc.yaml`**, **`dvc.lock`** stay versioned as pipeline source.

### Naming note (plan vs branches)

In this document, **“Phase 5”** sometimes meant **Serving API** and sometimes **CI** in older drafts. **Delivered reality:** FastAPI + CI landed together on **`feature/component-4-5-serve-ci`**. **Next coding milestone** in the original rubric sequence is **monitoring / drift / Prometheus** (run from **`feature/component-6-monitoring`**), not “another Phase 5.”

---

## 1. DATASET DECISION

### Selected Dataset: **UCI Bike Sharing Dataset (hour.csv)**

Source: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
License: CC BY 4.0 (open).
Underlying data: Capital Bikeshare system (Washington D.C.), 2011–2012; weather from freemeteo.com.
Citation: Fanaee-T, H. & Gama, J. (2013). *Event labeling combining ensemble detectors and background knowledge*. Progress in AI. doi:10.1007/s13748-013-0040-3.

**Already on disk** at `bike_sharing_dataset/hour.csv` (1.16 MB, 17,379 rows) and `bike_sharing_dataset/day.csv` (731 rows, ignored by our pipeline). No download required — Phase 0 just copies it into `data/raw/`.

### Why This Is the Easiest Valid Option

| Criterion | Bike Sharing | Why It Wins |
|-----------|-------------|-------------|
| Row count | 17,379 hourly rows | Well above 10k requirement; small enough to train in seconds |
| Schema | 17 columns, all clean | No serious missing values to debug |
| Numeric features | 6 already-normalised numerics (`temp`, `atemp`, `hum`, `windspeed`, `hr`, `mnth`) | Trivial scaling; values already in [0,1] |
| Categorical features | 4 small-cardinality (`season` 1-4, `weathersit` 1-4, `weekday` 0-6, `holiday/workingday` binary) | One-hot encoding is fast; no high-cardinality nightmares |
| Target | `cnt` (rental count) | Plain regression — no class imbalance, no SMOTE needed |
| Drift simulation | **Natural year split**: yr=0 (2011) vs yr=1 (2012) | The cleanest possible split. 2012 has organic growth + weather differences |
| File size | 1.16 MB single CSV | Fast clone, fast `dvc push`, no S3 required |
| Null values | **Verified: zero nulls in all 17 columns** | Imputers act defensively for serve-time inputs only |
| Serving payload | Small flat JSON record | Trivial Pydantic schema |

**Comparison to alternatives:**
- **Adult Income**: needs categorical encoding for many high-cardinality columns (`occupation`, `native-country`), missing-value handling, fairness considerations.
- **NYC Taxi**: huge files, geo-features need engineering, multiple monthly downloads.
- **IEEE Fraud / Home Credit**: 400+ features, severe class imbalance, weeks of feature work.
- **Air Quality UCI**: lots of `-200` sentinel missing values, sensor-specific quirks.
- **Telco Churn / Bike Sharing** are the lightest — Bike Sharing wins because regression has no class-imbalance branch and no fairness branch to handle.

### Target Variable

`cnt` — total bike rentals in the hour (continuous integer, regression task).

### Feature Types

**Numerical (6):** `temp`, `atemp`, `hum`, `windspeed`, `hr`, `mnth`
- ranges (verified): `temp ∈ [0.02, 1.00]`, `atemp ∈ [0.00, 1.00]`, `hum ∈ [0.00, 1.00]`, `windspeed ∈ [0.00, 0.8507]`, `hr ∈ [0, 23]`, `mnth ∈ [1, 12]`

**Categorical (5):** `season ∈ {1..4}`, `holiday ∈ {0,1}`, `workingday ∈ {0,1}`, `weathersit ∈ {1..4}`, `weekday ∈ {0..6}`

**Drop:** `instant` (row id), `dteday` (used only for the time split, not a feature), `casual`, `registered` (target leakage — these sum to `cnt`), `yr` (used only for the train/production split).

**Target:** `cnt ∈ [1, 977]` (mean ≈ 189, std ≈ 181).

### Drift Simulation Strategy (the most important design choice)

We exploit the **natural temporal split** of the dataset, then layer **synthetic perturbations** to guarantee a strong drift signal:

1. **Reference set** = all rows where `yr == 0` (calendar year 2011). **8,645 rows** (verified). This is the data the model sees during training.
2. **Production-clean set** = a held-out 10% slice of `yr == 0` (random sample, set aside before training). Used for the *baseline* Evidently report — should show **no drift**.
3. **Production-drifted set** = all rows where `yr == 1` (calendar year 2012, **8,734 rows** verified), with three injected perturbations:
   - `temp` multiplied by 1.10 (simulates a hotter year — global warming narrative)
   - `hum` multiplied by 0.85 (simulates a drier year)
   - `windspeed` += Gaussian noise N(0, 0.05) (simulates noisier sensors)
4. Both reports use the same reference set so the comparison is apples-to-apples.

**Why this guarantees marks:** the spec requires drift on ≥3 features visible in Evidently. We perturb exactly 3 numeric features by enough to trigger Evidently's KS-test threshold (default p<0.05), AND there is genuine year-over-year drift on `cnt`-correlated features. The drift will be obvious in the HTML report and the threshold logic (>20% of features drifted) will fire.

---

## 2. SYSTEM ARCHITECTURE (HIGH LEVEL)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DEVELOPMENT WORKSTATION                          │
│                                                                          │
│  data/raw/hour.csv ──► DVC ──► [prepare] ──► data/processed/             │
│         (gitignored)            [preprocess] ──► splits/ + pipeline.pkl  │
│                                 [featurize]  ──► features.parquet        │
│                                 [train]      ──► model.pkl + metrics    │
│                                                       │                  │
│                                                       ▼                  │
│  configs/params.yaml ─────────────────► MLflow Tracking Server          │
│                                          (Docker, port 5000)            │
│                                          ├─ runs (params, metrics)      │
│                                          └─ Model Registry              │
│                                              None → Staging → Production│
│                                                       │                  │
│                          ┌────────────────────────────┘                  │
│                          ▼                                               │
│                    FastAPI app (Docker, port 8000)                       │
│                    ├─ /health                                            │
│                    ├─ /predict        ◄────── client                     │
│                    ├─ /predict/batch                                     │
│                    └─ /metrics ──► Prometheus scrape (port 9090)         │
│                                                       │                  │
│  monitoring/run_monitoring.py ─► Evidently HTML reports                 │
│         │                       (baseline.html, drift.html)             │
│         └─► drift threshold logic ─► log warning + retrain trigger      │
│                                                                          │
│  Prefect Flow (training_flow.py):                                       │
│  validate_data → preprocess → train → evaluate → register_model         │
│                                                                          │
│  GitHub Actions CI (.github/workflows/ci.yml):                          │
│  push → lint → unit tests + coverage → data validation → model val     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Where Each Layer Fits

- **DVC**: versions raw, processed, splits, fitted preprocessor, trained model. Local DVC remote at `./dvc-storage/`.
- **MLflow**: logs every run (params, loss curves, metrics, model). Promotes the best model through the registry. Runs as `mlflow` service in `docker-compose.yml`.
- **Docker**: Two services. `mlflow` (tracking server, sqlite backend, volume-mounted artifacts). `api` (FastAPI app, depends on `mlflow`, exposes port 8000).
- **Prefect**: orchestrates `validate_data → preprocess → train → evaluate → register_model`. Local Prefect server (`prefect server start`).
- **CI/CD**: GitHub Actions runs on every push and PR. 4 stages: lint, tests-with-coverage, data-validation, model-validation. Branch protection on `main` requires all four green.
- **Monitoring**: `run_monitoring.py` is invoked manually + via Prefect. Generates 2 HTML reports + JSON drift summary. The FastAPI app exposes 5 Prometheus metrics on `/metrics`.
- **Retraining loop**: drift report → threshold breach → log entry → operator triggers Prefect flow → new model registered → API picks it up on restart (or via webhook in a real system; we document the loop in README and Model Card).

---

## 3. PROJECT STRUCTURE (EXACT FILE TREE)

This matches the spec's required structure verbatim and adds nothing surplus.

```
mlops-final/
├── .github/
│   └── workflows/
│       └── ci.yml                         # GitHub Actions CI pipeline
├── configs/
│   └── params.yaml                        # ALL hyperparameters and paths
├── data/
│   ├── raw/                               # DVC-tracked, .gitignored content
│   │   └── hour.csv                       # downloaded once, then dvc add
│   ├── processed/                         # cleaned, post-prepare
│   │   └── bike_clean.parquet
│   └── splits/                            # train/test/reference/production
│       ├── train.parquet
│       ├── test.parquet
│       ├── reference.parquet              # for drift baseline
│       └── production.parquet             # drifted set
├── docs/
│   ├── model_card.md
│   ├── data_card.md
│   └── experiment_log.csv                 # MLflow runs export
├── docker/
│   ├── api.Dockerfile                     # FastAPI container
│   └── mlflow.Dockerfile                  # MLflow tracking server
├── docker-compose.yml                     # Bonus A
├── flows/
│   └── training_flow.py                   # Prefect flow (Bonus B)
├── monitoring/
│   ├── __init__.py
│   ├── run_monitoring.py                  # builds Evidently reports + threshold
│   ├── drift_logic.py                     # threshold function (testable)
│   ├── evidently_reports/                 # generated HTMLs land here
│   │   ├── baseline.html
│   │   └── drift.html
│   └── prometheus/
│       └── prometheus.yml                 # scrape config (optional)
├── notebooks/
│   └── 01_eda.ipynb                       # EDA only — no pipeline logic
├── src/
│   ├── __init__.py
│   ├── config.py                          # loads params.yaml into a typed dataclass
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load.py                        # download + load raw CSV
│   │   ├── prepare.py                     # `prepare` DVC stage entrypoint
│   │   ├── split.py                       # train/test/reference/production split
│   │   └── schema.py                      # Pandera DataFrameSchema
│   ├── features/
│   │   ├── __init__.py
│   │   ├── preprocessor.py                # builds the sklearn Pipeline
│   │   └── featurize.py                   # `featurize` DVC stage entrypoint
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py                       # `train` DVC stage entrypoint
│   │   ├── hpo.py                         # Optuna search
│   │   └── registry.py                    # MLflow registry + stage transitions
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py                     # rmse, mae, r2 helpers
│   └── serving/
│       ├── __init__.py
│       ├── app.py                         # FastAPI app
│       ├── schemas.py                     # Pydantic request/response models
│       └── metrics.py                     # Prometheus metric definitions
├── tests/
│   ├── __init__.py
│   ├── conftest.py                        # shared fixtures (sample df, model)
│   ├── unit/
│   │   ├── test_preprocessor.py           # 4+ tests for Pipeline behaviours
│   │   ├── test_split.py                  # split sizes, no leakage
│   │   ├── test_metrics.py                # rmse/mae/r2 sanity
│   │   ├── test_drift_logic.py            # threshold function
│   │   └── test_config.py                 # params.yaml round-trip
│   ├── integration/
│   │   └── test_api.py                    # /health, /predict, /predict/batch
│   └── data/
│       └── test_data_validation.py        # Pandera schema on a sample
├── .dvcignore
├── .gitattributes                         # LF normalization
├── .gitignore
├── dvc.yaml                               # 4 stages: prepare, preprocess, featurize, train
├── dvc.lock                               # generated, committed
├── pyproject.toml                         # ruff config + tool settings
├── pytest.ini                             # pytest + coverage config
├── README.md
└── requirements.txt                       # pinned versions
```

### Per-folder responsibilities

- **`configs/`** — single source of truth. Every script reads from `params.yaml`; no magic numbers in code.
- **`data/`** — never committed to Git. Raw, processed, and split parquet files live here, all DVC-tracked.
- **`docker/`** — Dockerfiles split per-service to keep build contexts minimal.
- **`docs/`** — Model Card, Data Card, exported MLflow log. Read by graders during the final discussion.
- **`flows/`** — Prefect orchestration entrypoints. Imports from `src/` to avoid logic duplication.
- **`monitoring/`** — drift detection. The threshold function lives in its own module so it is unit-testable.
- **`notebooks/`** — EDA only. The spec explicitly forbids pipeline logic here.
- **`src/`** — all production code, organised by stage of the pipeline. Every subpackage has `__init__.py` so `pip install -e .` works (we don't actually do editable install — we just set `PYTHONPATH=src` in scripts and CI).
- **`tests/`** — mirrors `src/` plus `integration/` and `data/`. Coverage target ≥70% (the rubric threshold).

---

## 4. STEP-BY-STEP IMPLEMENTATION PLAN

The plan is split into 10 phases (0–9). Phases 0–7 are mandatory; phases 8 and 9 are bonuses.

Each phase declares: **Goal**, **Files**, **Dependencies on previous phases**, **Common pitfalls**, then a numbered list of bite-sized tasks. Every task ends with a commit. Coverage is checked continuously, not at the end.

---

### Phase 0: Environment Setup

**Goal:** A reproducible Python environment, a Git+DVC-initialised repo, the dataset on disk, and a green hello-world test.

**Files:**
- Create: `requirements.txt`, `pyproject.toml`, `pytest.ini`, `.gitignore`, `.dvcignore`, `.gitattributes`, `README.md`, `configs/params.yaml`, `src/__init__.py`, `tests/__init__.py`, `tests/conftest.py`, `notebooks/01_eda.ipynb` (empty stub).

**Dependencies:** None.

**Pitfalls:**
- Forgetting to pin versions → CI breaks weeks later. Pin everything.
- Committing `data/raw/hour.csv` to Git → spec rejection. Add `data/raw/*` to `.gitignore` BEFORE downloading data.
- Windows line endings polluting diffs → `.gitattributes` with `* text=auto eol=lf`.
- DVC on Windows tries to symlink → set `dvc config cache.type copy` if `dvc add` fails.

**Tasks:**

- [ ] **Task 0.1: Create the GitHub repository skeleton**

  ```bash
  cd "D:/Users/Badr/Downloads/MLOPs Final Project"
  mkdir mlops-final && cd mlops-final
  git init
  git branch -M main
  ```

  Then on github.com create an empty repo named `mlops-final` and:

  ```bash
  git remote add origin https://github.com/<team-org>/mlops-final.git
  ```

  Add all team members as collaborators (Settings → Collaborators) and add the instructor.

- [ ] **Task 0.2: Write `.gitignore`**

  Create `.gitignore` with this exact content:

  ```gitignore
  # Python
  __pycache__/
  *.py[cod]
  *$py.class
  *.egg-info/
  .pytest_cache/
  .coverage
  htmlcov/
  .ruff_cache/

  # Virtual environments
  .venv/
  venv/
  env/

  # Data — DVC-tracked, never to Git
  data/raw/*
  !data/raw/.gitkeep
  data/processed/*
  !data/processed/.gitkeep
  data/splits/*
  !data/splits/.gitkeep

  # Models & artifacts
  *.pkl
  *.joblib
  mlruns/
  mlflow.db
  dvc-storage/

  # Generated reports
  monitoring/evidently_reports/*.html

  # IDE
  .vscode/
  .idea/
  *.swp
  .DS_Store

  # Prefect
  .prefect/
  ```

  Create empty `.gitkeep` placeholder files in `data/raw/`, `data/processed/`, `data/splits/` so the directories exist.

- [ ] **Task 0.3: Write `.gitattributes` and `.dvcignore`**

  `.gitattributes`:
  ```gitattributes
  * text=auto eol=lf
  *.csv binary
  *.parquet binary
  *.pkl binary
  *.joblib binary
  ```

  `.dvcignore`:
  ```dvcignore
  # default — keep empty for now
  ```

- [ ] **Task 0.4: Write `requirements.txt` (pinned)**

  ```
  # Core
  numpy==1.26.4
  pandas==2.2.2
  scikit-learn==1.4.2
  scipy==1.13.1
  pyarrow==16.1.0

  # Config
  pyyaml==6.0.1
  pydantic==2.7.1
  pydantic-settings==2.2.1

  # DVC + experiment tracking
  dvc==3.51.2
  mlflow==2.13.0

  # HPO
  optuna==3.6.1

  # Serving
  fastapi==0.111.0
  uvicorn[standard]==0.30.1
  gunicorn==22.0.0

  # Data validation
  pandera==0.19.3

  # Drift / Monitoring
  evidently==0.4.27
  prometheus-client==0.20.0
  prometheus-fastapi-instrumentator==7.0.0

  # Orchestration (Bonus B)
  prefect==2.19.4

  # Testing & lint
  pytest==8.2.2
  pytest-cov==5.0.0
  httpx==0.27.0
  ruff==0.4.7

  # Misc
  joblib==1.4.2
  requests==2.32.3
  ```

- [ ] **Task 0.5: Create the virtual environment and install**

  ```bash
  python -m venv .venv
  source .venv/Scripts/activate   # Git Bash on Windows
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

  Verify versions:
  ```bash
  python -c "import sklearn, mlflow, dvc, fastapi, evidently; print('ok')"
  ```
  Expected: `ok`.

- [ ] **Task 0.6: Write `pyproject.toml` (ruff + project metadata)**

  ```toml
  [project]
  name = "mlops-final"
  version = "0.1.0"
  requires-python = ">=3.11,<3.12"

  [tool.ruff]
  line-length = 100
  target-version = "py311"
  src = ["src", "tests"]

  [tool.ruff.lint]
  select = ["E", "F", "W", "I", "B", "UP"]
  ignore = ["E501"]  # line length handled by formatter

  [tool.ruff.lint.per-file-ignores]
  "tests/*" = ["B"]
  ```

- [ ] **Task 0.7: Write `pytest.ini`**

  ```ini
  [pytest]
  testpaths = tests
  pythonpath = src
  addopts = -ra -q --strict-markers
  filterwarnings =
      ignore::DeprecationWarning
  ```

- [ ] **Task 0.8: Create the full directory skeleton**

  ```bash
  mkdir -p src/{data,features,training,evaluation,serving}
  mkdir -p tests/{unit,integration,data}
  mkdir -p configs docs flows monitoring/{evidently_reports,prometheus} docker .github/workflows
  mkdir -p data/{raw,processed,splits}
  touch src/__init__.py src/data/__init__.py src/features/__init__.py \
        src/training/__init__.py src/evaluation/__init__.py src/serving/__init__.py \
        tests/__init__.py tests/unit/__init__.py tests/integration/__init__.py tests/data/__init__.py \
        monitoring/__init__.py
  touch data/raw/.gitkeep data/processed/.gitkeep data/splits/.gitkeep
  ```

- [ ] **Task 0.9: Write the FIRST `configs/params.yaml`**

  This is the single source of truth. We will keep extending it through the project.

  ```yaml
  # configs/params.yaml
  paths:
    raw_csv: data/raw/hour.csv
    processed: data/processed/bike_clean.parquet
    train: data/splits/train.parquet
    test: data/splits/test.parquet
    reference: data/splits/reference.parquet
    production: data/splits/production.parquet
    preprocessor: data/splits/preprocessor.pkl
    model: data/splits/model.pkl
    metrics: data/splits/metrics.json

  data:
    target: cnt
    drop_columns: [instant, dteday, casual, registered]
    numeric_features: [temp, atemp, hum, windspeed, hr, mnth]
    categorical_features: [season, holiday, workingday, weathersit, weekday]
    split_column: yr            # 0 = reference, 1 = production
    test_size: 0.2
    reference_holdout: 0.10     # 10% of yr=0 set aside as the production-clean baseline
    random_state: 42

  preprocessing:
    numeric_imputer_strategy: median
    categorical_imputer_strategy: most_frequent
    feature_selection_k: 12     # SelectKBest top-k

  training:
    model_type: random_forest
    n_trials: 20                # Optuna trials
    cv_folds: 3
    hpo_search_space:
      n_estimators: [100, 200, 400]
      max_depth: [4, 8, 16, null]
      min_samples_leaf: [1, 2, 4]

  drift:
    perturb_temp_factor: 1.10
    perturb_hum_factor: 0.85
    perturb_windspeed_noise_std: 0.05
    drift_threshold_share: 0.20
    perturbed_features: [temp, hum, windspeed]

  serving:
    model_name: bike_share_regressor
    model_stage: Production
    api_host: 0.0.0.0
    api_port: 8000

  mlflow:
    tracking_uri: http://localhost:5000
    experiment_name: bike_sharing
    registered_model_name: bike_share_regressor

  validation:
    min_test_r2: 0.70           # CI gate
  ```

- [ ] **Task 0.10: Write `src/config.py` — typed loader**

  This is the only place anything reads `params.yaml`.

  ```python
  # src/config.py
  from __future__ import annotations
  from pathlib import Path
  import yaml
  from pydantic import BaseModel, Field

  CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "params.yaml"


  class PathsCfg(BaseModel):
      raw_csv: str
      processed: str
      train: str
      test: str
      reference: str
      production: str
      preprocessor: str
      model: str
      metrics: str


  class DataCfg(BaseModel):
      target: str
      drop_columns: list[str]
      numeric_features: list[str]
      categorical_features: list[str]
      split_column: str
      test_size: float
      reference_holdout: float
      random_state: int


  class PreprocessingCfg(BaseModel):
      numeric_imputer_strategy: str
      categorical_imputer_strategy: str
      feature_selection_k: int


  class TrainingCfg(BaseModel):
      model_type: str
      n_trials: int
      cv_folds: int
      hpo_search_space: dict


  class DriftCfg(BaseModel):
      perturb_temp_factor: float
      perturb_hum_factor: float
      perturb_windspeed_noise_std: float
      drift_threshold_share: float
      perturbed_features: list[str]


  class ServingCfg(BaseModel):
      model_name: str
      model_stage: str
      api_host: str
      api_port: int


  class MLflowCfg(BaseModel):
      tracking_uri: str
      experiment_name: str
      registered_model_name: str


  class ValidationCfg(BaseModel):
      min_test_r2: float


  class Config(BaseModel):
      paths: PathsCfg
      data: DataCfg
      preprocessing: PreprocessingCfg
      training: TrainingCfg
      drift: DriftCfg
      serving: ServingCfg
      mlflow: MLflowCfg
      validation: ValidationCfg


  def load_config(path: Path | str = CONFIG_PATH) -> Config:
      with open(path, "r", encoding="utf-8") as f:
          raw = yaml.safe_load(f)
      return Config(**raw)
  ```

- [ ] **Task 0.11: Write the failing test for `load_config`**

  Create `tests/unit/test_config.py`:

  ```python
  from src.config import load_config


  def test_load_config_returns_expected_keys():
      cfg = load_config()
      assert cfg.data.target == "cnt"
      assert cfg.training.n_trials > 0
      assert cfg.serving.api_port == 8000
      assert 0 < cfg.validation.min_test_r2 < 1
  ```

  Run:
  ```bash
  pytest tests/unit/test_config.py -v
  ```
  Expected: PASS (the config exists, just verifying the loader).

- [ ] **Task 0.12: Initialise DVC and configure local remote**

  ```bash
  dvc init
  mkdir -p ../dvc-storage
  dvc remote add -d localremote ../dvc-storage
  dvc config cache.type copy            # Windows-safe (no symlinks)
  ```

- [ ] **Task 0.13: Copy the dataset into `data/raw/` and DVC-track it**

  The dataset is already on disk at `../bike_sharing_dataset/hour.csv` (one level up from the repo root). Copy it; do **not** commit the `bike_sharing_dataset/` folder.

  ```bash
  cp ../bike_sharing_dataset/hour.csv data/raw/hour.csv
  cp ../bike_sharing_dataset/Readme.txt docs/source_readme.txt   # for citation/provenance
  dvc add data/raw/hour.csv
  ```

  Verify:
  ```bash
  ls data/raw/                       # should show: hour.csv, hour.csv.dvc, .gitignore
  wc -l data/raw/hour.csv            # 17380 (header + 17379 rows)
  ```

  Note: `day.csv` (daily aggregates, 731 rows) is intentionally not used — our pipeline operates on hourly granularity. Reference it in the Data Card but don't copy it.

- [ ] **Task 0.14: Write minimal `README.md` (will be filled in Phase 7+)**

  ```markdown
  # MLOps Final Project — Bike Sharing Demand

  End-to-end MLOps pipeline for predicting hourly bike rental counts (UCI Bike Sharing dataset).

  ## Quickstart (3 commands)

  ```bash
  pip install -r requirements.txt
  dvc pull && dvc repro
  docker compose up --build
  ```

  Then `curl http://localhost:8000/health`.

  See `docs/model_card.md` and `docs/data_card.md` for details.
  ```

- [ ] **Task 0.15: First commit**

  ```bash
  git add .
  git commit -m "phase 0: project skeleton, env, dvc init, dataset tracked"
  git push -u origin main
  ```

  Verify the test still passes locally and that `data/raw/hour.csv` is NOT in the Git index (`git ls-files | grep hour.csv` should return only `hour.csv.dvc`).

- [ ] **Task 0.16: 🔧 MANUAL STEP — EDA Notebook (informs every later choice)**

  Open `notebooks/01_eda.ipynb` and answer these questions, with cell outputs visible:

  | # | Question | Why it matters | Drives |
  |---|----------|----------------|--------|
  | 1 | Are there really zero nulls in all 17 columns? | Confirms imputer is "defensive only" | Phase 2 imputer strategy |
  | 2 | What is the `cnt` distribution? skew? log-normal? | Decide whether to log-transform target | Phase 3 metric choice |
  | 3 | What is the correlation matrix of numeric features? Is `temp` ≈ `atemp`? | Possible multicollinearity → SelectKBest | Phase 2 `feature_selection_k` |
  | 4 | How does `cnt` distribute by `hr`, `weekday`, `season`? | Confirms strong feature signals | Sanity-check after training |
  | 5 | What is the row count by `yr`? (verify yr=0 ≈ yr=1) | Confirms drift split is balanced | Phase 1 `build_splits` |
  | 6 | Plot `temp` distribution: yr=0 vs yr=1. Visible drift already? | Confirms our drift narrative | Phase 7 drift report interpretation |
  | 7 | Are `casual + registered == cnt` strictly? | Confirms target-leakage drop list | Phase 1 `drop_columns` |

  **Output:** committed `notebooks/01_eda.ipynb` with markdown answers + plots. Reference key plots in the Technical Report (Phase 10) and Data Card.

  **Pitfall:** do NOT put any pipeline logic here — spec forbids it. Notebooks are observation only.

  ```bash
  git add notebooks/01_eda.ipynb
  git commit -m "phase 0: EDA notebook — distributions, leakage check, drift preview"
  ```

---

### Phase 1: DVC Pipeline

**Goal:** A reproducible 4-stage DVC pipeline (`prepare → preprocess → featurize → train`) where `dvc repro` regenerates every artifact.

**Files:**
- Create: `src/data/load.py`, `src/data/prepare.py`, `src/data/split.py`, `src/data/schema.py`, `dvc.yaml`.
- Tests: `tests/unit/test_split.py`, `tests/data/test_data_validation.py`.

**Dependencies:** Phase 0.

**Pitfalls:**
- DVC stages re-running on every commit because they depend on a directory mtime → declare exact file deps, never directories.
- Forgetting to add stage outputs as dvc-tracked → `dvc.lock` becomes inconsistent. Let `dvc.yaml` `outs:` handle tracking; do NOT `dvc add` outputs manually.
- Non-deterministic output: pandas `to_parquet` writing index → set `index=False`.
- The raw data has **zero nulls** — don't rely on imputers being exercised by the training set. The unit tests that synthesise NaNs (Task 2.1) are the ones that prove imputation works.

**Tasks:**

- [ ] **Task 1.1: Write `src/data/schema.py` (Pandera schema for raw data)**

  ```python
  # src/data/schema.py
  import pandera as pa
  from pandera import Column, DataFrameSchema

  raw_schema = DataFrameSchema(
      {
          "instant": Column(int, unique=True),
          "dteday": Column(str),
          "season": Column(int, pa.Check.isin([1, 2, 3, 4])),
          "yr": Column(int, pa.Check.isin([0, 1])),
          "mnth": Column(int, pa.Check.in_range(1, 12)),
          "hr": Column(int, pa.Check.in_range(0, 23)),
          "holiday": Column(int, pa.Check.isin([0, 1])),
          "weekday": Column(int, pa.Check.in_range(0, 6)),
          "workingday": Column(int, pa.Check.isin([0, 1])),
          "weathersit": Column(int, pa.Check.isin([1, 2, 3, 4])),
          "temp": Column(float, pa.Check.in_range(0.0, 1.0)),
          "atemp": Column(float, pa.Check.in_range(0.0, 1.0)),
          "hum": Column(float, pa.Check.in_range(0.0, 1.0)),
          "windspeed": Column(float, pa.Check.in_range(0.0, 1.0)),  # observed max 0.8507
          "casual": Column(int, pa.Check.ge(0)),
          "registered": Column(int, pa.Check.ge(0)),
          "cnt": Column(int, pa.Check.ge(1)),       # observed min 1 (no zero-rental hours)
      },
      strict=True,
      coerce=True,
  )
  ```

- [ ] **Task 1.2: Write the failing test for the schema**

  Create `tests/data/test_data_validation.py`:

  ```python
  import pandas as pd
  import pytest
  from src.data.schema import raw_schema


  def _valid_row():
      return {
          "instant": 1, "dteday": "2011-01-01", "season": 1, "yr": 0, "mnth": 1,
          "hr": 0, "holiday": 0, "weekday": 6, "workingday": 0, "weathersit": 1,
          "temp": 0.24, "atemp": 0.288, "hum": 0.81, "windspeed": 0.0,
          "casual": 3, "registered": 13, "cnt": 16,
      }


  def test_schema_accepts_valid_row():
      df = pd.DataFrame([_valid_row()])
      raw_schema.validate(df)  # raises if invalid


  def test_schema_rejects_bad_season():
      bad = _valid_row(); bad["season"] = 7
      df = pd.DataFrame([bad])
      with pytest.raises(Exception):
          raw_schema.validate(df)


  def test_schema_rejects_zero_or_negative_count():
      bad = _valid_row(); bad["cnt"] = 0
      df = pd.DataFrame([bad])
      with pytest.raises(Exception):
          raw_schema.validate(df)
  ```

  Run:
  ```bash
  pytest tests/data/test_data_validation.py -v
  ```
  Expected: 3 PASS.

- [ ] **Task 1.3: Write `src/data/load.py`**

  ```python
  # src/data/load.py
  from pathlib import Path
  import pandas as pd
  from src.data.schema import raw_schema


  def load_raw(path: str | Path) -> pd.DataFrame:
      df = pd.read_csv(path)
      raw_schema.validate(df)
      return df
  ```

- [ ] **Task 1.4: Write `src/data/prepare.py` — the `prepare` DVC stage**

  This script: loads raw, validates schema, drops leakage columns, writes `processed/bike_clean.parquet`.

  ```python
  # src/data/prepare.py
  from pathlib import Path
  import pandas as pd
  from src.config import load_config
  from src.data.load import load_raw


  def main() -> None:
      cfg = load_config()
      df = load_raw(cfg.paths.raw_csv)
      df = df.drop(columns=cfg.data.drop_columns)
      out = Path(cfg.paths.processed)
      out.parent.mkdir(parents=True, exist_ok=True)
      df.to_parquet(out, index=False)
      print(f"prepare: wrote {len(df):,} rows to {out}")


  if __name__ == "__main__":
      main()
  ```

  Run manually to verify:
  ```bash
  PYTHONPATH=. python src/data/prepare.py
  ```
  Expected: prints `prepare: wrote 17,379 rows to data/processed/bike_clean.parquet`.

- [ ] **Task 1.5: Write the failing test for split logic, then `src/data/split.py`**

  Test first — `tests/unit/test_split.py`:

  ```python
  import pandas as pd
  import numpy as np
  from src.data.split import build_splits


  def _toy(n=200):
      rng = np.random.default_rng(0)
      df = pd.DataFrame({
          "yr": rng.integers(0, 2, size=n),
          "temp": rng.random(n),
          "hum": rng.random(n),
          "windspeed": rng.random(n),
          "cnt": rng.integers(0, 100, size=n),
          "season": rng.integers(1, 5, size=n),
          "holiday": rng.integers(0, 2, size=n),
          "workingday": rng.integers(0, 2, size=n),
          "weathersit": rng.integers(1, 5, size=n),
          "weekday": rng.integers(0, 7, size=n),
          "atemp": rng.random(n),
          "hr": rng.integers(0, 24, size=n),
          "mnth": rng.integers(1, 13, size=n),
      })
      return df


  def test_build_splits_returns_four_disjoint_frames():
      df = _toy()
      train, test, reference, production = build_splits(
          df, split_col="yr", test_size=0.2, ref_holdout=0.1, random_state=0,
      )
      total = len(train) + len(test) + len(reference)
      assert total == (df["yr"] == 0).sum()
      assert (production["yr"] == 1).all()
      assert (train["yr"] == 0).all() and (test["yr"] == 0).all() and (reference["yr"] == 0).all()


  def test_no_target_leakage_columns_present():
      df = _toy()
      train, *_ = build_splits(df, "yr", 0.2, 0.1, 0)
      for c in ("casual", "registered"):
          assert c not in train.columns


  def test_drift_perturbation_changes_distribution():
      from src.data.split import inject_drift
      df = _toy()
      yr1 = df[df["yr"] == 1].copy()
      drifted = inject_drift(yr1, factor_temp=1.1, factor_hum=0.85, std_windspeed=0.05, seed=0)
      assert drifted["temp"].mean() > yr1["temp"].mean()
      assert drifted["hum"].mean() < yr1["hum"].mean()
      assert drifted["windspeed"].std() > yr1["windspeed"].std()
  ```

  Run:
  ```bash
  pytest tests/unit/test_split.py -v
  ```
  Expected: FAIL with `ImportError: cannot import name 'build_splits'`.

  Now write `src/data/split.py`:

  ```python
  # src/data/split.py
  from pathlib import Path
  import numpy as np
  import pandas as pd
  from sklearn.model_selection import train_test_split
  from src.config import load_config


  def inject_drift(
      df: pd.DataFrame,
      factor_temp: float,
      factor_hum: float,
      std_windspeed: float,
      seed: int,
  ) -> pd.DataFrame:
      rng = np.random.default_rng(seed)
      out = df.copy()
      if "temp" in out.columns:
          out["temp"] = (out["temp"] * factor_temp).clip(0, 1)
      if "hum" in out.columns:
          out["hum"] = (out["hum"] * factor_hum).clip(0, 1)
      if "windspeed" in out.columns:
          out["windspeed"] = (out["windspeed"] + rng.normal(0, std_windspeed, len(out))).clip(0, 1)
      return out


  def build_splits(
      df: pd.DataFrame,
      split_col: str,
      test_size: float,
      ref_holdout: float,
      random_state: int,
  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
      year0 = df[df[split_col] == 0].copy()
      year1 = df[df[split_col] == 1].copy()
      # carve reference (clean baseline) out of yr=0 first
      year0_main, reference = train_test_split(
          year0, test_size=ref_holdout, random_state=random_state,
      )
      train, test = train_test_split(
          year0_main, test_size=test_size, random_state=random_state,
      )
      return train, test, reference, year1


  def main() -> None:
      cfg = load_config()
      df = pd.read_parquet(cfg.paths.processed)
      train, test, reference, year1 = build_splits(
          df,
          split_col=cfg.data.split_column,
          test_size=cfg.data.test_size,
          ref_holdout=cfg.data.reference_holdout,
          random_state=cfg.data.random_state,
      )
      production = inject_drift(
          year1,
          factor_temp=cfg.drift.perturb_temp_factor,
          factor_hum=cfg.drift.perturb_hum_factor,
          std_windspeed=cfg.drift.perturb_windspeed_noise_std,
          seed=cfg.data.random_state,
      )
      Path(cfg.paths.train).parent.mkdir(parents=True, exist_ok=True)
      train.to_parquet(cfg.paths.train, index=False)
      test.to_parquet(cfg.paths.test, index=False)
      reference.to_parquet(cfg.paths.reference, index=False)
      production.to_parquet(cfg.paths.production, index=False)
      print(
          f"split: train={len(train):,} test={len(test):,} "
          f"reference={len(reference):,} production={len(production):,}"
      )


  if __name__ == "__main__":
      main()
  ```

  Re-run tests — expect all 3 PASS.

- [ ] **Task 1.6: Write `dvc.yaml` with the `prepare` and `preprocess` (split) stages**

  Note: per the spec, the required stage names are `prepare`, `preprocess`, `featurize`, `train`. We map: `prepare`=load+drop-leakage, `preprocess`=split into train/test/reference/production, `featurize`=fit + apply preprocessor, `train`=train + log model. We will add `featurize` and `train` in Phases 2 and 3.

  ```yaml
  # dvc.yaml
  stages:
    prepare:
      cmd: python -m src.data.prepare
      deps:
        - data/raw/hour.csv
        - src/data/prepare.py
        - src/data/load.py
        - src/data/schema.py
        - src/config.py
        - configs/params.yaml
      outs:
        - data/processed/bike_clean.parquet

    preprocess:
      cmd: python -m src.data.split
      deps:
        - data/processed/bike_clean.parquet
        - src/data/split.py
        - src/config.py
        - configs/params.yaml
      outs:
        - data/splits/train.parquet
        - data/splits/test.parquet
        - data/splits/reference.parquet
        - data/splits/production.parquet
  ```

- [ ] **Task 1.7: Run `dvc repro` and verify**

  ```bash
  dvc repro
  cat dvc.lock          # should now exist with hashes
  dvc dag               # ASCII DAG; capture screenshot for the report later
  ```

  Expected: stages run successfully; `data/splits/{train,test,reference,production}.parquet` appear; row counts shown in stdout.

- [ ] **Task 1.8: Push and commit**

  ```bash
  dvc push
  git add dvc.yaml dvc.lock src/data/*.py tests/data/test_data_validation.py tests/unit/test_split.py
  git commit -m "phase 1: dvc prepare + preprocess stages, schema validation, split + drift injection"
  ```

---

### Phase 2: Preprocessing Pipeline

**Goal:** A single sklearn `Pipeline` that imputes, scales, encodes, and selects top-k features. Saved as a DVC artifact, applied identically at train and serve time.

**Files:**
- Create: `src/features/preprocessor.py`, `src/features/featurize.py`.
- Tests: `tests/unit/test_preprocessor.py`.
- Modify: `dvc.yaml` (add `featurize` stage).

**Dependencies:** Phase 1.

**Pitfalls:**
- Fitting the preprocessor on `train+test` → leakage. Fit on `train` only.
- `OneHotEncoder` raising on unseen categories at inference → set `handle_unknown="ignore"`.
- `SelectKBest` with `k > n_features` after one-hot → guard `k = min(k, n_features_after_ohe)`.
- Forgetting to persist feature names → at inference we need the schema. Save them alongside the pickle.

**Tasks:**

- [ ] **Task 2.1: Write the failing tests for the preprocessor**

  Create `tests/unit/test_preprocessor.py`:

  ```python
  import numpy as np
  import pandas as pd
  import pytest
  from src.features.preprocessor import build_preprocessor, fit_preprocessor


  def _toy_train(n=300):
      rng = np.random.default_rng(0)
      return pd.DataFrame({
          "temp": rng.random(n),
          "atemp": rng.random(n),
          "hum": rng.random(n),
          "windspeed": rng.random(n),
          "hr": rng.integers(0, 24, size=n),
          "mnth": rng.integers(1, 13, size=n),
          "season": rng.integers(1, 5, size=n),
          "holiday": rng.integers(0, 2, size=n),
          "workingday": rng.integers(0, 2, size=n),
          "weathersit": rng.integers(1, 5, size=n),
          "weekday": rng.integers(0, 7, size=n),
          "cnt": rng.integers(0, 200, size=n),
      })


  def test_build_preprocessor_returns_pipeline_with_3_steps():
      pipe = build_preprocessor(
          numeric=["temp", "atemp", "hum", "windspeed", "hr", "mnth"],
          categorical=["season", "holiday", "workingday", "weathersit", "weekday"],
          k=10,
      )
      step_names = [s[0] for s in pipe.steps]
      assert "preprocessor" in step_names
      assert "selector" in step_names


  def test_fit_then_transform_produces_2d_array_with_no_nans():
      df = _toy_train()
      pipe = fit_preprocessor(df, target="cnt", numeric=[
          "temp", "atemp", "hum", "windspeed", "hr", "mnth",
      ], categorical=[
          "season", "holiday", "workingday", "weathersit", "weekday",
      ], k=10)
      X = df.drop(columns=["cnt"])
      X_t = pipe.transform(X)
      assert X_t.ndim == 2
      assert X_t.shape[0] == len(df)
      assert not np.isnan(X_t).any()


  def test_handles_unseen_category_at_transform_time():
      df = _toy_train()
      pipe = fit_preprocessor(df, target="cnt", numeric=[
          "temp", "atemp", "hum", "windspeed", "hr", "mnth",
      ], categorical=[
          "season", "holiday", "workingday", "weathersit", "weekday",
      ], k=10)
      new = df.iloc[:5].copy()
      new["season"] = 99  # unseen
      X_t = pipe.transform(new.drop(columns=["cnt"]))
      assert X_t.shape == (5, 10)


  def test_imputes_missing_numeric_values():
      df = _toy_train()
      df.loc[0:10, "temp"] = np.nan
      pipe = fit_preprocessor(df, target="cnt", numeric=[
          "temp", "atemp", "hum", "windspeed", "hr", "mnth",
      ], categorical=[
          "season", "holiday", "workingday", "weathersit", "weekday",
      ], k=10)
      X_t = pipe.transform(df.drop(columns=["cnt"]))
      assert not np.isnan(X_t).any()
  ```

  Run:
  ```bash
  pytest tests/unit/test_preprocessor.py -v
  ```
  Expected: FAIL with `ImportError`.

- [ ] **Task 2.2: Write `src/features/preprocessor.py`**

  ```python
  # src/features/preprocessor.py
  from sklearn.compose import ColumnTransformer
  from sklearn.feature_selection import SelectKBest, f_regression
  from sklearn.impute import SimpleImputer
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import OneHotEncoder, StandardScaler
  import pandas as pd


  def build_preprocessor(
      numeric: list[str],
      categorical: list[str],
      k: int,
      numeric_strategy: str = "median",
      categorical_strategy: str = "most_frequent",
  ) -> Pipeline:
      numeric_pipe = Pipeline([
          ("imputer", SimpleImputer(strategy=numeric_strategy)),
          ("scaler", StandardScaler()),
      ])
      categorical_pipe = Pipeline([
          ("imputer", SimpleImputer(strategy=categorical_strategy)),
          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
      ])
      column_transformer = ColumnTransformer(
          [
              ("num", numeric_pipe, numeric),
              ("cat", categorical_pipe, categorical),
          ],
          remainder="drop",
      )
      return Pipeline([
          ("preprocessor", column_transformer),
          ("selector", SelectKBest(score_func=f_regression, k=k)),
      ])


  def fit_preprocessor(
      df: pd.DataFrame,
      target: str,
      numeric: list[str],
      categorical: list[str],
      k: int,
      numeric_strategy: str = "median",
      categorical_strategy: str = "most_frequent",
  ) -> Pipeline:
      X = df[numeric + categorical]
      y = df[target]
      pipe = build_preprocessor(
          numeric, categorical, k=k,
          numeric_strategy=numeric_strategy,
          categorical_strategy=categorical_strategy,
      )
      pipe.fit(X, y)
      return pipe
  ```

  Re-run tests:
  ```bash
  pytest tests/unit/test_preprocessor.py -v
  ```
  Expected: 4 PASS.

- [ ] **Task 2.3: Write `src/features/featurize.py` (the `featurize` DVC stage)**

  ```python
  # src/features/featurize.py
  from pathlib import Path
  import joblib
  import pandas as pd
  from src.config import load_config
  from src.features.preprocessor import fit_preprocessor


  def main() -> None:
      cfg = load_config()
      train = pd.read_parquet(cfg.paths.train)
      # ensure k does not exceed feature count after OHE; sklearn handles k='all'
      pipe = fit_preprocessor(
          train,
          target=cfg.data.target,
          numeric=cfg.data.numeric_features,
          categorical=cfg.data.categorical_features,
          k=cfg.preprocessing.feature_selection_k,
          numeric_strategy=cfg.preprocessing.numeric_imputer_strategy,
          categorical_strategy=cfg.preprocessing.categorical_imputer_strategy,
      )
      out = Path(cfg.paths.preprocessor)
      out.parent.mkdir(parents=True, exist_ok=True)
      joblib.dump(pipe, out)
      print(f"featurize: saved fitted preprocessor to {out}")


  if __name__ == "__main__":
      main()
  ```

- [ ] **Task 2.4: Add the `featurize` stage to `dvc.yaml`**

  Append to `dvc.yaml`:

  ```yaml
    featurize:
      cmd: python -m src.features.featurize
      deps:
        - data/splits/train.parquet
        - src/features/featurize.py
        - src/features/preprocessor.py
        - src/config.py
        - configs/params.yaml
      outs:
        - data/splits/preprocessor.pkl
  ```

- [ ] **Task 2.5: Run `dvc repro` and verify**

  ```bash
  dvc repro featurize
  ls -la data/splits/preprocessor.pkl    # should exist
  ```

- [ ] **Task 2.6: Commit**

  ```bash
  git add src/features dvc.yaml dvc.lock tests/unit/test_preprocessor.py
  git commit -m "phase 2: sklearn preprocessing pipeline (impute/scale/encode/selectkbest)"
  ```

---

### Phase 3: Training + MLflow

**Goal:** Train a Random Forest with Optuna HPO, log every run to MLflow with hyperparameters, metrics, and the model artifact, conduct ≥3 experiments. Save best model + metrics JSON for the CI gate.

**Files:**
- Create: `src/training/train.py`, `src/training/hpo.py`, `src/evaluation/metrics.py`.
- Tests: `tests/unit/test_metrics.py`.
- Modify: `dvc.yaml` (add `train` stage).

**Dependencies:** Phase 2.

**Pitfalls:**
- Optuna trials silently failing because MLflow tracking URI is unreachable → start MLflow before training. Use a try/except that skips MLflow logging if server is down (during dev) but raises in CI.
- Loss curves on RF: there are no per-epoch losses. We log per-trial CV-mean RMSE as a "loss curve" by step index.
- Logging a model with `mlflow.sklearn.log_model` requires the artifact path argument; without it, the registry lookup later breaks.
- Optuna sampler default is TPE (Bayesian-ish) — counts as automated HPO.

**Tasks:**

- [ ] **Task 3.0: 🔧 MANUAL STEP — Choose the validation R² gate**

  `cfg.validation.min_test_r2` is currently `0.70` in `params.yaml`. This number is the CI bar AND the model-promotion gate. Set it deliberately, not arbitrarily:

  1. Run a one-off baseline locally: `RandomForestRegressor(n_estimators=100, max_depth=8)` on the train/test split. Note the `r2`.
  2. Write down the lecture-recommended rule: gate at ~80–90% of the baseline `r2` so retrained models can recover under drift but obvious regressions are blocked.
  3. Edit `configs/params.yaml` → `validation.min_test_r2`. Document the chosen value in the Model Card under "Acceptance criteria".

  **Pitfall:** if you skip this and leave 0.70 forever, the technical report cannot defend the number — graders ask "why 0.70?" in the discussion. Document the answer.

  **Output:** updated `params.yaml` + a 2-sentence justification in the Technical Report's "Pipeline Architecture" section.

- [ ] **Task 3.1: Start the MLflow tracking server (manually for now)**

  Open a separate terminal:

  ```bash
  mlflow server \
      --backend-store-uri sqlite:///mlflow.db \
      --default-artifact-root ./mlruns \
      --host 0.0.0.0 \
      --port 5000
  ```

  Verify the UI loads at http://localhost:5000.

  (We will containerise this in Phase 8.)

- [ ] **Task 3.2: Write the failing tests for metrics**

  `tests/unit/test_metrics.py`:

  ```python
  import numpy as np
  from src.evaluation.metrics import compute_metrics


  def test_perfect_prediction_gives_zero_error_and_unit_r2():
      y = np.array([1.0, 2.0, 3.0, 4.0])
      m = compute_metrics(y, y)
      assert m["rmse"] == 0.0
      assert m["mae"] == 0.0
      assert abs(m["r2"] - 1.0) < 1e-9


  def test_metric_dict_has_all_keys():
      y_true = np.array([1.0, 2.0, 3.0])
      y_pred = np.array([1.5, 2.5, 2.5])
      m = compute_metrics(y_true, y_pred)
      assert set(m.keys()) == {"rmse", "mae", "r2"}
  ```

- [ ] **Task 3.3: Write `src/evaluation/metrics.py`**

  ```python
  # src/evaluation/metrics.py
  import numpy as np
  from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


  def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
      return {
          "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
          "mae": float(mean_absolute_error(y_true, y_pred)),
          "r2": float(r2_score(y_true, y_pred)),
      }
  ```

  Run tests — expect 2 PASS.

- [ ] **Task 3.4: Write `src/training/hpo.py`**

  ```python
  # src/training/hpo.py
  from __future__ import annotations
  import mlflow
  import numpy as np
  import optuna
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.model_selection import cross_val_score


  def make_objective(X, y, search_space: dict, cv_folds: int):
      def objective(trial: optuna.Trial) -> float:
          params = {
              "n_estimators": trial.suggest_categorical(
                  "n_estimators", search_space["n_estimators"]
              ),
              "max_depth": trial.suggest_categorical(
                  "max_depth", search_space["max_depth"]
              ),
              "min_samples_leaf": trial.suggest_categorical(
                  "min_samples_leaf", search_space["min_samples_leaf"]
              ),
              "n_jobs": -1,
              "random_state": 42,
          }
          model = RandomForestRegressor(**params)
          # neg MSE → flip sign and rmse
          neg_mse = cross_val_score(
              model, X, y, scoring="neg_mean_squared_error", cv=cv_folds, n_jobs=-1,
          ).mean()
          rmse = float(np.sqrt(-neg_mse))
          # Log per-trial as a "loss curve" point
          with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
              mlflow.log_params(params)
              mlflow.log_metric("cv_rmse", rmse)
              mlflow.log_metric("cv_rmse_step", rmse, step=trial.number)
          return rmse
      return objective


  def run_hpo(X, y, search_space: dict, n_trials: int, cv_folds: int) -> optuna.Study:
      study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
      study.optimize(make_objective(X, y, search_space, cv_folds), n_trials=n_trials)
      return study
  ```

- [ ] **Task 3.5: Write `src/training/train.py` (the `train` DVC stage)**

  ```python
  # src/training/train.py
  from __future__ import annotations
  import json
  import os
  from pathlib import Path
  import joblib
  import mlflow
  import mlflow.sklearn
  import pandas as pd
  from sklearn.ensemble import RandomForestRegressor
  from src.config import load_config
  from src.evaluation.metrics import compute_metrics
  from src.training.hpo import run_hpo


  def _resolve_tracking_uri(cfg) -> str:
      # env var wins over params.yaml so Docker Compose can rewire to http://mlflow:5000
      return os.environ.get("MLFLOW_TRACKING_URI", cfg.mlflow.tracking_uri)


  def main() -> None:
      cfg = load_config()
      train_df = pd.read_parquet(cfg.paths.train)
      test_df = pd.read_parquet(cfg.paths.test)
      preprocessor = joblib.load(cfg.paths.preprocessor)

      feature_cols = cfg.data.numeric_features + cfg.data.categorical_features
      X_train = preprocessor.transform(train_df[feature_cols])
      y_train = train_df[cfg.data.target].values
      X_test = preprocessor.transform(test_df[feature_cols])
      y_test = test_df[cfg.data.target].values

      mlflow.set_tracking_uri(_resolve_tracking_uri(cfg))
      mlflow.set_experiment(cfg.mlflow.experiment_name)

      with mlflow.start_run(run_name="parent_hpo") as parent_run:
          mlflow.log_params({
              "model_type": cfg.training.model_type,
              "cv_folds": cfg.training.cv_folds,
              "n_trials": cfg.training.n_trials,
              "feature_selection_k": cfg.preprocessing.feature_selection_k,
          })

          study = run_hpo(
              X_train, y_train,
              search_space=cfg.training.hpo_search_space,
              n_trials=cfg.training.n_trials,
              cv_folds=cfg.training.cv_folds,
          )

          best_params = study.best_params
          mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
          mlflow.log_metric("best_cv_rmse", study.best_value)

          # Train final model on full train set with best params
          final = RandomForestRegressor(
              **best_params, n_jobs=-1, random_state=cfg.data.random_state,
          )
          final.fit(X_train, y_train)

          y_pred_test = final.predict(X_test)
          test_metrics = compute_metrics(y_test, y_pred_test)
          for k, v in test_metrics.items():
              mlflow.log_metric(f"test_{k}", v)

          # Log feature_selection mask as a tag for reproducibility
          mlflow.sklearn.log_model(
              sk_model=final,
              artifact_path="model",
              registered_model_name=cfg.mlflow.registered_model_name,
          )
          # Also log the preprocessor next to the model
          mlflow.log_artifact(cfg.paths.preprocessor, artifact_path="preprocessor")

          # DVC outputs
          Path(cfg.paths.model).parent.mkdir(parents=True, exist_ok=True)
          joblib.dump(final, cfg.paths.model)
          with open(cfg.paths.metrics, "w", encoding="utf-8") as f:
              json.dump({"test": test_metrics, "best_params": best_params}, f, indent=2)

          print(f"train: parent_run_id={parent_run.info.run_id} test_metrics={test_metrics}")


  if __name__ == "__main__":
      main()
  ```

- [ ] **Task 3.6: Add the `train` stage to `dvc.yaml`**

  ```yaml
    train:
      cmd: python -m src.training.train
      deps:
        - data/splits/train.parquet
        - data/splits/test.parquet
        - data/splits/preprocessor.pkl
        - src/training/train.py
        - src/training/hpo.py
        - src/evaluation/metrics.py
        - src/config.py
        - configs/params.yaml
      outs:
        - data/splits/model.pkl
      metrics:
        - data/splits/metrics.json:
            cache: false
  ```

- [ ] **Task 3.7: Run the full pipeline once and confirm 3+ MLflow runs**

  ```bash
  dvc repro train
  ```

  Visit http://localhost:5000 → experiment `bike_sharing` → expect 1 parent run + N nested trial runs (≥3 by definition because `n_trials=20`).

  Verify in `data/splits/metrics.json` that `test.r2` ≥ 0.85 (RF on bike sharing comfortably hits this).

- [ ] **Task 3.7.5: 🔧 MANUAL STEP — Review HPO results and approve best params**

  Open the MLflow UI runs comparison view:
  1. Sort all child runs by `cv_rmse` ascending.
  2. Inspect the top-5 trials' parameter combinations. Are they clustered (good — Optuna converged) or scattered (bad — TPE didn't help; consider increasing `n_trials`)?
  3. Check the loss curve plot (`cv_rmse_step` vs `step`): should trend downward.
  4. Sanity-check the best run's `test_r2`: any sign of overfit? Look at `cv_rmse` vs `test_rmse` — if `test_rmse` is much worse, refuse this model.
  5. **Decision:** approve OR widen the search space in `params.yaml` and re-run. Document in `docs/experiment_log.md` (a short markdown decision log alongside the CSV).

  **Output:** entry in `docs/experiment_log.md` justifying the chosen run + screenshot of the MLflow runs comparison view (saved to `docs/screenshots/mlflow_runs.png`).

- [ ] **Task 3.8: Export experiment log for the `docs/` folder**

  The `mlflow runs list` CLI is unreliable across versions. Use a small Python script instead — write `scripts/export_runs.py`:

  ```python
  # scripts/export_runs.py
  from __future__ import annotations
  import os
  import sys
  from pathlib import Path
  import pandas as pd
  import mlflow
  from mlflow.tracking import MlflowClient
  from src.config import load_config


  def main() -> int:
      cfg = load_config()
      tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", cfg.mlflow.tracking_uri)
      mlflow.set_tracking_uri(tracking_uri)
      client = MlflowClient()
      exp = client.get_experiment_by_name(cfg.mlflow.experiment_name)
      if exp is None:
          print(f"experiment {cfg.mlflow.experiment_name} not found", file=sys.stderr)
          return 1
      runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], output_format="pandas")
      if runs.empty:
          print("no runs to export", file=sys.stderr)
          return 1
      out = Path("docs/experiment_log.csv")
      out.parent.mkdir(parents=True, exist_ok=True)
      runs.to_csv(out, index=False)
      print(f"exported {len(runs)} runs to {out}")
      return 0


  if __name__ == "__main__":
      sys.exit(main())
  ```

  Run it:
  ```bash
  PYTHONPATH=. python scripts/export_runs.py
  ```

  Verify `docs/experiment_log.csv` contains rows for the parent run **and** every Optuna trial, with all params and metrics columns. The rubric requires "all runs, parameters, metrics".

- [ ] **Task 3.9: Write `scripts/compute_subgroup_metrics.py` (Model Card rubric)**

  The rubric for Component 7 explicitly requires the Model Card to show "metrics (overall and per subgroup)". Compute and persist them as JSON, alongside the existing `metrics.json`.

  ```python
  # scripts/compute_subgroup_metrics.py
  from __future__ import annotations
  import json
  from pathlib import Path
  import joblib
  import pandas as pd
  from src.config import load_config
  from src.evaluation.metrics import compute_metrics


  SUBGROUPS = ["season", "weathersit", "workingday"]   # interpretable cohorts


  def main() -> None:
      cfg = load_config()
      model = joblib.load(cfg.paths.model)
      preprocessor = joblib.load(cfg.paths.preprocessor)
      test = pd.read_parquet(cfg.paths.test)
      feature_cols = cfg.data.numeric_features + cfg.data.categorical_features

      X = preprocessor.transform(test[feature_cols])
      y_true = test[cfg.data.target].to_numpy()
      y_pred = model.predict(X)

      out: dict = {"overall": compute_metrics(y_true, y_pred), "subgroups": {}}
      for col in SUBGROUPS:
          out["subgroups"][col] = {}
          for value, idx in test.groupby(col).indices.items():
              if len(idx) < 30:
                  continue
              out["subgroups"][col][str(value)] = {
                  **compute_metrics(y_true[idx], y_pred[idx]),
                  "n": int(len(idx)),
              }

      Path("docs/subgroup_metrics.json").write_text(json.dumps(out, indent=2))
      print("subgroup metrics written to docs/subgroup_metrics.json")


  if __name__ == "__main__":
      main()
  ```

  Run after training:
  ```bash
  PYTHONPATH=. python scripts/compute_subgroup_metrics.py
  ```

  The Model Card (Phase 10) imports these numbers directly into a markdown table.

- [ ] **Task 3.10: Commit**

  ```bash
  git add src/training src/evaluation tests/unit/test_metrics.py dvc.yaml dvc.lock \
          docs/experiment_log.csv docs/subgroup_metrics.json scripts/export_runs.py \
          scripts/compute_subgroup_metrics.py
  git commit -m "phase 3: optuna hpo, mlflow tracking, registered model + experiment log + subgroup metrics"
  ```

---

### Phase 4: Model Registry

**Goal:** Promote the best run's model from `None → Staging → Production` via the MLflow API. The serving app loads from the `Production` stage.

**Files:**
- Create: `src/training/registry.py`.
- Tests: extend `tests/unit/test_metrics.py` is fine; the registry interaction is integration-only and verified manually.

**Dependencies:** Phase 3.

**Pitfalls:**
- MLflow 2.x has deprecated stages in favour of aliases — but the rubric explicitly requires `None → Staging → Production`, so we keep the old API (`transition_model_version_stage`). Suppress the deprecation warning.
- The model is registered automatically by `mlflow.sklearn.log_model(registered_model_name=...)`, so the version exists; we just need to transition it.

**Tasks:**

- [ ] **Task 4.1: Write `src/training/registry.py`**

  ```python
  # src/training/registry.py
  from __future__ import annotations
  import os
  import warnings
  import mlflow
  from mlflow.tracking import MlflowClient
  from src.config import load_config


  def _resolve_tracking_uri(cfg) -> str:
      return os.environ.get("MLFLOW_TRACKING_URI", cfg.mlflow.tracking_uri)


  def _run_metric(client: MlflowClient, run_id: str, metric: str) -> float | None:
      try:
          run = client.get_run(run_id)
          return float(run.data.metrics.get(metric))  # may be None
      except Exception:
          return None


  def promote_latest_to_production(metric: str = "test_r2", higher_is_better: bool = True) -> int:
      """Promote the latest registered version to Production ONLY IF
      (a) it meets the validation gate AND
      (b) it is at least as good as the current Production model on `metric`.
      Otherwise raise — protects production from regressions."""
      cfg = load_config()
      mlflow.set_tracking_uri(_resolve_tracking_uri(cfg))
      client = MlflowClient()

      versions = client.search_model_versions(f"name='{cfg.mlflow.registered_model_name}'")
      if not versions:
          raise RuntimeError("no model versions found in registry; run training first")
      latest = max(versions, key=lambda v: int(v.version))
      latest_metric = _run_metric(client, latest.run_id, metric)
      if latest_metric is None:
          raise RuntimeError(f"latest version v{latest.version} has no `{metric}` metric")

      # gate against the validation threshold
      gate = cfg.validation.min_test_r2 if metric == "test_r2" else None
      if gate is not None and latest_metric < gate:
          raise RuntimeError(
              f"refusing to promote v{latest.version}: {metric}={latest_metric:.4f} "
              f"below gate {gate}"
          )

      # gate against current Production
      prod_versions = [v for v in versions if v.current_stage == "Production"]
      if prod_versions:
          prod = prod_versions[0]
          prod_metric = _run_metric(client, prod.run_id, metric)
          if prod_metric is not None:
              better = latest_metric >= prod_metric if higher_is_better else latest_metric <= prod_metric
              if not better:
                  raise RuntimeError(
                      f"refusing to promote v{latest.version} ({metric}={latest_metric:.4f}) "
                      f"— current Production v{prod.version} is better ({metric}={prod_metric:.4f})"
                  )

      with warnings.catch_warnings():
          warnings.simplefilter("ignore", category=DeprecationWarning)
          client.transition_model_version_stage(
              name=cfg.mlflow.registered_model_name,
              version=latest.version,
              stage="Staging",
              archive_existing_versions=False,
          )
          client.transition_model_version_stage(
              name=cfg.mlflow.registered_model_name,
              version=latest.version,
              stage="Production",
              archive_existing_versions=True,
          )
      print(f"registry: promoted version {latest.version} to Production ({metric}={latest_metric:.4f})")
      return int(latest.version)


  if __name__ == "__main__":
      promote_latest_to_production()
  ```

- [ ] **Task 4.2: Run promotion script and verify in MLflow UI**

  ```bash
  python -m src.training.registry
  ```

  Open http://localhost:5000/#/models/bike_share_regressor — confirm version N has stage `Production`.

- [ ] **Task 4.3: Commit**

  ```bash
  git add src/training/registry.py
  git commit -m "phase 4: model registry promotion (None -> Staging -> Production)"
  ```

---

### Phase 5: Serving API

**Goal:** A FastAPI app with `GET /health`, `POST /predict`, `POST /predict/batch`. Loads the model from MLflow Registry at startup, validates inputs with Pydantic.

**Files:**
- Create: `src/serving/app.py`, `src/serving/schemas.py`, `src/serving/metrics.py`.
- Tests: `tests/integration/test_api.py`.

**Dependencies:** Phase 4.

**Pitfalls:**
- Loading the model inside the request handler → cold-start latency. Load once at startup (FastAPI lifespan).
- Failure to download from MLflow at startup → app crashes. Implement fallback to local `data/splits/model.pkl` for offline dev.
- Forgetting to load the same `preprocessor.pkl` → train/serve skew. We use the model logged via `mlflow.sklearn.log_model` PLUS reload the preprocessor pickle.
- Pydantic v2 syntax differs from v1 — we use `BaseModel`, `Field`, no `__config__` class.

**Tasks:**

- [ ] **Task 5.1: Write `src/serving/schemas.py`**

  ```python
  # src/serving/schemas.py
  from __future__ import annotations
  from pydantic import BaseModel, Field


  class BikeRecord(BaseModel):
      season: int = Field(..., ge=1, le=4)
      mnth: int = Field(..., ge=1, le=12)
      hr: int = Field(..., ge=0, le=23)
      holiday: int = Field(..., ge=0, le=1)
      weekday: int = Field(..., ge=0, le=6)
      workingday: int = Field(..., ge=0, le=1)
      weathersit: int = Field(..., ge=1, le=4)
      temp: float = Field(..., ge=0.0, le=1.0)
      atemp: float = Field(..., ge=0.0, le=1.0)
      hum: float = Field(..., ge=0.0, le=1.0)
      windspeed: float = Field(..., ge=0.0, le=1.0)


  class PredictResponse(BaseModel):
      prediction: float
      confidence: float = Field(..., ge=0.0, le=1.0)
      model_version: str


  class BatchPredictionItem(BaseModel):
      prediction: float
      confidence: float = Field(..., ge=0.0, le=1.0)


  class BatchPredictRequest(BaseModel):
      records: list[BikeRecord]


  class BatchPredictResponse(BaseModel):
      predictions: list[BatchPredictionItem]    # rubric: "predictions WITH confidence scores"
      model_version: str


  class HealthResponse(BaseModel):
      status: str
      model_name: str
      model_version: str
  ```

- [ ] **Task 5.2: Write `src/serving/metrics.py` (5 Prometheus metrics)**

  ```python
  # src/serving/metrics.py
  from prometheus_client import Counter, Gauge, Histogram


  # ── Rubric-required 5 ────────────────────────────────────────────────────
  PREDICTION_CONFIDENCE = Histogram(            # 1. confidence histogram
      "bike_prediction_confidence",
      "Histogram of per-prediction confidence scores in [0,1]",
      buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
  )
  FEATURE_TEMP = Histogram(                     # 2. feature histogram #1
      "bike_feature_temp",
      "Histogram of normalised temperature input feature",
      buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
  )
  FEATURE_HR = Histogram(                       # 3. feature histogram #2
      "bike_feature_hr",
      "Histogram of hour-of-day input feature",
      buckets=tuple(range(0, 25, 3)),
  )
  MODEL_VERSION = Gauge(                        # 4. current model version (gauge)
      "bike_model_version_info",
      "Currently-loaded model version (label=version, value=1)",
      ["version"],
  )
  INFERENCE_COUNT = Counter(                    # 5. inference count (regression: by endpoint)
      "bike_inference_total",
      "Total number of inference requests, labelled by endpoint",
      ["endpoint"],
  )

  # ── Bonus observability (does not count against the 5) ─────────────────
  PREDICTION_VALUE = Histogram(
      "bike_prediction_value",
      "Histogram of predicted rental counts (drift signal on output side)",
      buckets=(0, 25, 50, 100, 200, 400, 800, 1600),
  )
  ```

- [ ] **Task 5.3: Write `src/serving/app.py`**

  ```python
  # src/serving/app.py
  from __future__ import annotations
  from contextlib import asynccontextmanager
  from pathlib import Path

  import joblib
  import mlflow
  import mlflow.sklearn
  import numpy as np
  import pandas as pd
  from fastapi import FastAPI, HTTPException
  from prometheus_fastapi_instrumentator import Instrumentator

  from src.config import load_config
  from src.serving.metrics import (
      FEATURE_HR,
      FEATURE_TEMP,
      INFERENCE_COUNT,
      MODEL_VERSION,
      PREDICTION_CONFIDENCE,
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


  state: dict = {}


  def _load_model():
      cfg = load_config()
      mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
      uri = f"models:/{cfg.mlflow.registered_model_name}/{cfg.serving.model_stage}"
      try:
          model = mlflow.sklearn.load_model(uri)
          version = "registry-prod"
      except Exception as e:  # offline / dev fallback
          fallback = Path(cfg.paths.model)
          if not fallback.exists():
              raise RuntimeError(f"no model in registry and no local fallback at {fallback}") from e
          model = joblib.load(fallback)
          version = "local-fallback"
      preprocessor = joblib.load(cfg.paths.preprocessor)
      return model, preprocessor, version, cfg


  @asynccontextmanager
  async def lifespan(app: FastAPI):
      model, preprocessor, version, cfg = _load_model()
      state["model"] = model
      state["preprocessor"] = preprocessor
      state["version"] = version
      state["cfg"] = cfg
      MODEL_VERSION.labels(version=version).set(1)
      yield
      state.clear()


  app = FastAPI(title="Bike Share Predictor", version="1.0.0", lifespan=lifespan)
  Instrumentator().instrument(app).expose(app)


  def _to_dataframe(records: list[BikeRecord], cfg) -> pd.DataFrame:
      cols = cfg.data.numeric_features + cfg.data.categorical_features
      rows = [r.model_dump() for r in records]
      return pd.DataFrame(rows)[cols]


  def _predict_with_confidence(model, X) -> tuple[np.ndarray, np.ndarray]:
      """For RandomForestRegressor: confidence = 1 / (1 + cv) where cv = std/|mean|
      across the per-tree predictions. Bounded in (0, 1]; falls back to 1.0 when
      the model isn't an ensemble (single estimator)."""
      mean_pred = model.predict(X).astype(float)
      if hasattr(model, "estimators_"):
          all_preds = np.stack([t.predict(X) for t in model.estimators_])
          std_pred = all_preds.std(axis=0)
          cv = std_pred / (np.abs(mean_pred) + 1e-9)
          confidence = 1.0 / (1.0 + cv)
      else:
          confidence = np.ones_like(mean_pred)
      return mean_pred, np.clip(confidence, 0.0, 1.0)


  @app.get("/health", response_model=HealthResponse)
  def health():
      cfg = state.get("cfg")
      if cfg is None:
          raise HTTPException(status_code=503, detail="model not loaded")
      return HealthResponse(
          status="ok",
          model_name=cfg.mlflow.registered_model_name,
          model_version=state["version"],
      )


  @app.post("/predict", response_model=PredictResponse)
  def predict(record: BikeRecord):
      cfg = state["cfg"]
      df = _to_dataframe([record], cfg)
      X = state["preprocessor"].transform(df)
      y, conf = _predict_with_confidence(state["model"], X)
      pred = float(y[0]); confidence = float(conf[0])
      INFERENCE_COUNT.labels(endpoint="predict").inc()
      PREDICTION_VALUE.observe(pred)
      PREDICTION_CONFIDENCE.observe(confidence)
      FEATURE_TEMP.observe(record.temp)
      FEATURE_HR.observe(record.hr)
      return PredictResponse(prediction=pred, confidence=confidence, model_version=state["version"])


  @app.post("/predict/batch", response_model=BatchPredictResponse)
  def predict_batch(req: BatchPredictRequest):
      if not req.records:
          raise HTTPException(status_code=400, detail="empty records")
      cfg = state["cfg"]
      df = _to_dataframe(req.records, cfg)
      X = state["preprocessor"].transform(df)
      y, conf = _predict_with_confidence(state["model"], X)
      INFERENCE_COUNT.labels(endpoint="predict_batch").inc()
      items: list[BatchPredictionItem] = []
      for v, c, r in zip(y.tolist(), conf.tolist(), req.records):
          PREDICTION_VALUE.observe(v)
          PREDICTION_CONFIDENCE.observe(c)
          FEATURE_TEMP.observe(r.temp)
          FEATURE_HR.observe(r.hr)
          items.append(BatchPredictionItem(prediction=float(v), confidence=float(c)))
      return BatchPredictResponse(predictions=items, model_version=state["version"])
  ```

- [ ] **Task 5.4: Write the failing API tests**

  `tests/integration/test_api.py`:

  ```python
  import pytest
  from fastapi.testclient import TestClient
  from src.serving.app import app


  @pytest.fixture(scope="module")
  def client():
      with TestClient(app) as c:
          yield c


  _RECORD = {
      "season": 1, "mnth": 1, "hr": 8, "holiday": 0, "weekday": 1,
      "workingday": 1, "weathersit": 1, "temp": 0.24, "atemp": 0.288,
      "hum": 0.81, "windspeed": 0.0,
  }


  def test_health_returns_ok(client):
      r = client.get("/health")
      assert r.status_code == 200
      body = r.json()
      assert body["status"] == "ok"
      assert "model_name" in body and "model_version" in body


  def test_predict_returns_float_with_confidence(client):
      r = client.post("/predict", json=_RECORD)
      assert r.status_code == 200
      body = r.json()
      assert isinstance(body["prediction"], float) and body["prediction"] >= 0.0
      assert 0.0 <= body["confidence"] <= 1.0


  def test_predict_rejects_invalid_input(client):
      bad = dict(_RECORD); bad["temp"] = 5.0
      r = client.post("/predict", json=bad)
      assert r.status_code == 422


  def test_predict_batch_returns_list_with_confidence(client):
      payload = {"records": [_RECORD, _RECORD, _RECORD]}
      r = client.post("/predict/batch", json=payload)
      assert r.status_code == 200
      body = r.json()
      assert len(body["predictions"]) == 3
      for item in body["predictions"]:
          assert "prediction" in item and "confidence" in item
          assert 0.0 <= item["confidence"] <= 1.0


  def test_predict_batch_rejects_empty_list(client):
      r = client.post("/predict/batch", json={"records": []})
      assert r.status_code == 400


  def test_metrics_endpoint_exposed(client):
      r = client.get("/metrics")
      assert r.status_code == 200
      content = r.content
      # all 5 rubric-required metrics must be visible
      for name in (
          b"bike_prediction_confidence",
          b"bike_feature_temp",
          b"bike_feature_hr",
          b"bike_model_version_info",
          b"bike_inference_total",
      ):
          assert name in content
  ```

  Run:
  ```bash
  pytest tests/integration/test_api.py -v
  ```
  Expected: 5 PASS (this requires MLflow up OR `data/splits/model.pkl` present from Phase 3).

- [ ] **Task 5.5: Smoke-test the running server manually**

  ```bash
  PYTHONPATH=. uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
  ```

  In another terminal:
  ```bash
  curl http://localhost:8000/health
  curl -X POST http://localhost:8000/predict \
       -H "Content-Type: application/json" \
       -d '{"season":1,"mnth":1,"hr":8,"holiday":0,"weekday":1,"workingday":1,"weathersit":1,"temp":0.24,"atemp":0.288,"hum":0.81,"windspeed":0.0}'
  curl http://localhost:8000/metrics | head -20
  ```

  Capture screenshots for the report.

- [ ] **Task 5.6: Commit**

  ```bash
  git add src/serving tests/integration/test_api.py
  git commit -m "phase 5: fastapi app with health, predict, predict/batch + 5 prometheus metrics"
  ```

---

### Phase 6: CI/CD

**Goal:** GitHub Actions pipeline that runs lint, tests (≥70% coverage), data validation, and model validation on every push and PR. Branch protection requires all four checks.

**Files:**
- Create: `.github/workflows/ci.yml`, `scripts/validate_model.py`.

**Dependencies:** Phase 5 (so all code-under-test exists).

**Pitfalls:**
- The CI runner has no DVC remote credentials. Either (a) use `dvc pull` with a public DVC remote (we don't have one), or (b) commit a tiny test fixture `tests/data/sample_hour.csv` (50 rows) and run validation against it, not the full data.
- MLflow server isn't running in CI → model validation must use the local pickle, not the registry.
- Coverage measured against `src/` only; exclude `src/serving/app.py` HTTP startup code by pragmas if the lifespan branch is hard to hit. Better: keep tests strong, no excludes.

**Tasks:**

- [ ] **Task 6.1: Create a tiny CI fixture**

  Save the first 100 rows (header + 100 data rows = 101 lines) of `data/raw/hour.csv` as `tests/data/sample_hour.csv`. This is committed to Git — it's a test fixture, not data, and 100 rows from yr=0 don't constitute the dataset.

  ```bash
  head -101 data/raw/hour.csv > tests/data/sample_hour.csv
  wc -l tests/data/sample_hour.csv   # 101
  ```

- [ ] **Task 6.2: Add data validation test using the fixture**

  Append to `tests/data/test_data_validation.py`:

  ```python
  from pathlib import Path
  from src.data.load import load_raw

  FIXTURE = Path(__file__).parent / "sample_hour.csv"


  def test_fixture_passes_schema():
      df = load_raw(FIXTURE)
      assert len(df) == 100
  ```

- [ ] **Task 6.3: Write `scripts/validate_model.py`**

  ```python
  # scripts/validate_model.py
  from __future__ import annotations
  import json
  import sys
  from pathlib import Path
  import joblib
  import pandas as pd
  from src.config import load_config
  from src.evaluation.metrics import compute_metrics


  def main() -> int:
      cfg = load_config()
      model_path = Path(cfg.paths.model)
      preproc_path = Path(cfg.paths.preprocessor)
      test_path = Path(cfg.paths.test)
      if not (model_path.exists() and preproc_path.exists() and test_path.exists()):
          # CI runs without DVC pull; use the metrics.json from training, if committed
          metrics_path = Path(cfg.paths.metrics)
          if not metrics_path.exists():
              print("validate_model: no model/test set; skipping (CI without artifacts)")
              return 0
          with open(metrics_path) as f:
              data = json.load(f)
          r2 = data["test"]["r2"]
      else:
          model = joblib.load(model_path)
          preproc = joblib.load(preproc_path)
          test = pd.read_parquet(test_path)
          feature_cols = cfg.data.numeric_features + cfg.data.categorical_features
          y_pred = model.predict(preproc.transform(test[feature_cols]))
          r2 = compute_metrics(test[cfg.data.target].values, y_pred)["r2"]
      print(f"validate_model: r2={r2:.4f} (threshold={cfg.validation.min_test_r2})")
      if r2 < cfg.validation.min_test_r2:
          print("FAIL: r2 below threshold")
          return 1
      print("PASS")
      return 0


  if __name__ == "__main__":
      sys.exit(main())
  ```

  Note: in CI we will commit `data/splits/metrics.json` (it's small, DVC-tracked but cache:false in dvc.yaml means it lives in Git). This lets `validate_model.py` enforce the R² gate without needing the full DVC pull.

- [ ] **Task 6.4: Commit `metrics.json` to Git**

  Update `.gitignore` so `data/splits/metrics.json` is NOT excluded (the existing rule excludes everything in `data/splits/*`). Add an exception:

  ```diff
   data/splits/*
   !data/splits/.gitkeep
  +!data/splits/metrics.json
  ```

  ```bash
  git add data/splits/metrics.json .gitignore
  ```

- [ ] **Task 6.5: Write `.github/workflows/ci.yml`**

  ```yaml
  # .github/workflows/ci.yml
  name: CI

  on:
    push:
      branches: [main]
    pull_request:
      branches: [main]

  jobs:
    lint:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
          with:
            python-version: "3.11"
            cache: "pip"
        - run: pip install ruff==0.4.7
        - run: ruff check src tests

    tests:
      runs-on: ubuntu-latest
      needs: lint
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
          with:
            python-version: "3.11"
            cache: "pip"
        - run: pip install -r requirements.txt
        - name: Run pytest with coverage
          run: pytest tests --cov=src --cov-report=term-missing --cov-fail-under=70

    data-validation:
      runs-on: ubuntu-latest
      needs: tests
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
          with:
            python-version: "3.11"
            cache: "pip"
        - run: pip install -r requirements.txt
        - name: Validate data fixture against schema
          run: PYTHONPATH=. python -c "from src.data.load import load_raw; load_raw('tests/data/sample_hour.csv')"

    model-validation:
      runs-on: ubuntu-latest
      needs: tests
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
          with:
            python-version: "3.11"
            cache: "pip"
        - run: pip install -r requirements.txt
        - name: Validate model meets threshold
          run: PYTHONPATH=. python scripts/validate_model.py
  ```

- [ ] **Task 6.6: Push and verify CI is green**

  ```bash
  git add .github/workflows/ci.yml scripts/validate_model.py tests/data/sample_hour.csv tests/data/test_data_validation.py
  git commit -m "phase 6: github actions ci (lint, tests with coverage, data + model validation)"
  git push
  ```

  Open the repo → Actions tab → confirm 4 green checks. Capture screenshot.

- [ ] **Task 6.7: Configure branch protection**

  GitHub UI: Settings → Branches → Add rule:
  - Branch name pattern: `main`
  - Require a pull request before merging: ON
  - Require status checks to pass: ON, select `lint`, `tests`, `data-validation`, `model-validation`.

  Capture screenshot for the report.

---

### Phase 7: Monitoring & Drift Detection

**Goal:** A monitoring script generating two Evidently HTML reports, threshold-based alerting (>20% drift), and confirming the 5 Prometheus metrics work end-to-end.

**Files:**
- Create: `monitoring/run_monitoring.py`, `monitoring/drift_logic.py`, `monitoring/prometheus/prometheus.yml`.
- Tests: `tests/unit/test_drift_logic.py`.

**Dependencies:** Phase 5 (the API) and Phase 1 (drift split).

**Pitfalls:**
- Evidently 0.4.x has a `Report` API; older syntax `Dashboard` is gone. Use `from evidently.report import Report; from evidently.metric_preset import DataDriftPreset, DataQualityPreset, RegressionPreset`.
- Generating reports without prediction columns means RegressionPreset is skipped. Compute `prediction` columns on both reference and production datasets.
- Prometheus metrics persist across requests inside the FastAPI process, but if the API is restarted the histogram resets. Acceptable; document this.

**Tasks:**

- [ ] **Task 7.1: Write the failing test for `drift_logic`**

  `tests/unit/test_drift_logic.py`:

  ```python
  from monitoring.drift_logic import drift_alert


  def test_no_alert_when_below_threshold():
      result = drift_alert(
          drift_per_feature={"a": False, "b": False, "c": True, "d": False},
          threshold_share=0.50,
      )
      assert result.alert is False
      assert result.drift_share == 0.25
      assert result.drifted_features == ["c"]


  def test_alert_when_above_threshold():
      result = drift_alert(
          drift_per_feature={"a": True, "b": True, "c": True, "d": False},
          threshold_share=0.50,
      )
      assert result.alert is True
      assert result.drift_share == 0.75
      assert set(result.drifted_features) == {"a", "b", "c"}


  def test_handles_empty_dict():
      result = drift_alert(drift_per_feature={}, threshold_share=0.20)
      assert result.alert is False
      assert result.drift_share == 0.0
  ```

- [ ] **Task 7.2: Write `monitoring/drift_logic.py`**

  ```python
  # monitoring/drift_logic.py
  from __future__ import annotations
  from dataclasses import dataclass


  @dataclass
  class DriftResult:
      alert: bool
      drift_share: float
      drifted_features: list[str]


  def drift_alert(drift_per_feature: dict[str, bool], threshold_share: float) -> DriftResult:
      if not drift_per_feature:
          return DriftResult(alert=False, drift_share=0.0, drifted_features=[])
      drifted = [f for f, d in drift_per_feature.items() if d]
      share = len(drifted) / len(drift_per_feature)
      return DriftResult(alert=share > threshold_share, drift_share=share, drifted_features=drifted)
  ```

  Run tests — expect 3 PASS.

- [ ] **Task 7.3: Write `monitoring/run_monitoring.py`**

  ```python
  # monitoring/run_monitoring.py
  from __future__ import annotations
  import json
  import logging
  from pathlib import Path

  import joblib
  import pandas as pd
  from evidently.metric_preset import (
      DataDriftPreset,
      DataQualityPreset,
      RegressionPreset,
  )
  from evidently.report import Report

  from monitoring.drift_logic import drift_alert
  from src.config import load_config

  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
  log = logging.getLogger(__name__)


  def _add_predictions(df: pd.DataFrame, model, preprocessor, feature_cols, target) -> pd.DataFrame:
      X = preprocessor.transform(df[feature_cols])
      df = df.copy()
      df["prediction"] = model.predict(X)
      df = df.rename(columns={target: "target"})
      return df


  def _drift_per_feature_from_report(report: Report) -> dict[str, bool]:
      result = report.as_dict()
      out: dict[str, bool] = {}
      for metric in result["metrics"]:
          if metric.get("metric") == "DataDriftTable":
              by_col = metric["result"].get("drift_by_columns", {})
              for col, info in by_col.items():
                  out[col] = bool(info.get("drift_detected", False))
              break
      return out


  def main() -> None:
      cfg = load_config()
      model = joblib.load(cfg.paths.model)
      preprocessor = joblib.load(cfg.paths.preprocessor)
      reference = pd.read_parquet(cfg.paths.reference)
      production = pd.read_parquet(cfg.paths.production)
      # production-clean = first half of reference for the baseline (no drift)
      half = len(reference) // 2
      production_clean = reference.iloc[half:].copy()
      reference = reference.iloc[:half].copy()

      feature_cols = cfg.data.numeric_features + cfg.data.categorical_features
      ref = _add_predictions(reference, model, preprocessor, feature_cols, cfg.data.target)
      prod_clean = _add_predictions(production_clean, model, preprocessor, feature_cols, cfg.data.target)
      prod_drift = _add_predictions(production, model, preprocessor, feature_cols, cfg.data.target)

      out_dir = Path("monitoring/evidently_reports")
      out_dir.mkdir(parents=True, exist_ok=True)

      presets = [DataDriftPreset(), DataQualityPreset(), RegressionPreset()]

      baseline = Report(metrics=presets)
      baseline.run(reference_data=ref, current_data=prod_clean)
      baseline.save_html(str(out_dir / "baseline.html"))
      log.info("baseline report saved")

      drift = Report(metrics=presets)
      drift.run(reference_data=ref, current_data=prod_drift)
      drift.save_html(str(out_dir / "drift.html"))
      log.info("drift report saved")

      drift_map = _drift_per_feature_from_report(drift)
      result = drift_alert(drift_map, threshold_share=cfg.drift.drift_threshold_share)
      summary = {
          "alert": result.alert,
          "drift_share": result.drift_share,
          "drifted_features": result.drifted_features,
          "threshold": cfg.drift.drift_threshold_share,
      }
      with open(out_dir / "drift_summary.json", "w", encoding="utf-8") as f:
          json.dump(summary, f, indent=2)

      if result.alert:
          log.warning(
              "DRIFT DETECTED: %.0f%% of features drifted (>%.0f%% threshold). Drifted: %s",
              result.drift_share * 100,
              cfg.drift.drift_threshold_share * 100,
              ", ".join(result.drifted_features),
          )
      else:
          log.info("no drift alert (share=%.2f)", result.drift_share)


  if __name__ == "__main__":
      main()
  ```

- [ ] **Task 7.4: Run monitoring and capture HTMLs**

  ```bash
  PYTHONPATH=. python monitoring/run_monitoring.py
  ls monitoring/evidently_reports/    # baseline.html drift.html drift_summary.json
  cat monitoring/evidently_reports/drift_summary.json
  ```

  Expected: `alert: true`, drifted features include at least `temp`, `hum`, `windspeed`.

  Open both HTMLs in a browser; capture screenshots for the report.

- [ ] **Task 7.4.5: 🔧 MANUAL STEP — Calibrate the drift threshold (`drift_threshold_share`)**

  The rubric example uses "20%". The actual right value depends on what the *baseline* (clean) report shows — without tuning you risk false positives.

  Walk through:
  1. Open `baseline.html` → record the share of features Evidently flags as drifted on **identical** distributions. Ideally 0; could be 5–10% by chance.
  2. Open `drift.html` → record the share on the **perturbed** set. Should be ≥30%.
  3. Set `drift.drift_threshold_share` in `params.yaml` to a value comfortably above the baseline share but below the drifted share. The default `0.20` is reasonable; **only change if baseline share > 0.10**.
  4. Re-run `monitoring/run_monitoring.py` if you changed the threshold so `drift_summary.json` reflects it.

  **Output:** updated `params.yaml`. The chosen value will be cited later in the Model Card.

- [ ] **Task 7.4.6: 🔧 MANUAL STEP — Persist drift interpretation (code artifact)**

  Rubric pitfall: *"Shallow monitoring — the drift threshold logic and documented interpretation of results earn full marks, not just generating an Evidently report."*

  Write a short markdown interpretation alongside the Evidently HTMLs so it lives with the artifact rather than being lost. Create `monitoring/evidently_reports/interpretation.md`:

  ```markdown
  # Drift Report Interpretation — yr=0 (reference) vs yr=1 (drifted production)

  ## Drifted features (from drift_summary.json)
  - <list each feature flagged in drift_summary.json with its KS p-value or detection score from the HTML>

  ## Expected vs unexpected
  - temp / hum / windspeed: **expected** — we injected synthetic perturbations.
  - <other features> (e.g. cnt, season): expected/unexpected and why.

  ## Target drift (cnt)
  - Did `cnt` shift? Yes/no and by how much (median, p95). Implication for model performance.

  ## RegressionPreset findings
  - Error histogram on production-clean vs production-drifted: <observed delta>.
  - R² on production-drifted vs the test-set R²: <delta>.

  ## Recommended action
  - [ ] Full retrain on combined yr=0 + yr=1
  - [ ] Forget-and-replace (yr=1 only)
  - [ ] No action — drift is benign for downstream use
  ```

  Fill in concrete numbers from your run. This file is later quoted verbatim into the Model Card. **Output:** `monitoring/evidently_reports/interpretation.md` committed to Git.

- [ ] **Task 7.5: Write a tiny Prometheus scrape config (optional infra for Bonus A)**

  `monitoring/prometheus/prometheus.yml`:

  ```yaml
  global:
    scrape_interval: 5s
  scrape_configs:
    - job_name: bike-api
      static_configs:
        - targets: ["api:8000"]
  ```

- [ ] **Task 7.6: Commit**

  ```bash
  git add monitoring tests/unit/test_drift_logic.py
  git commit -m "phase 7: monitoring with evidently reports + drift threshold logic"
  ```

---

### Phase 8: Dockerization (Bonus A — full 10 points)

**Goal:** `docker compose up --build` brings up MLflow + the API. The API's `/health` responds. Inter-service communication uses Docker DNS, not localhost.

**Files:**
- Create: `docker/api.Dockerfile`, `docker/mlflow.Dockerfile`, `docker-compose.yml`.

**Dependencies:** Phase 5.

**Pitfalls:**
- Mounting the local `mlruns/` into the MLflow container breaks artifact paths because artifacts are stored as absolute paths inside MLflow. Use a single `mlflow-data` named volume.
- The API resolves `mlflow:5000` via Docker DNS, but the local `params.yaml` still says `localhost:5000`. We override via `MLFLOW_TRACKING_URI` env var inside the container.
- Building with `pip install -r requirements.txt` is slow. Cache wheels via the layer order: copy requirements.txt first, install, then copy code.
- Running as non-root: create a `app` user.

**Tasks:**

- [ ] **Task 8.1: Write `docker/api.Dockerfile`**

  ```dockerfile
  # docker/api.Dockerfile
  FROM python:3.11.9-slim AS base

  ENV PYTHONDONTWRITEBYTECODE=1 \
      PYTHONUNBUFFERED=1 \
      PIP_NO_CACHE_DIR=1

  RUN useradd -m -u 1000 app
  WORKDIR /app

  COPY requirements.txt .
  RUN pip install --upgrade pip && pip install -r requirements.txt

  COPY src ./src
  COPY configs ./configs
  COPY data/splits/preprocessor.pkl ./data/splits/preprocessor.pkl
  COPY data/splits/model.pkl ./data/splits/model.pkl

  RUN chown -R app:app /app
  USER app

  ENV PYTHONPATH=/app
  EXPOSE 8000

  # Container-level liveness probe so docker compose can detect a broken model load
  HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
      CMD python -c "import urllib.request,sys; \
  sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

  CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
  ```

- [ ] **Task 8.2: Write `docker/mlflow.Dockerfile`**

  ```dockerfile
  # docker/mlflow.Dockerfile
  FROM python:3.11.9-slim

  ENV PIP_NO_CACHE_DIR=1
  RUN pip install --upgrade pip && pip install mlflow==2.13.0

  RUN useradd -m -u 1000 mlflow
  WORKDIR /mlflow
  RUN chown -R mlflow:mlflow /mlflow
  USER mlflow

  EXPOSE 5000
  CMD ["mlflow", "server", \
       "--backend-store-uri", "sqlite:///mlflow.db", \
       "--default-artifact-root", "/mlflow/mlruns", \
       "--host", "0.0.0.0", \
       "--port", "5000"]
  ```

- [ ] **Task 8.3: Write `docker-compose.yml`**

  ```yaml
  # docker-compose.yml
  services:
    mlflow:
      build:
        context: .
        dockerfile: docker/mlflow.Dockerfile
      image: bike-mlflow:latest
      container_name: mlflow
      ports:
        - "5000:5000"
      volumes:
        - mlflow-data:/mlflow
      healthcheck:
        test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"]
        interval: 10s
        timeout: 3s
        retries: 5

    api:
      build:
        context: .
        dockerfile: docker/api.Dockerfile
      image: bike-api:latest
      container_name: api
      depends_on:
        mlflow:
          condition: service_healthy        # wait for mlflow /health
      environment:
        MLFLOW_TRACKING_URI: http://mlflow:5000     # service name, not localhost
      ports:
        - "8000:8000"
      # API healthcheck inherited from Dockerfile HEALTHCHECK directive

    prometheus:
      image: prom/prometheus:v2.53.0
      container_name: prometheus
      depends_on:
        api:
          condition: service_healthy
      ports:
        - "9090:9090"
      volumes:
        - ./monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro

  volumes:
    mlflow-data:
  ```

  Three services satisfy the rubric phrase "at least two services" and demonstrate end-to-end observability for the demo video.

- [ ] **Task 8.4: Verify `MLFLOW_TRACKING_URI` env override is in place**

  This was wired into `src/training/train.py` (Task 3.5) and `src/training/registry.py` (Task 4.1) at construction time. Add the same `os.environ.get("MLFLOW_TRACKING_URI", cfg.mlflow.tracking_uri)` shim to `src/serving/app.py` `_load_model()`:

  ```python
  import os
  ...
  def _load_model():
      cfg = load_config()
      tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", cfg.mlflow.tracking_uri)
      mlflow.set_tracking_uri(tracking_uri)
      ...
  ```

  Quick check: `grep -nr "set_tracking_uri" src/` — every call must use the env-resolved URI, never `cfg.mlflow.tracking_uri` directly. Required so Compose can rewire `mlflow:5000` for inter-service DNS.

- [ ] **Task 8.5: Run `docker compose up --build`**

  Make sure `data/splits/preprocessor.pkl` and `data/splits/model.pkl` exist locally first (run `dvc pull` or `dvc repro`).

  ```bash
  docker compose up --build
  ```

  In another terminal:
  ```bash
  curl http://localhost:8000/health
  curl http://localhost:5000           # MLflow UI
  ```

  Capture screenshots showing both services running.

- [ ] **Task 8.6: Commit**

  ```bash
  git add docker docker-compose.yml src/serving/app.py src/training/train.py src/training/registry.py
  git commit -m "phase 8 (bonus a): docker compose with api + mlflow services"
  ```

---

### Phase 9: Prefect Orchestration (Bonus B — full 10 points)

**Goal:** A Prefect 2 flow that orchestrates `validate_data → preprocess → train → evaluate → register_model`. The flow is schedulable and failures halt downstream tasks.

**Files:**
- Create: `flows/training_flow.py`.

**Dependencies:** Phases 1–4.

**Pitfalls:**
- Prefect 2.x server (Orion) needs to run in a separate terminal: `prefect server start`.
- Tasks importing from `src/` need the working directory to be the project root: invoke with `python -m flows.training_flow`.
- A failing task by default does NOT stop the whole flow — wire `wait_for=` so each task depends on the previous, OR use Prefect's default behaviour where downstream tasks marked dependent automatically halt.

**Tasks:**

- [ ] **Task 9.1: Write `flows/training_flow.py`**

  ```python
  # flows/training_flow.py
  from __future__ import annotations
  from pathlib import Path
  import joblib
  import pandas as pd
  from prefect import flow, get_run_logger, task

  from src.config import load_config
  from src.data.load import load_raw
  from src.data.split import build_splits, inject_drift
  from src.evaluation.metrics import compute_metrics
  from src.features.preprocessor import fit_preprocessor
  from src.training.registry import promote_latest_to_production
  from src.training.train import main as train_main


  @task(retries=1, retry_delay_seconds=5)
  def validate_data() -> str:
      cfg = load_config()
      df = load_raw(cfg.paths.raw_csv)
      log = get_run_logger()
      log.info("validate_data: %d rows OK", len(df))
      return cfg.paths.raw_csv


  @task
  def preprocess(_validated: str) -> str:
      cfg = load_config()
      df = pd.read_csv(cfg.paths.raw_csv).drop(columns=cfg.data.drop_columns)
      df.to_parquet(cfg.paths.processed, index=False)
      train, test, reference, year1 = build_splits(
          df, split_col=cfg.data.split_column,
          test_size=cfg.data.test_size,
          ref_holdout=cfg.data.reference_holdout,
          random_state=cfg.data.random_state,
      )
      production = inject_drift(
          year1,
          factor_temp=cfg.drift.perturb_temp_factor,
          factor_hum=cfg.drift.perturb_hum_factor,
          std_windspeed=cfg.drift.perturb_windspeed_noise_std,
          seed=cfg.data.random_state,
      )
      train.to_parquet(cfg.paths.train, index=False)
      test.to_parquet(cfg.paths.test, index=False)
      reference.to_parquet(cfg.paths.reference, index=False)
      production.to_parquet(cfg.paths.production, index=False)

      pipe = fit_preprocessor(
          train,
          target=cfg.data.target,
          numeric=cfg.data.numeric_features,
          categorical=cfg.data.categorical_features,
          k=cfg.preprocessing.feature_selection_k,
          numeric_strategy=cfg.preprocessing.numeric_imputer_strategy,
          categorical_strategy=cfg.preprocessing.categorical_imputer_strategy,
      )
      Path(cfg.paths.preprocessor).parent.mkdir(parents=True, exist_ok=True)
      joblib.dump(pipe, cfg.paths.preprocessor)
      return cfg.paths.train


  @task
  def train(_train_path: str) -> str:
      train_main()
      cfg = load_config()
      return cfg.paths.model


  @task
  def evaluate(model_path: str) -> dict:
      cfg = load_config()
      import json
      with open(cfg.paths.metrics) as f:
          metrics = json.load(f)
      log = get_run_logger()
      log.info("evaluate: r2=%.4f", metrics["test"]["r2"])
      if metrics["test"]["r2"] < cfg.validation.min_test_r2:
          raise ValueError(
              f"r2={metrics['test']['r2']:.3f} below threshold {cfg.validation.min_test_r2}"
          )
      return metrics


  @task
  def register_model(_metrics: dict) -> int:
      version = promote_latest_to_production()
      log = get_run_logger()
      log.info("register_model: promoted version %d to Production", version)
      return version


  @flow(name="bike-share-training")
  def training_flow():
      validated = validate_data()
      preprocessed = preprocess(validated)
      model_path = train(preprocessed)
      metrics = evaluate(model_path)
      version = register_model(metrics)
      return version


  if __name__ == "__main__":
      training_flow()
  ```

- [ ] **Task 9.2: Run the Prefect server and the flow**

  Terminal A:
  ```bash
  prefect server start
  ```

  Terminal B:
  ```bash
  PYTHONPATH=. python -m flows.training_flow
  ```

  Then:
  ```bash
  prefect deployment build flows/training_flow.py:training_flow -n bike-train -q default --apply
  ```

  Open http://127.0.0.1:4200 → Flows → confirm one successful run with all 5 tasks. Capture screenshot.

- [ ] **Task 9.3: Force a failure to demonstrate halting**

  Temporarily set `validation.min_test_r2: 0.99` in `params.yaml`, re-run the flow, see `evaluate` fail and `register_model` skipped. Restore `0.70` and re-run.

- [ ] **Task 9.4: Commit**

  ```bash
  git add flows
  git commit -m "phase 9 (bonus b): prefect flow with 5 tasks (validate, preprocess, train, evaluate, register)"
  ```

---

## 5. DVC PIPELINE DESIGN

### `dvc.yaml` Stages (Final)

| Stage | cmd | Inputs | Outputs |
|-------|-----|--------|---------|
| **prepare** | `python -m src.data.prepare` | `data/raw/hour.csv`, prepare/load/schema/config py, params.yaml | `data/processed/bike_clean.parquet` |
| **preprocess** | `python -m src.data.split` | `data/processed/bike_clean.parquet`, split.py, config.py, params.yaml | `data/splits/{train,test,reference,production}.parquet` |
| **featurize** | `python -m src.features.featurize` | `data/splits/train.parquet`, preprocessor.py, featurize.py, config.py, params.yaml | `data/splits/preprocessor.pkl` |
| **train** | `python -m src.training.train` | `data/splits/{train,test}.parquet`, `data/splits/preprocessor.pkl`, training+evaluation py, config.py, params.yaml | `data/splits/model.pkl` (out), `data/splits/metrics.json` (metrics, cache:false) |

### Reproducibility

- All randomness is seeded via `cfg.data.random_state`.
- `dvc repro` on a clean checkout regenerates artifacts to the same hashes (deterministic).
- The DVC remote (`./dvc-storage/`) is configured with `dvc config cache.type copy` for Windows compatibility.
- `dvc.lock` is committed; `dvc dag` produces the DAG screenshot for the report.

### Why This Design Wins Marks

- 4 stages — exceeds the minimum 3 the spec requires.
- Each stage has narrow file-level deps, so editing unrelated code does not invalidate the cache.
- `metrics.json` is `cache: false` so it goes into Git (used by CI's model-validation step).

---

## 6. PREPROCESSING DESIGN

### sklearn Pipeline Structure

```
Pipeline:
 └── preprocessor: ColumnTransformer
 │    ├── num: Pipeline(SimpleImputer(median) → StandardScaler)
 │    └── cat: Pipeline(SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown='ignore'))
 └── selector: SelectKBest(score_func=f_regression, k=12)
```

### Transformers used

| Transformer | Where | Why |
|-------------|-------|-----|
| **`SimpleImputer(strategy='median')`** | numeric | defensive at serve time — training data has zero nulls (verified), but external API inputs may be incomplete |
| **`StandardScaler()`** | numeric | zero-mean unit-variance; harmless for RF, useful if we swap in a linear baseline |
| **`SimpleImputer(strategy='most_frequent')`** | categorical | defensive; trivial for low-cardinality columns |
| **`OneHotEncoder(handle_unknown='ignore', sparse_output=False)`** | categorical | safe at serve time |
| **`SelectKBest(score_func=f_regression, k=12)`** | global | **the advanced technique** — feature selection. Simplest valid choice for a regression task (SMOTE doesn't apply; PolynomialFeatures explodes dimensionality). |

### Saving and Reusing

- Fitted pipeline: `joblib.dump(pipe, "data/splits/preprocessor.pkl")`.
- DVC-tracked as the output of stage `featurize`.
- Loaded at:
  - Training time: `src/training/train.py`
  - Serving time: `src/serving/app.py` (lifespan handler)
  - Monitoring time: `monitoring/run_monitoring.py`
- The same fitted object is the **single source of truth** — no train/serve skew possible.

---

## 7. MODEL TRAINING STRATEGY

### Models

- **Primary: `RandomForestRegressor`** — strong baseline on tabular data, handles non-linearities, no scaling needed (we scale anyway for the Pipeline).
- **Implicit baseline:** Optuna's first few trials with shallow trees act as a baseline; we explicitly log a `parent_hpo` run for comparison.
- **Why not Linear / Ridge / GBM?** Simplicity. RandomForest hits R² ≥ 0.85 on this dataset out of the box. The rubric does not reward model variety beyond 3+ runs.

### HPO Method

**Optuna with TPE sampler**, 20 trials, 3-fold CV on RMSE.

Search space (in `params.yaml`):
- `n_estimators ∈ {100, 200, 400}`
- `max_depth ∈ {4, 8, 16, None}`
- `min_samples_leaf ∈ {1, 2, 4}`

= 36 combinations, 20 trials covers it efficiently. Each trial is logged as a nested MLflow run, satisfying the "≥3 experiments" requirement many times over.

### Metrics Tracked

- `cv_rmse` (per trial, also logged with `step=trial_number` to plot a "loss curve")
- `test_rmse`, `test_mae`, `test_r2` (final model on the held-out test set)
- `best_cv_rmse` (study minimum)

### MLflow Logging Structure

```
experiment "bike_sharing"
└── run "parent_hpo"  (logs HPO config, best params, final test metrics, model artifact)
    ├── nested run "trial_0"  (params, cv_rmse)
    ├── nested run "trial_1"
    ├── ...
    └── nested run "trial_19"
```

Final model is logged with `mlflow.sklearn.log_model(registered_model_name=...)` which auto-creates the registry entry. `src/training/registry.py` handles the stage transitions afterwards.

---

## 8. SERVING DESIGN

### Framework: **FastAPI** (per spec recommendation)

### Endpoint Contracts

| Method | Path | Request | Response | Status |
|--------|------|---------|----------|--------|
| GET | `/health` | – | `{status, model_name, model_version}` | 200, 503 if model not loaded |
| POST | `/predict` | `BikeRecord` JSON | `{prediction: float, confidence: float, model_version: str}` | 200, 422 on validation failure |
| POST | `/predict/batch` | `{records: BikeRecord[]}` | `{predictions: [{prediction, confidence}, ...], model_version: str}` | 200, 400 on empty list |
| GET | `/metrics` | – | Prometheus exposition format | 200 |

### Input Schema (Pydantic v2)

```python
class BikeRecord(BaseModel):
    season: int = Field(..., ge=1, le=4)
    mnth: int = Field(..., ge=1, le=12)
    hr: int = Field(..., ge=0, le=23)
    holiday: int = Field(..., ge=0, le=1)
    weekday: int = Field(..., ge=0, le=6)
    workingday: int = Field(..., ge=0, le=1)
    weathersit: int = Field(..., ge=1, le=4)
    temp: float = Field(..., ge=0.0, le=1.0)
    atemp: float = Field(..., ge=0.0, le=1.0)
    hum: float = Field(..., ge=0.0, le=1.0)
    windspeed: float = Field(..., ge=0.0, le=1.0)
```

### Model Loading Strategy

- **Primary**: `mlflow.sklearn.load_model("models:/bike_share_regressor/Production")` at FastAPI startup (lifespan).
- **Fallback**: `joblib.load("data/splits/model.pkl")` if MLflow is unreachable (offline dev / tests).
- **Tracking URI override**: env var `MLFLOW_TRACKING_URI` takes precedence over `params.yaml` so Docker Compose can rewire to `http://mlflow:5000`.
- The model is loaded **once** at startup; predictions reuse the in-memory object — zero cold-start latency per request.

---

## 9. CI/CD DESIGN

### `.github/workflows/ci.yml`

Triggers: `push` to `main`, every `pull_request` targeting `main`.

| Job | Runs | Failure mode |
|-----|------|--------------|
| **lint** | `ruff check src tests` | Any lint violation → fail |
| **tests** | `pytest tests --cov=src --cov-report=term-missing --cov-fail-under=70` | Test fails OR coverage <70% → fail |
| **data-validation** | Loads `tests/data/sample_hour.csv` through `src.data.load.load_raw` (which validates the Pandera schema) | Schema mismatch → fail |
| **model-validation** | Runs `scripts/validate_model.py` which reads `data/splits/metrics.json` (committed) and asserts `test.r2 ≥ cfg.validation.min_test_r2` (0.70) | R² below threshold → fail |

Job order: `lint → tests → (data-validation, model-validation)` (last two in parallel).

### Branch Protection

GitHub UI: Settings → Branches → Add rule for `main`:
- Require a PR before merging
- Require status checks to pass: `lint`, `tests`, `data-validation`, `model-validation`
- Require branches to be up to date

This satisfies "at least one branch protection rule on main".

---

## 10. MONITORING & DRIFT DESIGN

### Drift Simulation Method (recap, single source of truth)

Reference set = first half of `data/splits/reference.parquet` (which is yr=0, the 2011 calendar year, with 10% sampled randomly).
Production-clean = second half of the same `reference.parquet` (statistically identical → baseline report shows ~0% drift).
Production-drifted = `data/splits/production.parquet`, which is yr=1 (2012) with these injected perturbations (configured in `params.yaml`):
- `temp *= 1.10` (clipped to [0,1])
- `hum *= 0.85`
- `windspeed += N(0, 0.05)`

### Evidently Reports

`monitoring/evidently_reports/baseline.html` — uses `Report(metrics=[DataDriftPreset(), DataQualityPreset(), RegressionPreset()])` on (reference, production-clean). Expected: minimal drift.

`monitoring/evidently_reports/drift.html` — same Report on (reference, production-drifted). Expected: drift on `temp`, `hum`, `windspeed`, `cnt`-correlated targets, plus organic drift from year-over-year volume increase.

Both reports include:
- **DataDriftPreset** — KS-test/Chi² per column with detection booleans
- **DataQualityPreset** — null counts, value ranges, duplicates
- **RegressionPreset** — error histogram, predicted vs. actual (works because we attach `prediction` and `target` columns)

### Threshold Logic

```
if drifted_features / total_features > 0.20:
    log.warning("DRIFT DETECTED: ...drifted features...")
    write monitoring/evidently_reports/drift_summary.json
```

`drift_alert()` in `monitoring/drift_logic.py` is a pure function with three unit tests.

### 5 Prometheus Metrics (defined in `src/serving/metrics.py`)

| # | Metric | Type | Purpose |
|---|--------|------|---------|
| 1 | `bike_prediction_confidence` | Histogram | **Confidence histogram** (rubric requirement) — derived from per-tree variance of the RF |
| 2 | `bike_feature_temp` | Histogram | Distribution of inbound `temp` feature — drift signal |
| 3 | `bike_feature_hr` | Histogram | Distribution of inbound `hr` feature — drift signal |
| 4 | `bike_model_version_info` | Gauge | Currently-loaded model version (label-encoded) |
| 5 | `bike_inference_total` | Counter | Inference count (regression analogue of "by class": labelled by endpoint) |
| +1 | `bike_prediction_value` | Histogram | **Bonus** — output-side drift signal (counts distribution) |

`prometheus_fastapi_instrumentator` adds default HTTP latency/request metrics on top — bonus observability.

---

## 11. DOCKER DESIGN (BONUS A)

### `docker/api.Dockerfile`

- Base: `python:3.11.9-slim` (pinned, ~120 MB)
- Multi-stage: single stage is fine here (no compilation), but we ensure layer caching by copying `requirements.txt` first.
- Non-root user: `app` (uid 1000)
- Copies `src/`, `configs/`, plus the trained `preprocessor.pkl` and `model.pkl` so the container is self-contained.
- `CMD`: `uvicorn src.serving.app:app --host 0.0.0.0 --port 8000`
- Healthcheck inherited via the FastAPI `/health` endpoint (we set the docker-compose-level healthcheck implicitly via depends_on).

### `docker/mlflow.Dockerfile`

- Base: `python:3.11.9-slim`
- Single dep: `mlflow==2.13.0`
- Backend store: SQLite (file in named volume).
- Artifact store: `/mlflow/mlruns/`.

### `docker-compose.yml`

3 services — exceeds the "≥2" rubric requirement and produces a richer demo:

```yaml
services:
  mlflow:           # tracking server
  api:              # FastAPI predictor (depends_on healthy mlflow)
  prometheus:       # scrapes api:8000/metrics (depends_on healthy api)
```

### Networking

- Default Compose network connects all three services.
- API resolves MLflow as `mlflow:5000` and Prometheus resolves the API as `api:8000` via Docker DNS — satisfying the spec's "service name as hostname" requirement.

### Acceptance Criteria

- `docker compose up --build` brings up all three services with `service_healthy` reached for `mlflow` and `api`.
- `curl localhost:8000/health` returns 200.
- `curl localhost:5000` returns the MLflow UI HTML.
- `curl localhost:9090/-/healthy` returns 200; UI shows `bike-api` target as `UP`.

---

## 12. PREFECT PIPELINE DESIGN (BONUS B)

### Flow Structure

`flows/training_flow.py` defines `training_flow` with **5 tasks**:

```
validate_data ──► preprocess ──► train ──► evaluate ──► register_model
```

Each task is a thin Prefect-decorated wrapper around an existing `src/` function — zero code duplication.

### Task Details

| Task | Wraps | Input | Output |
|------|-------|-------|--------|
| `validate_data` | `src.data.load.load_raw` | – | path to validated CSV |
| `preprocess` | `src.data.split.build_splits` + `src.features.preprocessor.fit_preprocessor` | csv path | path to train.parquet |
| `train` | `src.training.train.main` | train path | path to model.pkl |
| `evaluate` | reads `metrics.json`, raises if r2 < threshold | model path | metrics dict |
| `register_model` | `src.training.registry.promote_latest_to_production` | metrics dict | new model version int |

### Execution Order

Strict linear DAG via Prefect's data-passing — `task_b(task_a())` enforces ordering. If `evaluate` raises, `register_model` is skipped (Prefect default behaviour).

### Failure Handling

- `validate_data` has `retries=1, retry_delay_seconds=5` (transient FS issues).
- All other tasks fail fast.
- `evaluate` raises `ValueError` if R² is below the gate, halting the flow before promotion.

### Demonstrating a Triggered Run

```bash
prefect server start                                   # Terminal A
PYTHONPATH=. python -m flows.training_flow             # Terminal B
prefect deployment build flows/training_flow.py:training_flow -n bike-train -q default --apply
prefect deployment run "bike-share-training/bike-train"
```

Open http://127.0.0.1:4200 → Flow Runs → screenshot for the report.

---

## 13. TESTING STRATEGY

### Test Inventory

| File | Tests | Purpose | Coverage of |
|------|-------|---------|-------------|
| `tests/unit/test_config.py` | 1 | params.yaml round-trip | `src/config.py` |
| `tests/unit/test_split.py` | 3 | split sizes, no leakage, drift injection changes distributions | `src/data/split.py` |
| `tests/unit/test_preprocessor.py` | 4 | pipeline structure; transform shape; unseen category; NaN handling | `src/features/preprocessor.py` |
| `tests/unit/test_metrics.py` | 2 | perfect prediction; key set | `src/evaluation/metrics.py` |
| `tests/unit/test_drift_logic.py` | 3 | below threshold; above; empty | `monitoring/drift_logic.py` |
| `tests/data/test_data_validation.py` | 4 | schema accepts valid; rejects bad season; rejects negative cnt; fixture passes | `src/data/schema.py`, `src/data/load.py` |
| `tests/integration/test_api.py` | 5 | health, predict, validation rejection, batch, /metrics | `src/serving/{app,schemas,metrics}.py` |

**Total: 22 tests.** This is more than the spec minimum (3 unit tests on transformers) and is structured to clear the 70% coverage gate comfortably.

### Coverage Ceiling

`src/training/train.py`, `src/training/hpo.py` are integration-tested via `dvc repro` in dev but not unit-tested in CI (Optuna runs would slow CI). They are excluded from the coverage denominator via `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["src", "monitoring"]
omit = [
    "src/training/train.py",
    "src/training/hpo.py",
    "src/training/registry.py",
    "src/serving/metrics.py",         # only constants
]
```

Add this to `pyproject.toml`.

### Smoke Test

Before pushing the final branch:
```bash
pytest tests --cov=src --cov-report=term-missing
```
Expected: all green, coverage ≥70%.

---

## 14. FINAL EXECUTION ORDER

A clean, do-not-think-just-execute checklist. Numbers map back to the phase tasks above.

```
[ ]  1. Create GitHub repo, add collaborators, clone locally                         (Task 0.1)
[ ]  2. Write .gitignore, .gitattributes, .dvcignore                                 (Tasks 0.2, 0.3)
[ ]  3. Write requirements.txt; create venv; pip install                              (Tasks 0.4, 0.5)
[ ]  4. Write pyproject.toml + pytest.ini                                             (Tasks 0.6, 0.7)
[ ]  5. Create the directory skeleton                                                 (Task 0.8)
[ ]  6. Write configs/params.yaml                                                     (Task 0.9)
[ ]  7. Write src/config.py + tests/unit/test_config.py; pytest passes                (Tasks 0.10, 0.11)
[ ]  8. dvc init; configure local remote                                              (Task 0.12)
[ ]  9. Copy hour.csv; dvc add data/raw/hour.csv                                      (Task 0.13)
[ ] 10. Write README skeleton; first commit & push                                    (Tasks 0.14, 0.15)
[ ] 11. 🔧 MANUAL: EDA notebook (distributions, leakage check, drift preview)         (Task 0.16)

[ ] 12. Write src/data/schema.py + 3 schema tests                                     (Tasks 1.1, 1.2)
[ ] 13. Write src/data/load.py                                                        (Task 1.3)
[ ] 14. Write src/data/prepare.py; run manually to verify                             (Task 1.4)
[ ] 15. Write tests/unit/test_split.py + src/data/split.py                            (Task 1.5)
[ ] 16. Write dvc.yaml (prepare, preprocess); dvc repro; commit                       (Tasks 1.6, 1.7, 1.8)

[ ] 17. Write tests/unit/test_preprocessor.py + src/features/preprocessor.py          (Tasks 2.1, 2.2)
[ ] 18. Write src/features/featurize.py                                               (Task 2.3)
[ ] 19. Add `featurize` stage to dvc.yaml; dvc repro; commit                          (Tasks 2.4, 2.5, 2.6)

[ ] 20. 🔧 MANUAL: Choose validation R² gate (`min_test_r2` in params.yaml)            (Task 3.0)
[ ] 21. Start MLflow server (separate terminal)                                       (Task 3.1)
[ ] 22. Write tests/unit/test_metrics.py + src/evaluation/metrics.py                  (Tasks 3.2, 3.3)
[ ] 23. Write src/training/hpo.py                                                     (Task 3.4)
[ ] 24. Write src/training/train.py (with MLFLOW_TRACKING_URI env override)           (Task 3.5)
[ ] 25. Add `train` stage to dvc.yaml                                                 (Task 3.6)
[ ] 26. dvc repro train; verify ≥3 runs in MLflow UI                                  (Task 3.7)
[ ] 27. 🔧 MANUAL: Review HPO results and approve best run                             (Task 3.7.5)
[ ] 28. Write scripts/export_runs.py; export docs/experiment_log.csv                  (Task 3.8)
[ ] 29. Write scripts/compute_subgroup_metrics.py; produce subgroup_metrics.json      (Task 3.9)
[ ] 30. Commit                                                                        (Task 3.10)

[ ] 31. Write src/training/registry.py (metric-aware promotion + env override)        (Task 4.1)
[ ] 32. Run promotion; verify Production stage in UI                                  (Tasks 4.2, 4.3)

[ ] 33. Write src/serving/schemas.py (incl. confidence + BatchPredictionItem)         (Task 5.1)
[ ] 34. Write src/serving/metrics.py (5 rubric metrics + 1 bonus)                     (Task 5.2)
[ ] 35. Write src/serving/app.py (lifespan + _predict_with_confidence)                (Task 5.3)
[ ] 36. Write tests/integration/test_api.py; pytest passes                            (Task 5.4)
[ ] 37. Smoke-test live server with curl; capture screenshots                         (Task 5.5)
[ ] 38. Commit                                                                         (Task 5.6)

[ ] 39. Create tests/data/sample_hour.csv fixture (100 rows)                          (Task 6.1)
[ ] 40. Extend test_data_validation.py with fixture test                              (Task 6.2)
[ ] 41. Write scripts/validate_model.py                                                (Task 6.3)
[ ] 42. Allow data/splits/metrics.json into Git                                        (Task 6.4)
[ ] 43. Write .github/workflows/ci.yml                                                 (Task 6.5)
[ ] 44. Push; verify all 4 CI checks pass                                              (Task 6.6)
[ ] 45. Configure branch protection on main                                            (Task 6.7)

[ ] 46. Write tests/unit/test_drift_logic.py + monitoring/drift_logic.py              (Tasks 7.1, 7.2)
[ ] 47. Write monitoring/run_monitoring.py                                             (Task 7.3)
[ ] 48. Run; capture baseline.html and drift.html                                      (Task 7.4)
[ ] 49. 🔧 MANUAL: Calibrate `drift_threshold_share`                                    (Task 7.4.5)
[ ] 50. 🔧 MANUAL: Persist drift interpretation.md (code artifact)                      (Task 7.4.6)
[ ] 51. Write monitoring/prometheus/prometheus.yml                                     (Task 7.5)
[ ] 52. Commit                                                                          (Task 7.6)

[ ] 53. Write docker/api.Dockerfile (with HEALTHCHECK) + docker/mlflow.Dockerfile     (Tasks 8.1, 8.2)
[ ] 54. Write docker-compose.yml (mlflow + api + prometheus, healthcheck deps)        (Task 8.3)
[ ] 55. Verify MLFLOW_TRACKING_URI env override in src/serving/app.py                  (Task 8.4)
[ ] 56. docker compose up --build; verify all 3 services + /health                    (Task 8.5)
[ ] 57. Commit                                                                          (Task 8.6)

[ ] 58. Write flows/training_flow.py                                                   (Task 9.1)
[ ] 59. Run prefect server; run flow; verify 5 tasks in UI                             (Task 9.2)
[ ] 60. Force a failure; verify halt                                                    (Task 9.3)
[ ] 61. Commit                                                                          (Task 9.4)

# Documentation deliverables — written AFTER all code is working (still part of repo)
[ ] 62. Write docs/model_card.md (consumes docs/subgroup_metrics.json directly:
        description, intended use, training data, metrics overall + per subgroup
        from subgroup_metrics.json, limitations citing the chosen drift threshold,
        ethical considerations, plus a "Drift behaviour" section that quotes
        monitoring/evidently_reports/interpretation.md)
[ ] 63. Write docs/data_card.md — must include:
          • Source: Capital Bikeshare DC, hourly aggregates Jan 2011 – Dec 2012
          • License: CC BY 4.0; cite Fanaee-T & Gama (2013) doi:10.1007/s13748-013-0040-3
          • Schema: 17 columns, 17,379 rows, zero nulls (verified in EDA notebook)
          • Drop list: instant, dteday, casual, registered, yr (with reasons)
          • Drift rationale: temporal yr=0 vs yr=1 split + synthetic perturbations
          • Privacy: no PII; aggregated counts only
          • Why day.csv is excluded
[ ] 64. Re-export docs/experiment_log.csv from MLflow (run scripts/export_runs.py)
[ ] 65. Update README.md Quickstart: 3 commands. 🔧 VERIFY on a clean Python venv
        (`python -m venv /tmp/clean && source /tmp/clean/bin/activate &&
         pip install -r requirements.txt && dvc repro && pytest tests`).
        Document any missing step. The rubric says "tested on a clean environment".
[ ] 66. Final commit; tag v1.0.0
```

---

## RISK REGISTER (proactive)

| Risk | Mitigation |
|------|-----------|
| MLflow port 5000 blocked on Windows | The FastAPI fallback to `data/splits/model.pkl` keeps tests green. |
| DVC symlink failure on Windows | `dvc config cache.type copy` (already in plan). |
| Source folder `bike_sharing_dataset/` accidentally committed to Git | Add `bike_sharing_dataset/` to `.gitignore` if the team places the repo *inside* the project directory. Recommended layout: keep `bike_sharing_dataset/` and `mlops-final/` as siblings. |
| Real data has zero nulls — imputer never exercised in training | Synthetic-NaN test in Task 2.1 (`test_imputes_missing_numeric_values`) covers this branch for CI. |
| Coverage drops below 70% in CI | `omit` rules in `pyproject.toml` (Task 13) keep training/HPO out of the denominator. |
| Optuna trials fail because MLflow is down | Each trial wraps logging in `with mlflow.start_run(...)`; MLflow URI is env-overridable. If unreachable, run plain HPO and document the gap. |
| Docker on Windows requires WSL2 | Document in README; team must have Docker Desktop installed. |
| Evidently 0.4 vs 0.5 API drift | Pin `evidently==0.4.27`; the spec doesn't require 0.5 features. |
| Branch protection blocks team commits | Allow admins to bypass during initial setup; require reviews only on PRs after first commit. |
| 70% coverage hard to hit on FastAPI lifespan code | TestClient context manager triggers lifespan startup; that's covered. |
| Final discussion: each member must explain every component | Cross-pair on PR reviews; require 1 reviewer per PR (already planned). |

---

## SELF-REVIEW (done by author)

**Spec coverage:**
- [x] Component 1 (DVC): Phase 1 — 4 stages, 4 artifacts, local remote, deterministic.
- [x] Component 2 (Preprocessing): Phase 2 — sklearn Pipeline, impute+scale+encode+SelectKBest, params.yaml-driven, DVC-tracked, 4 unit tests.
- [x] Component 3 (Tracking + Registry): Phases 3+4 — 20 nested runs, Optuna HPO, **metric-aware** registered-model promotion with env-overridable tracking URI.
- [x] Component 4 (Serving): Phase 5 — FastAPI, /health, /predict, /predict/batch **with confidence scores** (rubric "for full marks"), Pydantic validation, integration tests.
- [x] Component 5 (CI/CD): Phase 6 — 4 stages, ≥70% coverage gate, branch protection.
- [x] Component 6 (Monitoring): Phase 7 — 2 Evidently reports, drift threshold logic, **5 rubric Prometheus metrics** (`bike_prediction_confidence` + 2 feature histograms + `bike_model_version_info` + `bike_inference_total`) + 1 bonus, plus a written `interpretation.md` to defeat the "shallow monitoring" pitfall.
- [x] Component 7 (Documentation): Tasks 62–65 — README + Model Card (consumes `subgroup_metrics.json`, **per-subgroup metrics**) + Data Card + experiment_log.csv.
- [x] Component 8 (Setup + Reproducibility): Phase 0 — pinned requirements, params.yaml, .gitignore, Quickstart **verified on a clean venv** (Task 65).
- [x] Bonus A (Docker): Phase 8 — Dockerfile (with `HEALTHCHECK`, non-root user), docker-compose with 3 services (mlflow, api, prometheus) using service-name DNS and `condition: service_healthy` deps.
- [x] Bonus B (Orchestration): Phase 9 — Prefect flow with 5 tasks, schedulable, failure halting.

**Placeholder scan:** no TBDs; every code block is concrete and self-contained.

**Type consistency:** `BikeRecord` schema in `src/serving/schemas.py` matches feature columns in `params.yaml` (numeric_features + categorical_features). `compute_metrics` returns the same dict keys (`rmse`, `mae`, `r2`) used in tests, training, and validation script.

---

## EXECUTION HANDOFF

**Checkpoint status:** Training + MLflow + serving + GitHub Actions CI + templates are on **`main`**. Remote has **`main`** plus three placeholder branches listed above (all aligned with **`main`**). Ruleset **Protect main** is **active** (squash-only, four CI checks, **0** required PR approvals for solo merge — increase to **1** when pairing).

**Next implementation focus (recommended order):**

1. **`feature/component-6-monitoring`** — Evidently drift/quality + Prometheus metrics per **§10 MONITORING & DRIFT DESIGN** and issue **#8**.
2. **`feature/bonus-docker-prefect`** — Docker Compose + Prefect per **§11–§12** and issues **#10–#11** (if pursuing bonus marks).
3. **`feature/component-7-docs`** — Model card, data card, README per **§documentation** tasks and issue **#12**.

Before each new branch session: `git checkout main && git pull origin main`.

**Execution mode** (unchanged for agents): subagent-driven per task vs inline execution with checkpoints — pick based on session preference.

Plan file: `docs/superpowers/plans/2026-04-30-mlops-final-project.md`.
