# File-by-file Audit (2026-05-05)

## `.dvc/.gitignore`

### Purpose
- Project file `.dvc/.gitignore` used by build, automation, or documentation.

### Line-by-line findings
- L1-L3: content in `.dvc/.gitignore` appears consistent with current repository usage; no direct defect found.

### Exact code fixes
- No patch required for `.dvc/.gitignore` in this audit pass.

## `.dvc/config`

### Purpose
- Project file `.dvc/config` used by build, automation, or documentation.

### Line-by-line findings
- L1-L6: content in `.dvc/config` appears consistent with current repository usage; no direct defect found.

### Exact code fixes
- No patch required for `.dvc/config` in this audit pass.

## `.dvcignore`

### Purpose
- Project file `.dvcignore` used by build, automation, or documentation.

### Line-by-line findings
- L1-L1: content in `.dvcignore` appears consistent with current repository usage; no direct defect found.

### Exact code fixes
- No patch required for `.dvcignore` in this audit pass.

## `.gitattributes`

### Purpose
- Project file `.gitattributes` used by build, automation, or documentation.

### Line-by-line findings
- L1-L5: content in `.gitattributes` appears consistent with current repository usage; no direct defect found.

### Exact code fixes
- No patch required for `.gitattributes` in this audit pass.

## `.github/ISSUE_TEMPLATE/bug_report.md`

### Purpose
- Markdown documentation artifact `.github/ISSUE_TEMPLATE/bug_report.md`.

### Line-by-line findings
- L1-L31: documentation text in `.github/ISSUE_TEMPLATE/bug_report.md` is consistent with current repo structure; no executable logic risk found in this file.

### Exact code fixes
- No patch required. Keep `.github/ISSUE_TEMPLATE/bug_report.md` synchronized with any changed commands/paths it documents.

## `.github/ISSUE_TEMPLATE/config.yml`

### Purpose
- Project file `.github/ISSUE_TEMPLATE/config.yml` used by build, automation, or documentation.

### Line-by-line findings
- L1-L1: configuration in `.github/ISSUE_TEMPLATE/config.yml` is syntactically valid and aligned with current pipeline wiring; no direct defect found.

### Exact code fixes
- No patch required now; review `.github/ISSUE_TEMPLATE/config.yml` whenever related pipeline/config files are modified.

## `.github/ISSUE_TEMPLATE/task.md`

### Purpose
- Markdown documentation artifact `.github/ISSUE_TEMPLATE/task.md`.

### Line-by-line findings
- L1-L35: documentation text in `.github/ISSUE_TEMPLATE/task.md` is consistent with current repo structure; no executable logic risk found in this file.

### Exact code fixes
- No patch required. Keep `.github/ISSUE_TEMPLATE/task.md` synchronized with any changed commands/paths it documents.

## `.github/pull_request_template.md`

### Purpose
- Markdown documentation artifact `.github/pull_request_template.md`.

### Line-by-line findings
- L1-L46: documentation text in `.github/pull_request_template.md` is consistent with current repo structure; no executable logic risk found in this file.

### Exact code fixes
- No patch required. Keep `.github/pull_request_template.md` synchronized with any changed commands/paths it documents.

## `.github/workflows/.gitkeep`

### Purpose
- Keeps directory `.github/workflows` present in git when empty.

### Line-by-line findings
- L1: intentionally empty marker file. Evidence: `.github/workflows` currently contains tracked files; marker is optional and kept only for directory persistence.

### Exact code fixes
- No change needed. Keep the marker file so the directory remains tracked when generated files are absent.

## `.github/workflows/ci.yml`

### Purpose
- Executes lint/test/data-validation/model-validation gates on push/PR.

### Line-by-line findings
- [HIGH] L34/L62/L97/L126: dependency installation skips hash verification, so CI trusts package index responses by name+version only; this leaves supply-chain tampering risk unmitigated.
- [MEDIUM] L66 combined with coverage omissions in `pyproject.toml` leaves core train/registry paths with weaker regression signal in CI.

### Exact code fixes
```powershell
# PowerShell (local, run when dependencies change)
pip install pip-tools
pip-compile --generate-hashes --output-file requirements-hashed.txt requirements.txt
git add requirements-hashed.txt
```
```yaml
- name: Install dependencies (hash-verified committed lock)
  run: pip install --require-hashes -r requirements-hashed.txt
```

## `.github/workflows/monitoring-drift.yml`

### Purpose
- Schedules and manually triggers drift reporting job.

### Line-by-line findings
- [HIGH] L24-L27: monitoring execution is marked `continue-on-error: true`, which can hide failed drift checks behind green workflow status.

### Exact code fixes
```yaml
- name: Run monitoring pipeline
  run: python -m monitoring.run_monitoring
```

## `.gitignore`

### Purpose
- Project file `.gitignore` used by build, automation, or documentation.

### Line-by-line findings
- L1-L55: content in `.gitignore` appears consistent with current repository usage; no direct defect found.

### Exact code fixes
- No patch required for `.gitignore` in this audit pass.

## `Dockerfile`

### Purpose
- Builds the root-level container image for serving workloads.

### Line-by-line findings
- [MEDIUM] L1-L12: image has dependency install and EXPOSE, but no default start command; direct `docker run` can fail to start service consistently.

### Exact code fixes
```dockerfile
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## `README.md`

### Purpose
- Markdown documentation artifact `README.md`.

### Line-by-line findings
- L17: local endpoint references are present; this is correct for local quickstart but should remain explicitly scoped to localhost-only usage.

### Exact code fixes
- No patch required. Keep `README.md` synchronized with any changed commands/paths it documents.

## `configs/params.yaml`

### Purpose
- Project file `configs/params.yaml` used by build, automation, or documentation.

### Line-by-line findings
- L1-L64: configuration in `configs/params.yaml` is syntactically valid and aligned with current pipeline wiring; no direct defect found.

### Exact code fixes
- No patch required now; review `configs/params.yaml` whenever related pipeline/config files are modified.

## `data/processed/.gitkeep`

### Purpose
- Keeps directory `data/processed` present in git when empty.

### Line-by-line findings
- L1: intentionally empty marker file. Evidence: `data/processed` currently contains tracked files; marker is optional and kept only for directory persistence.

### Exact code fixes
- No change needed. Keep the marker file so the directory remains tracked when generated files are absent.

## `data/raw/.gitkeep`

### Purpose
- Keeps directory `data/raw` present in git when empty.

### Line-by-line findings
- L1: intentionally empty marker file. Evidence: `data/raw` currently contains tracked files; marker is optional and kept only for directory persistence.

### Exact code fixes
- No change needed. Keep the marker file so the directory remains tracked when generated files are absent.

## `data/raw/hour.csv.dvc`

### Purpose
- DVC pointer file for `hour.csv.dvc` describing checksum/size and remote linkage.

### Line-by-line findings
- L1-L5: configuration in `data/raw/hour.csv.dvc` is syntactically valid and aligned with current pipeline wiring; no direct defect found.

### Exact code fixes
- No patch required now; review `data/raw/hour.csv.dvc` whenever related pipeline/config files are modified.

## `data/splits/.gitkeep`

### Purpose
- Keeps directory `data/splits` present in git when empty.

### Line-by-line findings
- L1: intentionally empty marker file. Evidence: `data/splits` currently contains tracked files; marker is optional and kept only for directory persistence.

### Exact code fixes
- No change needed. Keep the marker file so the directory remains tracked when generated files are absent.

## `data/splits/metrics.json`

### Purpose
- JSON artifact/config consumed by repository tooling (`data/splits/metrics.json`).

### Line-by-line findings
- L1-L18: static JSON snapshot; risk is drift from current pipeline output if producer scripts are not rerun before release.

### Exact code fixes
- No code patch needed. Regenerate `data/splits/metrics.json` from its source script/workflow before release to avoid stale evidence.

## `docker-compose.yml`

### Purpose
- Composes MLflow, API, and Prometheus services for local stack execution.

### Line-by-line findings
- L18-L42: api/prometheus do not declare restart policy; transient errors require manual recovery and reduce reproducibility of demos/checks.

### Exact code fixes
```yaml
api:
  restart: unless-stopped
prometheus:
  restart: unless-stopped
```

## `docker/.gitkeep`

### Purpose
- Keeps directory `docker` present in git when empty.

### Line-by-line findings
- L1: intentionally empty marker file. Evidence: `docker` currently contains tracked files; marker is optional and kept only for directory persistence.

### Exact code fixes
- No change needed. Keep the marker file so the directory remains tracked when generated files are absent.

## `docker/api.Dockerfile`

### Purpose
- Project file `docker/api.Dockerfile` used by build, automation, or documentation.

### Line-by-line findings
- L1-L29: content in `docker/api.Dockerfile` appears consistent with current repository usage; no direct defect found.

### Exact code fixes
- No patch required for `docker/api.Dockerfile` in this audit pass.

## `docker/mlflow.Dockerfile`

### Purpose
- Project file `docker/mlflow.Dockerfile` used by build, automation, or documentation.

### Line-by-line findings
- L1-L17: content in `docker/mlflow.Dockerfile` appears consistent with current repository usage; no direct defect found.

### Exact code fixes
- No patch required for `docker/mlflow.Dockerfile` in this audit pass.

## `docs/audits/2026-05-05-inventory.txt`

### Purpose
- Project file `docs/audits/2026-05-05-inventory.txt` used by build, automation, or documentation.

### Line-by-line findings
- L1-L122: documentation text in `docs/audits/2026-05-05-inventory.txt` is consistent with current repo structure; no executable logic risk found in this file.

### Exact code fixes
- No patch required. Keep `docs/audits/2026-05-05-inventory.txt` synchronized with any changed commands/paths it documents.

## `docs/data_card.md`

### Purpose
- Markdown documentation artifact `docs/data_card.md`.

### Line-by-line findings
- L1-L27: documentation text in `docs/data_card.md` is consistent with current repo structure; no executable logic risk found in this file.

### Exact code fixes
- No patch required. Keep `docs/data_card.md` synchronized with any changed commands/paths it documents.

## `docs/experiment_log.csv`

### Purpose
- CSV artifact/data file used in project workflows (`docs/experiment_log.csv`).

### Line-by-line findings
- L1-L65: snapshot CSV is structurally valid in audit pass; freshness depends on regenerating from the owning script/pipeline step.

### Exact code fixes
- No code patch needed. Regenerate `docs/experiment_log.csv` from its source script/workflow before release to avoid stale evidence.

## `docs/feature_scores.json`

### Purpose
- JSON artifact/config consumed by repository tooling (`docs/feature_scores.json`).

### Line-by-line findings
- L1-L109: static JSON snapshot; risk is drift from current pipeline output if producer scripts are not rerun before release.

### Exact code fixes
- No code patch needed. Regenerate `docs/feature_scores.json` from its source script/workflow before release to avoid stale evidence.

## `docs/model_card.md`

### Purpose
- Markdown documentation artifact `docs/model_card.md`.

### Line-by-line findings
- L1-L28: documentation text in `docs/model_card.md` is consistent with current repo structure; no executable logic risk found in this file.

### Exact code fixes
- No patch required. Keep `docs/model_card.md` synchronized with any changed commands/paths it documents.

## `docs/screenshots/api_health_200.png`

### Purpose
- Captures concrete evidence screenshot `api_health_200.png` for documentation and grading proof.

### Line-by-line findings
- File is a PNG evidence artifact (`docs/screenshots/api_health_200.png`) linked to documented verification proof in `README.md` and `docs/technical_report.md`; it is not machine-validated, so evidence can drift if commands are not rerun after code/config changes.

### Exact code fixes
```powershell
docker compose up --build -d
curl -sS http://localhost:8000/health
```

## `docs/screenshots/api_predict_swagger_1.png`

### Purpose
- Captures concrete evidence screenshot `api_predict_swagger_1.png` for documentation and grading proof.

### Line-by-line findings
- File is a PNG evidence artifact (`docs/screenshots/api_predict_swagger_1.png`) linked to documented verification proof in `README.md` and `docs/technical_report.md`; it is not machine-validated, so evidence can drift if commands are not rerun after code/config changes.

### Exact code fixes
```powershell
docker compose up --build -d
Start-Process "http://localhost:8000/docs"
```

## `docs/screenshots/api_predict_swagger_2.png`

### Purpose
- Captures concrete evidence screenshot `api_predict_swagger_2.png` for documentation and grading proof.

### Line-by-line findings
- File is a PNG evidence artifact (`docs/screenshots/api_predict_swagger_2.png`) linked to documented verification proof in `README.md` and `docs/technical_report.md`; it is not machine-validated, so evidence can drift if commands are not rerun after code/config changes.

### Exact code fixes
```powershell
docker compose up --build -d
Start-Process "http://localhost:8000/docs"
```

## `docs/screenshots/branch_protection_main.png`

### Purpose
- Captures concrete evidence screenshot `branch_protection_main.png` for documentation and grading proof.

### Line-by-line findings
- File is a PNG evidence artifact (`docs/screenshots/branch_protection_main.png`) linked to documented verification proof in `README.md` and `docs/technical_report.md`; it is not machine-validated, so evidence can drift if commands are not rerun after code/config changes.

### Exact code fixes
```powershell
gh repo view --json nameWithOwner -q .nameWithOwner
$repo = gh repo view --json nameWithOwner -q .nameWithOwner
gh api "repos/$repo/branches/main/protection" > docs/screenshots/branch_protection_main.json
```

## `docs/screenshots/branch_protection_main_2.png`

### Purpose
- Captures concrete evidence screenshot `branch_protection_main_2.png` for documentation and grading proof.

### Line-by-line findings
- File is a PNG evidence artifact (`docs/screenshots/branch_protection_main_2.png`) linked to documented verification proof in `README.md` and `docs/technical_report.md`; it is not machine-validated, so evidence can drift if commands are not rerun after code/config changes.

### Exact code fixes
```powershell
gh repo view --json nameWithOwner -q .nameWithOwner
$repo = gh repo view --json nameWithOwner -q .nameWithOwner
gh api "repos/$repo/branches/main/protection" > docs/screenshots/branch_protection_main_2.json
```

## `docs/screenshots/ci_green_run_1.png`

### Purpose
- Captures concrete evidence screenshot `ci_green_run_1.png` for documentation and grading proof.

### Line-by-line findings
- File is a PNG evidence artifact (`docs/screenshots/ci_green_run_1.png`) linked to documented verification proof in `README.md` and `docs/technical_report.md`; it is not machine-validated, so evidence can drift if commands are not rerun after code/config changes.

### Exact code fixes
```powershell
gh run list --limit 5
$runId = gh run list --limit 1 --json databaseId -q ".[0].databaseId"
gh run view $runId
```

## `docs/screenshots/ci_green_run_2.png`

### Purpose
- Captures concrete evidence screenshot `ci_green_run_2.png` for documentation and grading proof.

### Line-by-line findings
- File is a PNG evidence artifact (`docs/screenshots/ci_green_run_2.png`) linked to documented verification proof in `README.md` and `docs/technical_report.md`; it is not machine-validated, so evidence can drift if commands are not rerun after code/config changes.

### Exact code fixes
```powershell
gh run list --limit 5
$runId = gh run list --limit 1 --json databaseId -q ".[0].databaseId"
gh run view $runId
```

## `docs/screenshots/docker_compose_ps_healthy_1.png`

### Purpose
- Captures concrete evidence screenshot `docker_compose_ps_healthy_1.png` for documentation and grading proof.

### Line-by-line findings
- File is a PNG evidence artifact (`docs/screenshots/docker_compose_ps_healthy_1.png`) linked to documented verification proof in `README.md` and `docs/technical_report.md`; it is not machine-validated, so evidence can drift if commands are not rerun after code/config changes.

### Exact code fixes
```powershell
docker compose up --build -d
docker compose ps
```

## `docs/screenshots/docker_compose_ps_healthy_2.png`

### Purpose
- Captures concrete evidence screenshot `docker_compose_ps_healthy_2.png` for documentation and grading proof.

### Line-by-line findings
- File is a PNG evidence artifact (`docs/screenshots/docker_compose_ps_healthy_2.png`) linked to documented verification proof in `README.md` and `docs/technical_report.md`; it is not machine-validated, so evidence can drift if commands are not rerun after code/config changes.

### Exact code fixes
```powershell
docker compose ps
```

## `docs/screenshots/dvc_dag.png`

### Purpose
- Captures concrete evidence screenshot `dvc_dag.png` for documentation and grading proof.

### Line-by-line findings
- File is a PNG evidence artifact (`docs/screenshots/dvc_dag.png`) linked to documented verification proof in `README.md` and `docs/technical_report.md`; it is not machine-validated, so evidence can drift if commands are not rerun after code/config changes.

### Exact code fixes
```powershell
python scripts/render_dvc_dag_png.py
```

## `docs/screenshots/dvc_repro_deterministic.png`

### Purpose
- Captures concrete evidence screenshot `dvc_repro_deterministic.png` for documentation and grading proof.

### Line-by-line findings
- File is a PNG evidence artifact (`docs/screenshots/dvc_repro_deterministic.png`) linked to documented verification proof in `README.md` and `docs/technical_report.md`; it is not machine-validated, so evidence can drift if commands are not rerun after code/config changes.

### Exact code fixes
```powershell
dvc repro
dvc repro
```

## `docs/screenshots/mlflow_registry_production.png`

### Purpose
- Captures concrete evidence screenshot `mlflow_registry_production.png` for documentation and grading proof.

### Line-by-line findings
- File is a PNG evidence artifact (`docs/screenshots/mlflow_registry_production.png`) linked to documented verification proof in `README.md` and `docs/technical_report.md`; it is not machine-validated, so evidence can drift if commands are not rerun after code/config changes.

### Exact code fixes
```powershell
docker compose up mlflow -d
python -m src.training.registry evaluate
Start-Process "http://localhost:5000"
```

## `docs/screenshots/mlflow_runs_1.png`

### Purpose
- Captures concrete evidence screenshot `mlflow_runs_1.png` for documentation and grading proof.

### Line-by-line findings
- File is a PNG evidence artifact (`docs/screenshots/mlflow_runs_1.png`) linked to documented verification proof in `README.md` and `docs/technical_report.md`; it is not machine-validated, so evidence can drift if commands are not rerun after code/config changes.

### Exact code fixes
```powershell
docker compose up mlflow -d
python -m src.training.train
Start-Process "http://localhost:5000"
```

## `docs/screenshots/mlflow_runs_2.png`

### Purpose
- Captures concrete evidence screenshot `mlflow_runs_2.png` for documentation and grading proof.

### Line-by-line findings
- File is a PNG evidence artifact (`docs/screenshots/mlflow_runs_2.png`) linked to documented verification proof in `README.md` and `docs/technical_report.md`; it is not machine-validated, so evidence can drift if commands are not rerun after code/config changes.

### Exact code fixes
```powershell
docker compose up mlflow -d
python -m src.training.train_baseline
Start-Process "http://localhost:5000"
```

## `docs/screenshots/mlflow_runs_3.png`

### Purpose
- Captures concrete evidence screenshot `mlflow_runs_3.png` for documentation and grading proof.

### Line-by-line findings
- File is a PNG evidence artifact (`docs/screenshots/mlflow_runs_3.png`) linked to documented verification proof in `README.md` and `docs/technical_report.md`; it is not machine-validated, so evidence can drift if commands are not rerun after code/config changes.

### Exact code fixes
```powershell
python -m scripts.export_runs
```

## `docs/screenshots/mlflow_runs_4.png`

### Purpose
- Captures concrete evidence screenshot `mlflow_runs_4.png` for documentation and grading proof.

### Line-by-line findings
- File is a PNG evidence artifact (`docs/screenshots/mlflow_runs_4.png`) linked to documented verification proof in `README.md` and `docs/technical_report.md`; it is not machine-validated, so evidence can drift if commands are not rerun after code/config changes.

### Exact code fixes
```powershell
python -m scripts.export_runs --out docs/experiment_log.csv
```

## `docs/screenshots/prefect_flow_run_failed_halt.png`

### Purpose
- Captures concrete evidence screenshot `prefect_flow_run_failed_halt.png` for documentation and grading proof.

### Line-by-line findings
- File is a PNG evidence artifact (`docs/screenshots/prefect_flow_run_failed_halt.png`) linked to documented verification proof in `README.md` and `docs/technical_report.md`; it is not machine-validated, so evidence can drift if commands are not rerun after code/config changes.

### Exact code fixes
```powershell
python -m flows.training_flow
```

## `docs/screenshots/prefect_flow_run_success.png`

### Purpose
- Captures concrete evidence screenshot `prefect_flow_run_success.png` for documentation and grading proof.

### Line-by-line findings
- File is a PNG evidence artifact (`docs/screenshots/prefect_flow_run_success.png`) linked to documented verification proof in `README.md` and `docs/technical_report.md`; it is not machine-validated, so evidence can drift if commands are not rerun after code/config changes.

### Exact code fixes
```powershell
python -m flows.training_flow
```

## `docs/screenshots/prometheus_scrape.png`

### Purpose
- Captures concrete evidence screenshot `prometheus_scrape.png` for documentation and grading proof.

### Line-by-line findings
- File is a PNG evidence artifact (`docs/screenshots/prometheus_scrape.png`) linked to documented verification proof in `README.md` and `docs/technical_report.md`; it is not machine-validated, so evidence can drift if commands are not rerun after code/config changes.

### Exact code fixes
```powershell
docker compose up --build -d
curl -sS http://localhost:9090/api/v1/targets
```

## `docs/screenshots/pytest_coverage_report.png`

### Purpose
- Captures concrete evidence screenshot `pytest_coverage_report.png` for documentation and grading proof.

### Line-by-line findings
- File is a PNG evidence artifact (`docs/screenshots/pytest_coverage_report.png`) linked to documented verification proof in `README.md` and `docs/technical_report.md`; it is not machine-validated, so evidence can drift if commands are not rerun after code/config changes.

### Exact code fixes
```powershell
pytest tests/ --cov=src --cov-report=term-missing --cov-report=xml --cov-fail-under=70
```

## `docs/screenshots/quickstart_clean_install.png`

### Purpose
- Captures concrete evidence screenshot `quickstart_clean_install.png` for documentation and grading proof.

### Line-by-line findings
- File is a PNG evidence artifact (`docs/screenshots/quickstart_clean_install.png`) linked to documented verification proof in `README.md` and `docs/technical_report.md`; it is not machine-validated, so evidence can drift if commands are not rerun after code/config changes.

### Exact code fixes
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
dvc repro
```

## `docs/source_readme.txt`

### Purpose
- Project file `docs/source_readme.txt` used by build, automation, or documentation.

### Line-by-line findings
- L1-L111: this text mirrors upstream source README guidance used as documentation input; no execution logic exists, but stale content can mislead if not synchronized with `README.md`.

### Exact code fixes
- No patch required. Keep `docs/source_readme.txt` synchronized with any changed commands/paths it documents.

## `docs/subgroup_metrics.json`

### Purpose
- JSON artifact/config consumed by repository tooling (`docs/subgroup_metrics.json`).

### Line-by-line findings
- L1-L78: static JSON snapshot; risk is drift from current pipeline output if producer scripts are not rerun before release.

### Exact code fixes
- No code patch needed. Regenerate `docs/subgroup_metrics.json` from its source script/workflow before release to avoid stale evidence.

## `docs/technical_report.md`

### Purpose
- Markdown documentation artifact `docs/technical_report.md`.

### Line-by-line findings
- L66: local endpoint references are present; this is correct for local quickstart but should remain explicitly scoped to localhost-only usage.

### Exact code fixes
- No patch required. Keep `docs/technical_report.md` synchronized with any changed commands/paths it documents.

## `dvc.lock`

### Purpose
- Project file `dvc.lock` used by build, automation, or documentation.

### Line-by-line findings
- L1-L142: configuration in `dvc.lock` is syntactically valid and aligned with current pipeline wiring; no direct defect found.

### Exact code fixes
- No patch required now; review `dvc.lock` whenever related pipeline/config files are modified.

## `dvc.yaml`

### Purpose
- Declares DVC stages/dependencies/outputs for data prep through training.

### Line-by-line findings
- L38-L51: training stage dependency list omits `src/training/registry.py`; governance code updates are outside DVC invalidation graph.

### Exact code fixes
```yaml
  train:
    deps:
      - src/training/registry.py
```

## `flows/.gitkeep`

### Purpose
- Keeps directory `flows` present in git when empty.

### Line-by-line findings
- L1: intentionally empty marker file. Evidence: `flows` currently contains tracked files; marker is optional and kept only for directory persistence.

### Exact code fixes
- No change needed. Keep the marker file so the directory remains tracked when generated files are absent.

## `flows/__init__.py`

### Purpose
- Marks `flows` as an importable Python package.

### Line-by-line findings
- L1-L1: initializer code is minimal and import-safe; no mutation-at-import behavior detected in this file.

### Exact code fixes
- No patch required now; keep `flows/__init__.py` unchanged unless failing tests or upstream interface changes require edits.

## `flows/training_flow.py`

### Purpose
- Python source file implementing `def validate_data() -> str:` and related logic.

### Line-by-line findings
- L1-L126: implementation in `flows/training_flow.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `flows/training_flow.py` unchanged unless failing tests or upstream interface changes require edits.

## `monitoring/__init__.py`

### Purpose
- Marks `monitoring` as an importable Python package.

### Line-by-line findings
- L1-L1: initializer code is minimal and import-safe; no mutation-at-import behavior detected in this file.

### Exact code fixes
- No patch required now; keep `monitoring/__init__.py` unchanged unless failing tests or upstream interface changes require edits.

## `monitoring/drift_logic.py`

### Purpose
- Python source file implementing `class DriftResult:` and related logic.

### Line-by-line findings
- L1-L32: implementation in `monitoring/drift_logic.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `monitoring/drift_logic.py` unchanged unless failing tests or upstream interface changes require edits.

## `monitoring/evidently_reports/baseline.html`

### Purpose
- Project file `monitoring/evidently_reports/baseline.html` used by build, automation, or documentation.

### Line-by-line findings
- L1-L666: content in `monitoring/evidently_reports/baseline.html` appears consistent with current repository usage; no direct defect found.

### Exact code fixes
- No source-code patch in this file. Refresh `monitoring/evidently_reports/baseline.html` by rerunning its generating workflow/command when evidence changes.

## `monitoring/evidently_reports/drift.html`

### Purpose
- Project file `monitoring/evidently_reports/drift.html` used by build, automation, or documentation.

### Line-by-line findings
- L1-L666: content in `monitoring/evidently_reports/drift.html` appears consistent with current repository usage; no direct defect found.

### Exact code fixes
- No source-code patch in this file. Refresh `monitoring/evidently_reports/drift.html` by rerunning its generating workflow/command when evidence changes.

## `monitoring/evidently_reports/drift_summary.json`

### Purpose
- JSON artifact/config consumed by repository tooling (`monitoring/evidently_reports/drift_summary.json`).

### Line-by-line findings
- L1-L20: static JSON snapshot; risk is drift from current pipeline output if producer scripts are not rerun before release.

### Exact code fixes
- No code patch needed. Regenerate `monitoring/evidently_reports/drift_summary.json` from its source script/workflow before release to avoid stale evidence.

## `monitoring/evidently_reports/interpretation.md`

### Purpose
- Markdown documentation artifact `monitoring/evidently_reports/interpretation.md`.

### Line-by-line findings
- L1-L77: documentation text in `monitoring/evidently_reports/interpretation.md` is consistent with current repo structure; no executable logic risk found in this file.

### Exact code fixes
- No patch required. Keep `monitoring/evidently_reports/interpretation.md` synchronized with any changed commands/paths it documents.

## `monitoring/prometheus/prometheus.yml`

### Purpose
- Project file `monitoring/prometheus/prometheus.yml` used by build, automation, or documentation.

### Line-by-line findings
- L1-L6: configuration in `monitoring/prometheus/prometheus.yml` is syntactically valid and aligned with current pipeline wiring; no direct defect found.

### Exact code fixes
- No patch required now; review `monitoring/prometheus/prometheus.yml` whenever related pipeline/config files are modified.

## `monitoring/run_monitoring.py`

### Purpose
- Runs batch drift analysis and writes Evidently HTML/JSON outputs.

### Line-by-line findings
- [HIGH] L81-L91: missing-required-artifacts path previously returned success, making scheduler health and monitoring correctness ambiguous.

### Exact code fixes
```python
if missing:
    summary = {
        "skipped": True,
        "reason": "missing_required_artifacts",
        "missing": missing,
        "hint": "Run `dvc pull` or `dvc repro` before monitoring.",
    }
    with open(out_dir / "drift_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.warning("monitoring skipped: missing artifacts: %s", ", ".join(missing))
    raise SystemExit(2)
```

## `notebooks/01_eda.ipynb`

### Purpose
- Contains exploratory notebook for feature/data inspection used during project development.

### Line-by-line findings
- L1-L202: content in `notebooks/01_eda.ipynb` appears consistent with current repository usage; no direct defect found.

### Exact code fixes
- No source-code patch in this file. Refresh `notebooks/01_eda.ipynb` by rerunning its generating workflow/command when evidence changes.

## `pyproject.toml`

### Purpose
- Central config for Ruff and coverage behavior.

### Line-by-line findings
- L20-L25: coverage omit excludes train/hpo/registry/serving metrics modules, reducing test signal on critical paths.

### Exact code fixes
```toml
[tool.coverage.run]
source = ["src", "monitoring"]
omit = []
```

## `pytest.ini`

### Purpose
- Project file `pytest.ini` used by build, automation, or documentation.

### Line-by-line findings
- L1-L8: configuration in `pytest.ini` is syntactically valid and aligned with current pipeline wiring; no direct defect found.

### Exact code fixes
- No patch required now; review `pytest.ini` whenever related pipeline/config files are modified.

## `requirements.txt`

### Purpose
- Pins project dependencies used by local/dev/CI environments.

### Line-by-line findings
- [HIGH] L1-L51: versions are pinned but unhashed; integrity is not cryptographically enforced during install.

### Exact code fixes
```powershell
# PowerShell (local, whenever requirements.txt changes)
pip install pip-tools
pip-compile --generate-hashes --output-file requirements-hashed.txt requirements.txt
git add requirements-hashed.txt
```
- For reproducible local verification (including DVC), bootstrap from pinned requirements instead of ad-hoc installs:
```powershell
python -m pip install -r requirements.txt
```
- Update CI install steps to use the committed file:
```yaml
- name: Install dependencies (hash-verified committed lock)
  run: pip install --require-hashes -r requirements-hashed.txt
```

## `scripts/audit/build_inventory.py`


### Purpose
- Python source file implementing `def _is_excluded(path: Path, root: Path) -> bool:` and related logic.

### Line-by-line findings
- [MEDIUM] L1-L4: `python -m ruff check .` reports `I001` (import block un-sorted/un-formatted), causing CI lint gate failure for this file.

### Exact code fixes
```powershell
python -m ruff check scripts/audit/build_inventory.py --fix
```

## `scripts/audit/generate_audit_skeleton.py`

### Purpose
- Python source file implementing audit skeleton rendering and completeness verification logic.

### Line-by-line findings
- [MEDIUM] L1-L4: `python -m ruff check .` reports `I001` (import block un-sorted/un-formatted), causing CI lint gate failure for this file.

### Exact code fixes
```powershell
python -m ruff check scripts/audit/generate_audit_skeleton.py --fix
```

## `scripts/bootstrap_github_project.ps1`

### Purpose
- Project file `scripts/bootstrap_github_project.ps1` used by build, automation, or documentation.

### Line-by-line findings
- L1-L244: content in `scripts/bootstrap_github_project.ps1` appears consistent with current repository usage; no direct defect found.

### Exact code fixes
- No patch required for `scripts/bootstrap_github_project.ps1` in this audit pass.

## `scripts/compute_subgroup_metrics.py`

### Purpose
- Python source file implementing `def feature_columns(cfg) -> list[str]:` and related logic.

### Line-by-line findings
- L1-L81: implementation in `scripts/compute_subgroup_metrics.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `scripts/compute_subgroup_metrics.py` unchanged unless failing tests or upstream interface changes require edits.

## `scripts/export_runs.py`

### Purpose
- Python source file implementing `def resolve_tracking_uri(configured_uri: str) -> str:` and related logic.

### Line-by-line findings
- L1-L73: implementation in `scripts/export_runs.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `scripts/export_runs.py` unchanged unless failing tests or upstream interface changes require edits.

## `scripts/render_dvc_dag_png.py`

### Purpose
- Python source file implementing `def main() -> None:` and related logic.

### Line-by-line findings
- L1-L74: script builds a DAG image from current `dvc.yaml`; output is reproducible when DVC graph is unchanged. Risk is stale `docs/screenshots/dvc_dag.png` if this script is not rerun after pipeline edits.

### Exact code fixes
- No patch required now; keep `scripts/render_dvc_dag_png.py` unchanged unless failing tests or upstream interface changes require edits.

## `scripts/validate_model.py`

### Purpose
- Python source file implementing `def _load_rmse(metrics_path: Path) -> float:` and related logic.

### Line-by-line findings
- [MEDIUM] L47: broad `except Exception` in registry-load path blurs root cause classes and slows CI incident triage.

### Exact code fixes
```python
from mlflow.exceptions import MlflowException

try:
    mlflow.sklearn.load_model(uri)
except MlflowException as exc:
    print(f"ERROR: could not load registry model {uri}: {exc}", file=sys.stderr)
    return 1
```


## `src/__init__.py`

### Purpose
- Marks `src` as an importable Python package.

### Line-by-line findings
- L1-L1: initializer code is minimal and import-safe; no mutation-at-import behavior detected in this file.

### Exact code fixes
- No patch required now; keep `src/__init__.py` unchanged unless failing tests or upstream interface changes require edits.

## `src/config.py`

### Purpose
- Python source file implementing `class PathsCfg(BaseModel):` and related logic.

### Line-by-line findings
- L1-L89: implementation in `src/config.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `src/config.py` unchanged unless failing tests or upstream interface changes require edits.

## `src/data/__init__.py`

### Purpose
- Marks `src/data` as an importable Python package.

### Line-by-line findings
- L1-L1: initializer code is minimal and import-safe; no mutation-at-import behavior detected in this file.

### Exact code fixes
- No patch required now; keep `src/data/__init__.py` unchanged unless failing tests or upstream interface changes require edits.

## `src/data/columns.py`

### Purpose
- Python source file implementing `def drop_configured_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:` and related logic.

### Line-by-line findings
- L1-L11: implementation in `src/data/columns.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `src/data/columns.py` unchanged unless failing tests or upstream interface changes require edits.

## `src/data/load.py`

### Purpose
- Python source file implementing `def load_raw(path: str | Path) -> pd.DataFrame:` and related logic.

### Line-by-line findings
- L1-L12: implementation in `src/data/load.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `src/data/load.py` unchanged unless failing tests or upstream interface changes require edits.

## `src/data/prepare.py`

### Purpose
- Python source file implementing `def main() -> None:` and related logic.

### Line-by-line findings
- L1-L20: implementation in `src/data/prepare.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `src/data/prepare.py` unchanged unless failing tests or upstream interface changes require edits.

## `src/data/schema.py`

### Purpose
- Python source file implementing project logic/scripts/tests.

### Line-by-line findings
- L1-L26: implementation in `src/data/schema.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `src/data/schema.py` unchanged unless failing tests or upstream interface changes require edits.

## `src/data/split.py`

### Purpose
- Python source file implementing `def inject_drift(` and related logic.

### Line-by-line findings
- L1-L84: implementation in `src/data/split.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `src/data/split.py` unchanged unless failing tests or upstream interface changes require edits.

## `src/evaluation/__init__.py`

### Purpose
- Marks `src/evaluation` as an importable Python package.

### Line-by-line findings
- L1-L1: initializer code is minimal and import-safe; no mutation-at-import behavior detected in this file.

### Exact code fixes
- No patch required now; keep `src/evaluation/__init__.py` unchanged unless failing tests or upstream interface changes require edits.

## `src/evaluation/metrics.py`

### Purpose
- Python source file implementing `def compute_metrics(y_true, y_pred) -> dict[str, float]:` and related logic.

### Line-by-line findings
- L1-L12: implementation in `src/evaluation/metrics.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `src/evaluation/metrics.py` unchanged unless failing tests or upstream interface changes require edits.

## `src/features/__init__.py`

### Purpose
- Marks `src/features` as an importable Python package.

### Line-by-line findings
- L1-L1: initializer code is minimal and import-safe; no mutation-at-import behavior detected in this file.

### Exact code fixes
- No patch required now; keep `src/features/__init__.py` unchanged unless failing tests or upstream interface changes require edits.

## `src/features/featurize.py`

### Purpose
- Python source file implementing `def main() -> None:` and related logic.

### Line-by-line findings
- L1-L38: implementation in `src/features/featurize.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `src/features/featurize.py` unchanged unless failing tests or upstream interface changes require edits.

## `src/features/preprocessor.py`

### Purpose
- Python source file implementing `def _effective_numeric(numeric: list[str], cyclical_hr_mnth: bool) -> list[str]:` and related logic.

### Line-by-line findings
- L1-L149: implementation in `src/features/preprocessor.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `src/features/preprocessor.py` unchanged unless failing tests or upstream interface changes require edits.

## `src/serving/__init__.py`

### Purpose
- Marks `src/serving` as an importable Python package.

### Line-by-line findings
- L1-L1: initializer code is minimal and import-safe; no mutation-at-import behavior detected in this file.

### Exact code fixes
- No patch required now; keep `src/serving/__init__.py` unchanged unless failing tests or upstream interface changes require edits.

## `src/serving/app.py`

### Purpose
- Defines FastAPI app, startup artifact loading, prediction endpoints, and Prometheus metrics.

### Line-by-line findings
- L63-L66: broad fallback from MLflow to local model on any exception can hide registry/auth/network failures and serve stale local artifacts silently.
- L72: preprocessor load assumes file exists; startup failure mode is less actionable than explicit path validation.

### Exact code fixes
```python
from pathlib import Path
from mlflow.exceptions import MlflowException

try:
    return mlflow.sklearn.load_model(model_uri), "Production"
except MlflowException as exc:
    log.warning("MLflow load failed; fallback to local model: %s", exc)
    return joblib.load(cfg.paths.model), "local"

if not Path(cfg.paths.preprocessor).is_file():
    raise FileNotFoundError(f"Missing preprocessor: {cfg.paths.preprocessor}")
```

## `src/serving/metrics.py`

### Purpose
- Python source file implementing project logic/scripts/tests.

### Line-by-line findings
- L1-L43: implementation in `src/serving/metrics.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `src/serving/metrics.py` unchanged unless failing tests or upstream interface changes require edits.

## `src/serving/schemas.py`

### Purpose
- Python source file implementing `class BikeRecord(BaseModel):` and related logic.

### Line-by-line findings
- L1-L43: implementation in `src/serving/schemas.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `src/serving/schemas.py` unchanged unless failing tests or upstream interface changes require edits.

## `src/training/__init__.py`

### Purpose
- Marks `src/training` as an importable Python package.

### Line-by-line findings
- L1-L1: initializer code is minimal and import-safe; no mutation-at-import behavior detected in this file.

### Exact code fixes
- No patch required now; keep `src/training/__init__.py` unchanged unless failing tests or upstream interface changes require edits.

## `src/training/hpo.py`

### Purpose
- Python source file implementing `class HPOResult:` and related logic.

### Line-by-line findings
- L1-L77: implementation in `src/training/hpo.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `src/training/hpo.py` unchanged unless failing tests or upstream interface changes require edits.

## `src/training/registry.py`

### Purpose
- Implements run selection, registration, promotion decisions, and CLI commands for model registry governance.

### Line-by-line findings
- L302-L303 and L542-L543 catch `Exception`; MLflow-specific failures lose type/context, making incident triage slower in registry operations.

### Exact code fixes
```python
from mlflow.exceptions import MlflowException

# Replace broad catch at the current MLflow load/registry call site
except MlflowException as exc:
    raise RuntimeError(f"Model artifact is not readable for run {candidate.run_id}: {source}") from exc
```

## `src/training/train.py`

### Purpose
- Python source file implementing `def _resolve_tracking_uri(configured_uri: str) -> str:` and related logic.

### Line-by-line findings
- L1-L199: implementation in `src/training/train.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `src/training/train.py` unchanged unless failing tests or upstream interface changes require edits.

## `src/training/train_baseline.py`

### Purpose
- Python source file implementing `def _resolve_tracking_uri(configured_uri: str) -> str:` and related logic.

### Line-by-line findings
- L1-L77: implementation in `src/training/train_baseline.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `src/training/train_baseline.py` unchanged unless failing tests or upstream interface changes require edits.

## `tests/__init__.py`

### Purpose
- Marks `tests` as an importable Python package.

### Line-by-line findings
- L1-L1: initializer code is minimal and import-safe; no mutation-at-import behavior detected in this file.

### Exact code fixes
- No patch required now; keep `tests/__init__.py` unchanged unless failing tests or upstream interface changes require edits.

## `tests/conftest.py`

### Purpose
- Python source file implementing project logic/scripts/tests.

### Line-by-line findings
- L1-L1: implementation in `tests/conftest.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `tests/conftest.py` unchanged unless failing tests or upstream interface changes require edits.

## `tests/data/__init__.py`

### Purpose
- Marks `tests/data` as an importable Python package.

### Line-by-line findings
- L1-L1: initializer code is minimal and import-safe; no mutation-at-import behavior detected in this file.

### Exact code fixes
- No patch required now; keep `tests/data/__init__.py` unchanged unless failing tests or upstream interface changes require edits.

## `tests/data/sample_hour.csv`

### Purpose
- CSV artifact/data file used in project workflows (`tests/data/sample_hour.csv`).

### Line-by-line findings
- L1-L101: snapshot CSV is structurally valid in audit pass; freshness depends on regenerating from the owning script/pipeline step.

### Exact code fixes
- No code patch needed. Regenerate `tests/data/sample_hour.csv` from its source script/workflow before release to avoid stale evidence.

## `tests/data/test_data_validation.py`

### Purpose
- Python source file implementing `def _valid_row():` and related logic.

### Line-by-line findings
- L1-L45: implementation in `tests/data/test_data_validation.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `tests/data/test_data_validation.py` unchanged unless failing tests or upstream interface changes require edits.

## `tests/integration/__init__.py`

### Purpose
- Marks `tests/integration` as an importable Python package.

### Line-by-line findings
- L1-L1: initializer code is minimal and import-safe; no mutation-at-import behavior detected in this file.

### Exact code fixes
- No patch required now; keep `tests/integration/__init__.py` unchanged unless failing tests or upstream interface changes require edits.

## `tests/scripts/audit/test_inventory.py`

### Purpose
- Python source file implementing `def test_collect_files_is_sorted_and_unique(tmp_path: Path) -> None:` and related logic.

### Line-by-line findings
- L1-L97: implementation in `tests/scripts/audit/test_inventory.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `tests/scripts/audit/test_inventory.py` unchanged unless failing tests or upstream interface changes require edits.

## `tests/test_api.py`

### Purpose
- Python source file implementing `def _fake_load_artifacts() -> LoadedModel:` and related logic.

### Line-by-line findings
- L1-L99: implementation in `tests/test_api.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `tests/test_api.py` unchanged unless failing tests or upstream interface changes require edits.

## `tests/unit/__init__.py`

### Purpose
- Marks `tests/unit` as an importable Python package.

### Line-by-line findings
- L1-L1: initializer code is minimal and import-safe; no mutation-at-import behavior detected in this file.

### Exact code fixes
- No patch required now; keep `tests/unit/__init__.py` unchanged unless failing tests or upstream interface changes require edits.

## `tests/unit/test_columns.py`

### Purpose
- Python source file implementing `def test_drop_configured_columns_removes_only_present():` and related logic.

### Line-by-line findings
- L1-L9: implementation in `tests/unit/test_columns.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `tests/unit/test_columns.py` unchanged unless failing tests or upstream interface changes require edits.

## `tests/unit/test_config.py`

### Purpose
- Python source file implementing `def test_load_config_returns_expected_keys():` and related logic.

### Line-by-line findings
- L1-L10: implementation in `tests/unit/test_config.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `tests/unit/test_config.py` unchanged unless failing tests or upstream interface changes require edits.

## `tests/unit/test_drift_logic.py`

### Purpose
- Python source file implementing `def test_no_alert_when_below_threshold():` and related logic.

### Line-by-line findings
- L1-L47: implementation in `tests/unit/test_drift_logic.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `tests/unit/test_drift_logic.py` unchanged unless failing tests or upstream interface changes require edits.

## `tests/unit/test_featurize_main.py`

### Purpose
- Python source file implementing `def _toy_train(n: int = 80) -> pd.DataFrame:` and related logic.

### Line-by-line findings
- L1-L86: implementation in `tests/unit/test_featurize_main.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `tests/unit/test_featurize_main.py` unchanged unless failing tests or upstream interface changes require edits.

## `tests/unit/test_hpo.py`

### Purpose
- Python source file implementing `def test_run_hpo_uses_seeded_search_space_and_logs_trials(monkeypatch):` and related logic.

### Line-by-line findings
- L1-L109: implementation in `tests/unit/test_hpo.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `tests/unit/test_hpo.py` unchanged unless failing tests or upstream interface changes require edits.

## `tests/unit/test_load.py`

### Purpose
- Python source file implementing `def test_load_raw_reads_and_validates_sample(tmp_path):` and related logic.

### Line-by-line findings
- L1-L15: implementation in `tests/unit/test_load.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `tests/unit/test_load.py` unchanged unless failing tests or upstream interface changes require edits.

## `tests/unit/test_metrics.py`

### Purpose
- Python source file implementing `def test_compute_metrics_returns_zero_errors_and_unit_r2_for_perfect_predictions():` and related logic.

### Line-by-line findings
- L1-L17: implementation in `tests/unit/test_metrics.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `tests/unit/test_metrics.py` unchanged unless failing tests or upstream interface changes require edits.

## `tests/unit/test_prepare_main.py`

### Purpose
- Python source file implementing `def test_prepare_main_writes_parquet(tmp_path, monkeypatch):` and related logic.

### Line-by-line findings
- L1-L32: implementation in `tests/unit/test_prepare_main.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `tests/unit/test_prepare_main.py` unchanged unless failing tests or upstream interface changes require edits.

## `tests/unit/test_preprocessor.py`

### Purpose
- Python source file implementing `def _toy_train(n=300):` and related logic.

### Line-by-line findings
- L1-L99: implementation in `tests/unit/test_preprocessor.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `tests/unit/test_preprocessor.py` unchanged unless failing tests or upstream interface changes require edits.

## `tests/unit/test_registry.py`

### Purpose
- Python source file implementing `def test_resolve_tracking_uri_prefers_environment_override(monkeypatch):` and related logic.

### Line-by-line findings
- L1-L587: implementation in `tests/unit/test_registry.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `tests/unit/test_registry.py` unchanged unless failing tests or upstream interface changes require edits.

## `tests/unit/test_scripts.py`

### Purpose
- Python source file implementing `def test_build_subgroup_payload_includes_overall_and_filters_small_groups():` and related logic.

### Line-by-line findings
- [LOW] L1-L5: `python -m ruff check .` reports `I001` (import block un-sorted/un-formatted). Functional behavior is intact, but lint quality gate fails.

### Exact code fixes
```powershell
python -m ruff check tests/unit/test_scripts.py --fix
python -m ruff check .
```

## `tests/unit/test_serving.py`

### Purpose
- Python source file implementing `def _cfg(model_path="data/splits/model.pkl"):` and related logic.

### Line-by-line findings
- L1-L63: implementation in `tests/unit/test_serving.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `tests/unit/test_serving.py` unchanged unless failing tests or upstream interface changes require edits.

## `tests/unit/test_split.py`

### Purpose
- Python source file implementing `def _toy(n=200):` and related logic.

### Line-by-line findings
- L1-L61: implementation in `tests/unit/test_split.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `tests/unit/test_split.py` unchanged unless failing tests or upstream interface changes require edits.

## `tests/unit/test_train.py`

### Purpose
- Python source file implementing `def test_resolve_tracking_uri_prefers_environment_override(monkeypatch):` and related logic.

### Line-by-line findings
- L1-L103: implementation in `tests/unit/test_train.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `tests/unit/test_train.py` unchanged unless failing tests or upstream interface changes require edits.

## `tests/unit/test_train_baseline_helpers.py`

### Purpose
- Python source file implementing `def test_resolve_tracking_uri_prefers_env(monkeypatch):` and related logic.

### Line-by-line findings
- L1-L25: implementation in `tests/unit/test_train_baseline_helpers.py` matches current call sites/tests; no concrete defect identified in this pass.

### Exact code fixes
- No patch required now; keep `tests/unit/test_train_baseline_helpers.py` unchanged unless failing tests or upstream interface changes require edits.

