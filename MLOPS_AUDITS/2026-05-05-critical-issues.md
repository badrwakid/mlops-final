# Critical Issues Backlog (2026-05-05)

Prioritized from Task 3 (`docs/audits/2026-05-05-file-by-file-audit.md`) and Task 4 (`docs/audits/2026-05-05-project-checklist.md`) evidence.  
Each item includes exact repository evidence, why it matters for grading/production readiness, and a concrete patch path.

## Critical 1: CI does not enforce dependency integrity
- Severity: Critical
- Blocker class: Hard blocker
- Evidence:
  - `requirements.txt:L1-L51` pins versions but has no hashes.
  - `.github/workflows/ci.yml:L34-L34`, `L62-L62`, `L97-L97`, `L126-L126` install with `pip install -r requirements.txt` (no hash enforcement).
- Why it matters:
  - Supply-chain integrity is unverified in CI and local bootstrap; reproducibility/security marks are at direct risk.
- Exact fix guidance:
  - Generate and commit a hash-locked file, then force hash verification in CI.
```powershell
pip install pip-tools
pip-compile --generate-hashes --output-file requirements-hashed.txt requirements.txt
git add requirements-hashed.txt
```
```yaml
- name: Install dependencies (hash-verified committed lock)
  run: pip install --require-hashes -r requirements-hashed.txt
```
- Validation command:
  - `python -m pip install --require-hashes -r requirements-hashed.txt`

## Critical 2: Monitoring workflow masks runtime failures
- Severity: Critical
- Blocker class: Hard blocker
- Evidence:
  - `.github/workflows/monitoring-drift.yml:L25-L26` runs drift monitoring and sets `continue-on-error: true`.
- Why it matters:
  - Drift failures can appear green in GitHub Actions, invalidating monitoring compliance evidence.
- Exact fix guidance:
```yaml
- name: Run monitoring pipeline
  run: python -m monitoring.run_monitoring
```
- Validation command:
  - `gh run list --workflow monitoring-drift.yml --limit 1`
  - `$runId = gh run list --workflow monitoring-drift.yml --limit 1 --json databaseId -q ".[0].databaseId"; gh run view $runId`

## Critical 3: DVC reproducibility is currently non-verifiable
- Severity: Critical
- Blocker class: Hard blocker
- Evidence:
  - `docs/audits/2026-05-05-project-checklist.md:L7-L8` records `dvc dag` and `dvc repro --dry` as command-not-found.
  - `docs/audits/2026-05-05-project-checklist.md:L70-L77` marks both DVC checks as FAIL.
- Why it matters:
  - Core reproducibility requirement is not currently provable in this environment, blocking submission confidence.
- Exact fix guidance:
```powershell
python -m pip install -r requirements.txt
dvc dag
dvc repro --dry
```
- Validation command:
  - `dvc repro --dry`

## Critical 4: CI quality gate is red due to import-order violations
- Severity: Critical
- Blocker class: Hard blocker
- Evidence:
  - `docs/audits/2026-05-05-project-checklist.md:L40-L44` records `ruff check` FAIL with 3 `I001` violations in `scripts/audit/build_inventory.py`, `scripts/audit/generate_audit_skeleton.py`, and `tests/unit/test_scripts.py`.
  - `docs/audits/2026-05-05-project-checklist.md:L84-L86` confirms command status FAIL and `I001` outcome.
- Why it matters:
  - CI/CD requirement explicitly fails; merge readiness and grading gates remain blocked.
- Exact fix guidance:
```powershell
python -m ruff check scripts/audit/build_inventory.py --fix
python -m ruff check scripts/audit/generate_audit_skeleton.py --fix
python -m ruff check tests/unit/test_scripts.py --fix
python -m ruff check .
```
- Validation command:
  - `python -m ruff check .`

## Critical 5: API silently falls back to local model on broad errors
- Severity: Critical
- Blocker class: Hard blocker
- Evidence:
  - `src/serving/app.py:L64-L66` catches broad exceptions around MLflow load and silently falls back to local model.
- Why it matters:
  - Registry/auth/network incidents can be hidden while stale local artifacts are served, undermining production governance.
- Exact fix guidance:
```python
from mlflow.exceptions import MlflowException

try:
    return mlflow.sklearn.load_model(model_uri), "Production"
except MlflowException as exc:
    log.warning("MLflow load failed; fallback to local model: %s", exc)
    return joblib.load(cfg.paths.model), "local"
```
- Validation command:
  - `python -m pytest tests/test_api.py -q`

## Critical 6: Monitoring script must hard-fail when artifacts are missing
- Severity: Critical
- Blocker class: Hard blocker
- Evidence:
  - `monitoring/run_monitoring.py:L80-L81` detects missing required artifacts.
  - `monitoring/run_monitoring.py:L90-L91` logs warning and returns success path instead of non-zero exit.
- Why it matters:
  - Scheduled monitoring can report false health and skip alerting.
- Exact fix guidance:
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
    raise SystemExit(2)
```
- Validation command:
  - `python -m monitoring.run_monitoring`

## Critical 7: Coverage omits critical train/serving paths
- Severity: High
- Blocker class: Policy choice (recommended hardening)
- Evidence:
  - `pyproject.toml:L18-L20` defines coverage config and an explicit `omit` list.
  - `docs/audits/2026-05-05-file-by-file-audit.md:L847-L853` documents omission of train/hpo/registry/serving paths.
- Why it matters:
  - CI can pass while critical production paths regress, reducing confidence in release safety.
- Exact fix guidance:
```toml
[tool.coverage.run]
source = ["src", "monitoring"]
omit = []
```
- Validation command:
  - `pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=70`

## Critical 8: Model validation uses broad exception handling
- Severity: High
- Blocker class: Hard blocker
- Evidence:
  - `scripts/validate_model.py:L46-L47` catches broad `Exception` around `mlflow.sklearn.load_model(uri)`.
- Why it matters:
  - CI model-validation failures lose MLflow-specific error context, slowing incident triage and remediation.
- Exact fix guidance:
```python
from mlflow.exceptions import MlflowException

try:
    mlflow.sklearn.load_model(uri)
except MlflowException as exc:
    print(f"ERROR: could not load registry model {uri}: {exc}", file=sys.stderr)
    return 1
```
- Non-blocking hardening moved to improvements:
  - Registry broad-exception cleanup is tracked as a hardening item in `docs/audits/2026-05-05-improvements.md` (Improvement 10).
- Validation command:
  - `python -m pytest tests/unit/test_scripts.py -q`

## Critical 9: DVC train stage misses registry dependency
- Severity: Medium
- Blocker class: Policy choice (repro hardening)
- Evidence:
  - `dvc.yaml:L38-L44` defines the `train` stage deps and includes `src/training/train.py` but not `src/training/registry.py`.
- Why it matters:
  - Registry/governance logic changes may not invalidate and rerun pipeline stages, reducing reproducibility confidence.
- Exact fix guidance:
```yaml
train:
  deps:
    - src/training/registry.py
```
- Validation command:
  - `dvc dag`

## Critical 10: Root Docker image lacks deterministic runtime command
- Severity: Medium
- Blocker class: Policy choice (container UX hardening)
- Evidence:
  - `Dockerfile:L1-L10` defines base image, dependency install, and `EXPOSE` but no `CMD`/`ENTRYPOINT`.
- Why it matters:
  - `docker run` behavior is less explicit for ad-hoc usage, but project `docker-compose.yml` can still run the service.
- Exact fix guidance:
```dockerfile
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
```
- Validation command:
  - `docker build -t mlops-final-api .`
  - `docker run --rm -p 8000:8000 mlops-final-api`
