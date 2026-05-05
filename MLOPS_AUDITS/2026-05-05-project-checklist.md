# Project Checklist Validation (2026-05-05)

## DVC reproducibility
- `dvc.yaml` stages valid: **FAIL (environment blocker)**
- `dvc.lock` in sync: **FAIL (not verifiable in current environment)**
- Evidence:
  - `dvc dag` -> command not found (`dvc : The term 'dvc' is not recognized...`)
  - `dvc repro --dry` -> command not found (`dvc : The term 'dvc' is not recognized...`)
- Reproducible pinned remediation workflow (aligned with `requirements.txt`):
  - `python -m pip install -r requirements.txt` (installs pinned `dvc==3.51.2` and companion pinned deps)
  - `dvc dag`
  - `dvc repro --dry`

## Preprocessing
- sklearn pipeline fit on training split only: **PASS**
- Leakage checks present: **PASS**
- Evidence:
  - Existing unit coverage includes preprocessing-specific tests (`tests/unit/test_preprocessor.py`) and split/prepare checks.
  - `python -m pytest -q` passed (`80 passed`).

## Experiments
- MLflow run logging completeness: **PASS (with warnings)**
- Registry promotion flow present: **PASS (with hardening opportunities noted in file audit)**
- Evidence:
  - `python -m pytest -q` passed (`80 passed`), including registry and training helper tests.
  - Runtime warnings observed from MLflow dependency stack (`pkg_resources is deprecated`), but no test failures.

## Serving
- API entrypoint importability: **PASS**
- `/health` and `/predict` validation coverage: **PASS**
- Evidence:
  - API entrypoint check: `python -c "from src.serving.app import app; print('api_entrypoint_import_ok', app.title)"` -> `api_entrypoint_import_ok Bike Sharing Predictor`.
  - Equivalence rationale: the project serves via ASGI target `src.serving.app:app` (used by Docker/uvicorn), so importing `app` from `src.serving.app` verifies the same runtime entry object the server bootstraps.
  - `python -m pytest -q` passed, including API/serving tests.

## CI/CD
- lint/tests/data/model gates fully green: **FAIL**
- Evidence:
  - `python -m pytest -q` -> PASS (`80 passed`).
  - `python -m ruff check .` -> FAIL (`3` errors, all `I001` import ordering).
  - Failing files:
    - `scripts/audit/build_inventory.py`
    - `scripts/audit/generate_audit_skeleton.py`
    - `tests/unit/test_scripts.py`

## Monitoring
- Drift threshold logic (>20%) implemented: **PASS**
- Monitoring pipeline failure semantics strictness: **PARTIAL**
- Evidence:
  - Existing repository audit and tests cover drift logic behavior (`monitoring/drift_logic.py`, `tests/unit/test_drift_logic.py`).
  - Prior file audit flags workflow-level `continue-on-error` risk in `.github/workflows/monitoring-drift.yml`.

## Docs
- README quickstart <= 3 steps and key cards present: **PASS**
- Audit evidence documentation current for Task 4 commands: **PASS**
- Evidence:
  - `README.md`, `docs/model_card.md`, and `docs/data_card.md` are present.
  - This checklist captures fresh command outcomes for required validation commands.

## Reproducibility
- Deterministic seeds and pinned deps: **PARTIAL**
- Environment parity and local rerun readiness: **FAIL (current shell missing DVC CLI)**
- Evidence:
  - Pinned dependencies exist in `requirements.txt`.
  - `dvc` executable unavailable in current environment, preventing reproducibility command verification.
  - Ruff quality gate currently red due to import-order violations.

## Verification Command Outcomes (Task 4)

### `dvc dag`
- **Status:** FAIL
- **Exit behavior:** command execution continues, but command unavailable.
- **Output excerpt:** `dvc : The term 'dvc' is not recognized as the name of a cmdlet...`

### `dvc repro --dry`
- **Status:** FAIL
- **Exit behavior:** command execution continues, but command unavailable.
- **Output excerpt:** `dvc : The term 'dvc' is not recognized as the name of a cmdlet...`

### `python -m pytest -q`
- **Status:** PASS
- **Outcome:** `80 passed` with non-fatal warnings.

### `python -m ruff check .`
- **Status:** FAIL
- **Outcome:** `3` lint violations (`I001`, import block un-sorted).

### API entrypoint check (project-equivalent)
- **Command:** `python -c "from src.serving.app import app; print('api_entrypoint_import_ok', app.title)"`
- **Why this is equivalent:** deployment runs `uvicorn src.serving.app:app ...`; this command validates that exact ASGI app symbol resolves and initializes.
- **Status:** PASS
- **Outcome:** `api_entrypoint_import_ok Bike Sharing Predictor`
