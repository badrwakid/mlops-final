# Improvement Backlog (2026-05-05)

Prioritized non-blocking improvements derived from Task 3/4 evidence.  
Each item includes current state, why it matters, and an exact patch direction.

## Improvement 1: Add restart policies for local stack resilience
- Current state:
  - `docker-compose.yml` defines `api` and `prometheus` without restart policies (`docs/audits/2026-05-05-file-by-file-audit.md`, `docker-compose.yml` section).
- Recommended enhancement:
  - Add `restart: unless-stopped` for `api` and `prometheus` services in `docker-compose.yml`.
- Why it matters:
  - Demo/review environments recover poorly from transient failures.
- Exact implementation patch:
```yaml
api:
  restart: unless-stopped
prometheus:
  restart: unless-stopped
```
- Expected score impact:
  - Improves reproducibility and serving reliability confidence.
- Validation command:
  - `docker compose ps`

## Improvement 2: Guard preprocessor path explicitly at API startup
- Current state:
  - `src/serving/app.py` assumes preprocessor path exists; failure diagnostics can be unclear.
- Recommended enhancement:
  - Add explicit file-existence validation before loading `cfg.paths.preprocessor`.
- Why it matters:
  - Faster startup triage when model artifacts are incomplete.
- Exact implementation patch:
```python
from pathlib import Path

if not Path(cfg.paths.preprocessor).is_file():
    raise FileNotFoundError(f"Missing preprocessor: {cfg.paths.preprocessor}")
```
- Expected score impact:
  - Better operational clarity for serving and reproducibility checks.
- Validation command:
  - `python -c "from src.serving.app import app; print(app.title)"`

## Improvement 3: Add CI assertion for DVC graph validity when available
- Current state:
  - Checklist indicates DVC commands were not executable in the current shell (`docs/audits/2026-05-05-project-checklist.md`).
- Recommended enhancement:
  - Add a DVC graph validation step to CI after dependency installation.
- Why it matters:
  - DVC regressions may only be discovered manually.
- Exact implementation patch:
```yaml
- name: Validate DVC graph
  run: dvc dag
```
- Expected score impact:
  - Stronger DVC/compliance evidence in CI artifacts.
- Validation command:
  - `dvc dag`

## Improvement 4: Add targeted regression tests for monitoring failure semantics
- Current state:
  - `monitoring/run_monitoring.py` failure semantics are critical but not clearly pinned in dedicated tests.
- Recommended enhancement:
  - Add a unit test that asserts exit code `2` when required monitoring artifacts are missing.
- Why it matters:
  - Prevents accidental reintroduction of silent success on missing artifacts.
- Exact implementation patch:
```python
import pytest

def test_main_writes_skip_summary_when_artifacts_missing(monkeypatch, tmp_path):
    import json
    from monitoring import run_monitoring

    cfg = type(
        "Cfg",
        (),
        {
            "paths": type(
                "Paths",
                (),
                {
                    "model": str(tmp_path / "missing_model.pkl"),
                    "preprocessor": str(tmp_path / "missing_preprocessor.pkl"),
                    "reference": str(tmp_path / "missing_reference.parquet"),
                    "production": str(tmp_path / "missing_production.parquet"),
                },
            )(),
            "drift": type("Drift", (), {"drift_threshold_share": 0.2})(),
            "data": type("Data", (), {"numeric_features": [], "categorical_features": [], "target": "cnt"})(),
        },
    )()

    monkeypatch.setattr(run_monitoring, "load_config", lambda: cfg)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        run_monitoring.main()
    assert exc.value.code == 2
    payload = json.loads((tmp_path / "monitoring/evidently_reports/drift_summary.json").read_text(encoding="utf-8"))
    assert payload["reason"] == "missing_required_artifacts"
```
- Expected score impact:
  - Improves monitoring robustness and reviewer trust.
- Validation command:
  - `python -m pytest tests/unit/test_run_monitoring.py -q`

## Improvement 5: Strengthen audit/verification command log with exit codes
- Current state:
  - `docs/audits/2026-05-05-project-checklist.md` captures status and excerpts, but not standardized exit-code notation per command.
- Recommended enhancement:
  - Add explicit `Exit code` rows to every command block in the checklist.
- Why it matters:
  - Better traceability for graders and future maintainers.
- Exact implementation patch:
```markdown
### `python -m ruff check .`
- Status: FAIL
- Exit code: 1
- Output excerpt: 3 lint violations (I001 import order)
```
- Expected score impact:
  - Improves documentation quality and audit defensibility.
- Validation command:
  - `python -m ruff check .`

## Improvement 6: Add `ruff format --check` as a separate CI gate
- Current state:
  - Current CI evidence focuses on `ruff check`; formatting drift could still slip in.
- Recommended enhancement:
  - Add a separate formatting gate before test stages in `.github/workflows/ci.yml`.
- Why it matters:
  - Maintains style consistency and reduces noisy formatting-only diffs.
- Exact implementation patch:
```yaml
- name: Check formatting
  run: python -m ruff format --check .
```
- Expected score impact:
  - Improves CI/CD quality hygiene.
- Validation command:
  - `python -m ruff format --check .`

## Improvement 7: Add a smoke test for Dockerized API startup
- Current state:
  - Root Docker runtime command behavior was flagged; no explicit smoke test evidence appears in tests.
- Recommended enhancement:
  - Add an optional integration smoke test (for CI environments with Docker) that builds/runs the API image and checks `/health` on a random free port.
- Why it matters:
  - Catches container startup regressions before release without introducing fixed-port flakes.
- Exact implementation patch:
```python
import contextlib
import socket
    import subprocess
    import time
    import urllib.request as request

import pytest

@contextlib.contextmanager
def _free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    yield port

@pytest.mark.integration
@pytest.mark.skipif(
    subprocess.run(["docker", "--version"], capture_output=True).returncode != 0,
    reason="Docker not available in test environment",
)
def test_docker_image_serves_health_endpoint():
    subprocess.run(["docker", "build", "-t", "mlops-final-api-test", "."], check=True)
    with _free_port() as port:
        proc = subprocess.Popen(["docker", "run", "--rm", "-p", f"{port}:8000", "mlops-final-api-test"])
        try:
            for _ in range(30):
                try:
                    with request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
                        assert resp.status == 200
                        return
                except Exception:
                    time.sleep(1)
            raise AssertionError("API did not become healthy within 30 seconds")
        finally:
            proc.terminate()
            proc.wait(timeout=10)
```
- Example file target:
```python
# tests/integration/test_docker_smoke.py
import contextlib
import socket
import subprocess
import time
import urllib.request as request

import pytest

@contextlib.contextmanager
def _free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    yield port

@pytest.mark.integration
@pytest.mark.skipif(
    subprocess.run(["docker", "--version"], capture_output=True).returncode != 0,
    reason="Docker not available in test environment",
)
def test_docker_image_serves_health_endpoint():
    subprocess.run(["docker", "build", "-t", "mlops-final-api-test", "."], check=True)
    with _free_port() as port:
        proc = subprocess.Popen(["docker", "run", "--rm", "-p", f"{port}:8000", "mlops-final-api-test"])
        try:
            for _ in range(30):
                try:
                    with request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
                        assert resp.status == 200
                        return
                except Exception:
                    time.sleep(1)
            raise AssertionError("API did not become healthy within 30 seconds")
        finally:
            proc.terminate()
            proc.wait(timeout=10)
```
- Expected score impact:
  - Improves serving confidence and reproducibility narrative.
- Validation command:
  - `python -m pytest tests/integration/test_docker_smoke.py -q -m integration`

## Improvement 8: Add freshness metadata for generated evidence artifacts
- Current state:
  - Multiple docs/artifacts are marked as potentially stale snapshots (`docs/experiment_log.csv`, `docs/feature_scores.json`, `docs/subgroup_metrics.json`).
- Recommended enhancement:
  - Create and maintain `docs/audits/evidence_manifest.json` with generation metadata per artifact.
- Why it matters:
  - Reduces risk of outdated evidence in final submission.
- Exact implementation patch:
```json
{
  "generated_at_utc": "2026-05-05T21:00:00Z",
  "source_command": "python -m scripts.export_runs --out docs/experiment_log.csv"
}
```
- Expected score impact:
  - Improves documentation credibility and audit traceability.
- Validation command:
  - `python -m scripts.export_runs --out docs/experiment_log.csv`

## Improvement 9: Add explicit links between critical backlog IDs and scorecard blockers
- Current state:
  - Plan requires blocker references in final verdict, but no enforced ID schema yet (`docs/superpowers/plans/2026-05-05-complete-project-audit.md` Task 6).
- Recommended enhancement:
  - Add canonical `C1..C10` IDs in critical issues and reference those IDs in `docs/audits/2026-05-05-scorecard.md`.
- Why it matters:
  - Easier traceability from verdict -> issue -> file evidence.
- Exact implementation patch:
```markdown
- Blocking reasons: C1, C2, C3
- See: docs/audits/2026-05-05-critical-issues.md
```
- Expected score impact:
  - Improves final audit coherence and grading clarity.
- Validation command:
  - `rg "C[0-9]+" docs/audits/2026-05-05-scorecard.md`

## Improvement 10: Add minimal governance tests for MLflow-specific exception handling
- Current state:
  - Task 3 identified broad catches in registry and model validation paths.
- Recommended enhancement:
  - Add unit tests that inject `MlflowException` and assert explicit error behavior in registry/validation flows.
- Why it matters:
  - Ensures future edits preserve actionable error semantics.
- Exact implementation patch:
```python
import pytest

def test_registry_handles_mlflow_exception(monkeypatch):
    import mlflow
    from mlflow.exceptions import MlflowException
    from src.training import registry

    def _raise(*_args, **_kwargs):
        raise MlflowException("boom")

    monkeypatch.setattr(mlflow.sklearn, "load_model", _raise)
    with pytest.raises(RuntimeError):
        registry.evaluate_and_promote("bike-demand")
```
- Expected score impact:
  - Raises experiment/governance reliability signals.
- Validation command:
  - `python -m pytest tests/unit/test_registry.py -q`
