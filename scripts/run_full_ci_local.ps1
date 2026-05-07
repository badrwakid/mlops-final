# Mirrors .github/workflows/ci.yml locally: ruff, pytest+coverage, data tests,
# MLflow Production registry gate (same order as CI).
# Usage (repo root):  powershell -ExecutionPolicy Bypass -File scripts/run_full_ci_local.ps1
#
# Model validation matches the "model-validation" job: MLflow on 127.0.0.1:5000, seed Production,
# then scripts/validate_model.py (unless SKIP_MODEL_VALIDATION=1 for a faster dev-only run).

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$env:PYTHONPATH = $Root

Write-Host "== 1. Ruff (lint job) ==" -ForegroundColor Cyan
python -m ruff check src/ tests/
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== 2. Pytest + coverage 70% floor (test job) ==" -ForegroundColor Cyan
python -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=xml --cov-fail-under=70
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== 3. Data validation (data-validation job) ==" -ForegroundColor Cyan
python -m pytest tests/data/ -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== 4. Artifact sanity (CI-required paths) ==" -ForegroundColor Cyan
python -c @"
from pathlib import Path
import sys
required = [
    Path('data/splits/model.pkl'),
    Path('data/splits/preprocessor.pkl'),
    Path('data/splits/metrics.json'),
    Path('data/raw/hour.csv'),
]
missing = [str(p) for p in required if not p.is_file()]
if missing:
    print('ERROR: missing required artifact(s):', ', '.join(missing))
    print('Run dvc pull or dvc repro so CI/local parity can pass.')
    sys.exit(1)
print('OK: model splits, metrics, and raw data present')
"@

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== 5. Model validation (model-validation job) ==" -ForegroundColor Cyan
if ($env:SKIP_MODEL_VALIDATION -eq "1") {
    Write-Host "SKIP_MODEL_VALIDATION=1: skipping MLflow seed and validate_model.py (not full CI parity)." -ForegroundColor Yellow
    exit 0
}

$mlflowProc = $null
$embeddedUri = "http://127.0.0.1:5000"
$exitCode = 1

if ($env:MLFLOW_TRACKING_URI) {
    Write-Host "Using existing MLFLOW_TRACKING_URI=$($env:MLFLOW_TRACKING_URI)" -ForegroundColor Cyan
} else {
    Write-Host "Starting local MLflow (127.0.0.1:5000, same flags as CI)..." -ForegroundColor Cyan
    $mlflowProc = Start-Process -FilePath "python" -ArgumentList @(
        "-m", "mlflow", "server",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--default-artifact-root", "./mlartifacts",
        "--host", "127.0.0.1",
        "--port", "5000"
    ) -WorkingDirectory $Root -PassThru -WindowStyle Hidden
    Start-Sleep -Seconds 5
    if ($mlflowProc.HasExited) {
        Write-Host "ERROR: MLflow server exited immediately; check port 5000 or logs." -ForegroundColor Red
        exit 1
    }
    $env:MLFLOW_TRACKING_URI = $embeddedUri
}

try {
    Write-Host "Seeding Production model (scripts/seed_mlflow_production.py)..." -ForegroundColor Cyan
    python scripts/seed_mlflow_production.py
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    # Full parity: validate registry load (do not inherit SKIP_MLFLOW_REGISTRY from the shell).
    Remove-Item Env:\SKIP_MLFLOW_REGISTRY -ErrorAction SilentlyContinue
    $env:REQUIRE_LOCAL_MODEL_ARTIFACTS = "1"
    python scripts/validate_model.py
    $exitCode = $LASTEXITCODE
} finally {
    if ($null -ne $mlflowProc -and -not $mlflowProc.HasExited) {
        Stop-Process -Id $mlflowProc.Id -Force -ErrorAction SilentlyContinue
    }
}

exit $exitCode
