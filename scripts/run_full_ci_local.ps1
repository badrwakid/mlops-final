# Mirrors .github/workflows/ci.yml locally: ruff, pytest+coverage, data tests,
# MLflow Production registry gate, Docker image build, operational drift monitoring.
# Usage (repo root):  powershell -ExecutionPolicy Bypass -File scripts/run_full_ci_local.ps1
#
# Model validation matches the "model-validation" job: MLflow is the Docker Compose
# `mlflow` service (docker/mlflow.Dockerfile), host http://127.0.0.1:5001, then
# scripts/seed_mlflow_production.py and scripts/validate_model.py (unless SKIP_MODEL_VALIDATION=1).

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
    Path('data/splits/reference.parquet'),
    Path('data/splits/production.parquet'),
    Path('data/raw/hour.csv'),
]
missing = [str(p) for p in required if not p.is_file()]
if missing:
    print('ERROR: missing required artifact(s):', ', '.join(missing))
    print('Run dvc pull or dvc repro so CI/local parity can pass.')
    sys.exit(1)
print('OK: model, drift splits, metrics, and raw data present')
"@

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== 5. Model validation (model-validation job) ==" -ForegroundColor Cyan
if ($env:SKIP_MODEL_VALIDATION -eq "1") {
    Write-Host "SKIP_MODEL_VALIDATION=1: skipping MLflow seed and validate_model.py (not full CI parity)." -ForegroundColor Yellow
    exit 0
}

$exitCode = 1
$startedCompose = $false

try {
    if ($env:MLFLOW_TRACKING_URI) {
        Write-Host "Using existing MLFLOW_TRACKING_URI=$($env:MLFLOW_TRACKING_URI)" -ForegroundColor Cyan
    } else {
        Write-Host "Starting MLflow via Docker Compose (production Dockerfile, host :5001)..." -ForegroundColor Cyan
        docker compose build mlflow
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        docker compose up -d mlflow
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        $startedCompose = $true
        $deadline = (Get-Date).AddMinutes(2)
        $ready = $false
        while ((Get-Date) -lt $deadline) {
            try {
                Invoke-WebRequest -Uri "http://127.0.0.1:5001/health" -UseBasicParsing -TimeoutSec 3 | Out-Null
                $ready = $true
                break
            } catch {
                Start-Sleep -Seconds 2
            }
        }
        if (-not $ready) {
            Write-Host "ERROR: MLflow not healthy on http://127.0.0.1:5001" -ForegroundColor Red
            docker compose logs mlflow
            exit 1
        }
        $env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
    }

    Write-Host "Seeding Production model (scripts/seed_mlflow_production.py)..." -ForegroundColor Cyan
    python scripts/seed_mlflow_production.py --experiment ci-model-validation
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Remove-Item Env:\SKIP_MLFLOW_REGISTRY -ErrorAction SilentlyContinue
    $env:REQUIRE_LOCAL_MODEL_ARTIFACTS = "1"
    python scripts/validate_model.py
    $exitCode = $LASTEXITCODE
} finally {
    if ($startedCompose) {
        Write-Host "Stopping MLflow container (docker compose down mlflow)..." -ForegroundColor Cyan
        docker compose down mlflow --remove-orphans 2>$null | Out-Null
    }
}

if ($exitCode -ne 0) { exit $exitCode }

Write-Host "== 6. Compose image build (compose-validate job) ==" -ForegroundColor Cyan
docker compose config --quiet
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
docker compose build mlflow api dashboard
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "== 7. Operational drift monitoring (monitoring-validation job) ==" -ForegroundColor Cyan
python -m monitoring.run_monitoring
exit $LASTEXITCODE
