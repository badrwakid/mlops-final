# Verifies a running stack matches production semantics (registry Production model, strict-ready API).
# Prereqs: docker compose up -d mlflow -> seed Production -> docker compose up -d api
# Usage (repo root):  powershell -ExecutionPolicy Bypass -File scripts/verify_production_stack.ps1
# Optional:  $env:API_BASE_URL='http://127.0.0.1:8000'

$ErrorActionPreference = "Stop"
$Base = $env:API_BASE_URL
if ([string]::IsNullOrWhiteSpace($Base)) {
    $Base = "http://127.0.0.1:8000"
}
$Base = $Base.TrimEnd("/")

function Get-Json($path) {
    $u = "$Base$path"
    $r = Invoke-WebRequest -Uri $u -UseBasicParsing -TimeoutSec 15
    if ($r.StatusCode -ne 200) { throw "HTTP $($r.StatusCode) from $u" }
    return $r.Content | ConvertFrom-Json
}

Write-Host "Checking API at $Base ..." -ForegroundColor Cyan

$ready = Get-Json "/ready"
if (-not $ready.available) {
    Write-Host "FAIL: /ready reports model not available. Start MLflow, run scripts/seed_mlflow_production.py, then api container." -ForegroundColor Red
    exit 1
}

$info = Get-Json "/api/model-info"
if (-not $info.available) {
    Write-Host "FAIL: /api/model-info model not loaded." -ForegroundColor Red
    exit 1
}

if ($info.load_source -ne "registry_production") {
    Write-Host "FAIL: Expected load_source=registry_production (MLflow Production). Got: $($info.load_source)" -ForegroundColor Red
    Write-Host "      For Docker production: set PRODUCTION_STRICT=true, unset SERVE_USE_LOCAL_MODEL_ONLY, seed registry." -ForegroundColor Yellow
    exit 1
}

if ($info.registry_production_satisfied -ne $true) {
    Write-Host "FAIL: registry_production_satisfied should be true." -ForegroundColor Red
    exit 1
}

$sum = Get-Json "/api/dashboard-summary"
$mlm = $sum.model_load
if ($mlm.registry_satisfied -ne $true) {
    Write-Host "FAIL: dashboard-summary model_load.registry_satisfied should be true." -ForegroundColor Red
    exit 1
}

Write-Host "OK: Production registry load (load_source=$($info.load_source), version=$($info.model_version))." -ForegroundColor Green
Write-Host "OK: Dashboard summary registry_satisfied=true." -ForegroundColor Green
exit 0
