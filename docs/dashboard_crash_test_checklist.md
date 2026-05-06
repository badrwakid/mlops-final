# Dashboard Crash Test Checklist

Use this checklist during live professor demos to prove the serving app and dashboard are resilient.

## Startup

- [ ] Start API locally and open `http://localhost:8000/`
- [ ] Open `http://localhost:8000/dashboard`
- [ ] Open `http://localhost:8000/health`
- [ ] Open `http://localhost:8000/ready`
- [ ] Open `http://localhost:8000/metrics`

## Single Prediction

- [ ] Click **Load Safe Demo Example**
- [ ] Submit valid prediction and verify response shows `prediction`, `confidence`, `model_version`
- [ ] Submit with an empty required field and verify clean validation error
- [ ] Submit with wrong type (string for numeric field) and verify clean validation error
- [ ] Submit out-of-range value (e.g., `temp=1.5`) and verify clean validation error

## Batch Prediction

- [ ] Submit a valid batch (2-3 lines) and verify results table/json appears
- [ ] Submit empty records list and verify clean validation error
- [ ] Submit >100 records and verify message: `Batch size cannot exceed 100 records`

## Failure Scenarios

- [ ] Stop MLflow and verify API still responds on `/health` and dashboard loads
- [ ] Confirm `/ready` reports not ready if model cannot load
- [ ] Confirm `/predict` and `/predict/batch` return clean `503` when model is unavailable
- [ ] Temporarily move/remove `monitoring/evidently_reports/drift_summary.json` and verify `/api/drift-summary` returns safe JSON (no crash)
- [ ] Refresh dashboard several times and verify UI remains stable

## Observability and Evidence

- [ ] Confirm links to MLflow (`http://localhost:5000`) and Prometheus (`http://localhost:9090`) are visible on dashboard
- [ ] Confirm evidence links (README, technical report, model/data cards, drift reports) are visible
- [ ] Capture screenshots for dashboard, successful prediction, and validation errors
