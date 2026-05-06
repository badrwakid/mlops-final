# Drift report

This report summarizes runtime drift checks produced by `monitoring/run_monitoring.py`.

## Latest summary

- Alert: `true`
- Drift share: `0.4459`
- Drift share (inputs only): `0.4545`
- Severity: `P2`
- Threshold: `0.2`
- Drifted features: `hum`, `season`, `temp`, `weathersit`, `windspeed`

## Operational policy

Trigger retraining when input drift share exceeds threshold in two consecutive windows; investigate feature pipelines immediately.

## Sources

- `/api/drift-summary`
- `monitoring/evidently_reports/drift_summary.json`
- `monitoring/evidently_reports/drift.html`
