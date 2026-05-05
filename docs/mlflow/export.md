# MLflow experiment log export (rubric §7)

The project exports all runs for the configured experiment to a single CSV for submission evidence.

**Command (from repo root, venv activated):**

```powershell
python scripts/export_runs.py
```

**Default output:** [`docs/experiment_log.csv`](../experiment_log.csv)

**Requirements:** Tracking URI must be reachable (`MLFLOW_TRACKING_URI` or value in `configs/params.yaml`). The experiment name must exist (`configs/params.yaml` → `mlflow.experiment_name`).

If no runs exist yet, complete at least one training run (for example `python -m dvc repro` after data is present) before exporting.
