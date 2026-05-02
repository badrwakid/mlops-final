"""Gate CI on committed test metrics and (optionally) the MLflow Production model."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
from src.config import load_config


def _load_rmse(metrics_path: Path) -> float:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return float(payload["test"]["rmse"])


def main() -> int:
    cfg = load_config()
    metrics_path = Path(cfg.paths.metrics)
    if not metrics_path.is_file():
        print(f"ERROR: metrics file not found: {metrics_path}", file=sys.stderr)
        return 1

    rmse = _load_rmse(metrics_path)
    ceiling = cfg.validation.rmse_threshold
    if rmse >= ceiling:
        print(
            f"ERROR: test RMSE {rmse:.6f} >= validation.rmse_threshold {ceiling:.6f}",
            file=sys.stderr,
        )
        return 1
    print(f"OK: test RMSE {rmse:.6f} < rmse_threshold {ceiling:.6f}")

    if os.environ.get("SKIP_MLFLOW_REGISTRY", "").lower() in {"1", "true", "yes"}:
        print("SKIP_MLFLOW_REGISTRY set: skipping Production model load")
        return 0

    tracking = os.environ.get("MLFLOW_TRACKING_URI", cfg.mlflow.tracking_uri)
    mlflow.set_tracking_uri(tracking)
    name = cfg.mlflow.registered_model_name
    uri = f"models:/{name}/Production"
    try:
        mlflow.sklearn.load_model(uri)
    except Exception as exc:
        print(f"ERROR: could not load registry model {uri}: {exc}", file=sys.stderr)
        return 1
    print(f"OK: loaded registry model {uri}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
