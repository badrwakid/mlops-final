#!/usr/bin/env python3
"""Upload data/splits/model.pkl to MLflow registry and promote it to Production.

Use this after `docker compose up -d mlflow` so the API can load
`models:/<registered_model_name>/Production` instead of SERVE_USE_LOCAL_MODEL_ONLY.

Examples:
  MLFLOW_TRACKING_URI=http://127.0.0.1:5001 python scripts/seed_mlflow_production.py
  # Uses experiment_name from configs/params.yaml (e.g. bike_sharing) unless you pass --experiment.
"""

from __future__ import annotations

import argparse
import os
import sys

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        default=None,
        help="MLflow experiment for the seed run (default: mlflow.experiment_name in params.yaml).",
    )
    args = parser.parse_args()

    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        print(
            "error: set MLFLOW_TRACKING_URI (e.g. http://127.0.0.1:5001 for Docker Compose)",
            file=sys.stderr,
        )
        return 1

    cfg = load_config()
    experiment = args.experiment or cfg.mlflow.experiment_name
    model_path = cfg.paths.model
    if not os.path.isfile(model_path):
        print(f"error: model file not found: {model_path}", file=sys.stderr)
        return 1

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    model = joblib.load(model_path)
    client = MlflowClient(tracking_uri=uri)
    name = cfg.mlflow.registered_model_name

    with mlflow.start_run(run_name="seed-production-model") as run:
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=name,
        )
        run_id = run.info.run_id

    version = None
    for mv in client.search_model_versions(f"name='{name}'"):
        if mv.run_id == run_id:
            version = mv.version
            break
    if version is None:
        print("error: could not resolve registered model version for this run", file=sys.stderr)
        return 1

    client.transition_model_version_stage(
        name=name,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(
        f"OK: experiment={experiment} registered {name} version={version} -> Production "
        f"(tracking_uri={uri})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
