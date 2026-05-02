from __future__ import annotations

import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
from src.config import load_config

OUTPUT_PATH = Path("docs") / "experiment_log.csv"


def resolve_tracking_uri(configured_uri: str) -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or configured_uri


def write_experiment_log(runs: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> int:
    if runs.empty:
        raise ValueError("No runs found to export")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(output_path, index=False)
    return len(runs)


def export_runs(output_path: Path = OUTPUT_PATH) -> int:
    cfg = load_config()
    tracking_uri = resolve_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(cfg.mlflow.experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment not found: {cfg.mlflow.experiment_name}")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        output_format="pandas",
    )
    row_count = write_experiment_log(runs, output_path)
    print(f"exported {row_count} runs to {output_path}")
    return row_count


def main() -> int:
    try:
        export_runs()
    except ValueError as exc:
        print(f"export_runs: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
