"""
MLflow cleanup — rename runs, normalize tags, add descriptions.

Usage:
    python scripts/mlflow_cleanup.py --dry-run
    python scripts/mlflow_cleanup.py --apply
    python scripts/mlflow_cleanup.py --apply --experiment-id 886455878294338602
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_EXPERIMENT_ID = "886455878294338602"

AUTO_NAME_PATTERNS = [
    r"^[a-z]+-[a-z]+-\d+$",
    r"^run\d+$",
    r"^run_\d{4}-\d{2}-\d{2}.*$",
    r"^\s*$",
    r"^untitled.*$",
]


def is_auto_generated(name: str | None) -> bool:
    if not name:
        return True
    normalized = name.strip().lower()
    return any(re.match(pattern, normalized) for pattern in AUTO_NAME_PATTERNS)


def infer_stage_and_algo(run) -> tuple[str, str]:  # noqa: ANN001
    """Infer stage and algorithm from tags/params/metrics/run-name heuristics."""
    tags = run.data.tags or {}
    params = run.data.params or {}
    metrics = run.data.metrics or {}
    run_name = (run.info.run_name or "").lower()

    stage = tags.get("stage")
    algo = tags.get("algo") or tags.get("model_type") or tags.get("estimator")

    if not algo:
        if "n_estimators" in params and "max_depth" in params:
            algo = "rf" if "criterion" in params else "xgb"
        elif "num_leaves" in params:
            algo = "lgbm"
        elif "learning_rate_init" in params:
            algo = "mlp"
        elif "alpha" in params and "fit_intercept" in params:
            algo = "lr"
        elif "persistence" in run_name:
            algo = "persistence"
        elif "hourly" in run_name and "mean" in run_name:
            algo = "hourly_mean"
        elif "drift" in run_name:
            algo = "check"
        else:
            algo = "unknown"

    if not stage:
        if any(metric.startswith("drift") or "drift" in metric for metric in metrics):
            stage = "drift"
        elif "persistence" in run_name or "hourly_mean" in run_name:
            stage = "baseline"
        elif "cv_" in "".join(metrics.keys()) or "best_cv_rmse" in metrics:
            stage = "tune"
        elif "test_rmse" in metrics or "val_rmse" in metrics:
            stage = "train"
        elif "debug" in run_name:
            stage = "debug"
        else:
            stage = "train"

    return stage, algo


def short_id(run) -> str:  # noqa: ANN001
    ts = datetime.fromtimestamp(run.info.start_time / 1000)
    return ts.strftime("%y%m%d%H")


def proposed_name(run) -> str:  # noqa: ANN001
    stage, algo = infer_stage_and_algo(run)
    return f"{stage}__{algo}__{short_id(run)}"


def _short_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def proposed_tags(run) -> dict[str, str]:  # noqa: ANN001
    stage, algo = infer_stage_and_algo(run)
    existing_tags = run.data.tags or {}
    existing_params = run.data.params or {}
    dataset_hash = (
        existing_tags.get("dataset_hash")
        or existing_params.get("dataset_hash")
        or existing_tags.get("data_hash")
        or "unknown"
    )
    if len(dataset_hash) > 12:
        dataset_hash = dataset_hash[:12]
    author = existing_tags.get("author") or os.environ.get("USERNAME") or os.environ.get("USER") or "unknown"
    git_commit = (
        existing_tags.get("git_commit")
        or existing_params.get("git_commit")
        or _short_git_commit()
    )
    return {
        "stage": stage,
        "algo": algo,
        "dataset_hash": str(dataset_hash),
        "author": str(author),
        "git_commit": str(git_commit),
        "cleaned_by": "mlflow_cleanup.py",
    }


def proposed_description(run) -> str | None:  # noqa: ANN001
    existing = (run.data.tags or {}).get("mlflow.note.content")
    if existing and len(existing.strip()) > 5:
        return None
    params = run.data.params or {}
    metrics = run.data.metrics or {}
    _stage, algo = infer_stage_and_algo(run)
    parts = [f"{algo} run."]
    if "n_estimators" in params:
        parts.append(f"trees={params['n_estimators']}")
    if "max_depth" in params:
        parts.append(f"depth={params['max_depth']}")
    if "learning_rate" in params:
        parts.append(f"lr={params['learning_rate']}")
    if "test_rmse" in metrics:
        parts.append(f"test RMSE {metrics['test_rmse']:.2f}")
    elif "rmse" in metrics:
        parts.append(f"RMSE {metrics['rmse']:.2f}")
    return " ".join(parts) if len(parts) > 1 else None


def cleanup_experiment(client: MlflowClient, experiment_id: str, apply: bool) -> int:
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        run_view_type=ViewType.ALL,
        max_results=10000,
        order_by=["start_time DESC"],
    )

    print(f"Found {len(runs)} runs in experiment {experiment_id}\n")
    print(f"{'STATUS':<8} {'OLD NAME':<35} -> {'NEW NAME':<35}  TAGS")
    print("-" * 120)

    changes = 0
    for run in runs:
        old_name = run.info.run_name or "(empty)"
        new_name = proposed_name(run)
        new_tags = proposed_tags(run)
        new_desc = proposed_description(run)

        will_rename = is_auto_generated(old_name) and old_name != new_name
        existing_tags = run.data.tags or {}
        will_tag = any(existing_tags.get(k) != v for k, v in new_tags.items())

        if not will_rename and not will_tag and not new_desc:
            print(f"{'SKIP':<8} {old_name[:34]:<35}   (already clean)")
            continue

        changes += 1
        action = "APPLY" if apply else "DRY"
        target_name = new_name if will_rename else old_name
        print(f"{action:<8} {old_name[:34]:<35} -> {target_name[:34]:<35}  {new_tags}")

        if not apply:
            continue

        if will_rename:
            client.set_tag(run.info.run_id, "mlflow.runName", new_name)
        for key, value in new_tags.items():
            client.set_tag(run.info.run_id, key, value)
        if new_desc:
            client.set_tag(run.info.run_id, "mlflow.note.content", new_desc)

    print("-" * 120)
    print(f"\n{'Applied' if apply else 'Would apply'} {changes} change(s).")
    if not apply:
        print("\nRe-run with --apply to commit these changes.")
    return changes


def maybe_rename_experiment(client: MlflowClient, experiment_id: str, apply: bool) -> None:
    exp = client.get_experiment(experiment_id)
    current = exp.name
    proposed = "bike_share_training"
    if current == proposed:
        print(f"Experiment name already clean: {current!r}")
        return
    print(f"\nExperiment name: {current!r} -> {proposed!r}")
    if apply:
        client.rename_experiment(experiment_id, proposed)
        print("Renamed.")
    else:
        print("(dry run — pass --apply to rename)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default=TRACKING_URI)
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--apply", action="store_true", help="Actually apply changes")
    parser.add_argument("--dry-run", action="store_true", help="Show changes only (default)")
    parser.add_argument(
        "--rename-experiment",
        action="store_true",
        help="Also rename the experiment itself",
    )
    args = parser.parse_args()

    apply = args.apply and not args.dry_run
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient(tracking_uri=args.tracking_uri)

    try:
        client.get_experiment(args.experiment_id)
    except Exception as exc:
        print(f"Cannot reach experiment {args.experiment_id} at {args.tracking_uri}")
        print(f"Is the MLflow server running? Error: {exc}")
        sys.exit(1)

    cleanup_experiment(client, args.experiment_id, apply=apply)

    if args.rename_experiment:
        maybe_rename_experiment(client, args.experiment_id, apply=apply)


if __name__ == "__main__":
    main()
