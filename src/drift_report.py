from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.report import Report

FEATURE_COLS = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]
NUMERICAL_FEATURES = ["temp", "atemp", "hum", "windspeed"]
CATEGORICAL_FEATURES = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
]
REQUIRED_COLUMNS = FEATURE_COLS


def build_column_mapping(
    *,
    include_target: bool,
    include_prediction: bool,
    numerical_features: list[str],
    categorical_features: list[str],
) -> ColumnMapping:
    cm = ColumnMapping()
    if include_prediction:
        cm.prediction = "prediction"
    else:
        cm.prediction = None
    if include_target:
        cm.target = "target"
    cm.numerical_features = numerical_features
    cm.categorical_features = categorical_features
    return cm


def _prepare_current_df(current_df: pd.DataFrame, *, include_prediction: bool) -> pd.DataFrame:
    prepared = current_df.copy()
    required_columns = REQUIRED_COLUMNS + (["prediction"] if include_prediction else [])
    for col in required_columns:
        if col not in prepared.columns:
            prepared[col] = pd.NA
    prepared = prepared[required_columns]
    numeric_cols = NUMERICAL_FEATURES + (["prediction"] if include_prediction else [])
    for col in numeric_cols:
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce").astype("Int64")
    if include_prediction:
        prepared = prepared.dropna(subset=["prediction"])
    return prepared.reset_index(drop=True)


def _prepare_reference_df(reference: pd.DataFrame, *, include_prediction: bool) -> pd.DataFrame:
    prepared = reference.copy()
    required_columns = REQUIRED_COLUMNS + (["prediction"] if include_prediction else [])
    for col in required_columns:
        if col not in prepared.columns:
            prepared[col] = pd.NA
    prepared = prepared[required_columns]
    numeric_cols = NUMERICAL_FEATURES + (["prediction"] if include_prediction else [])
    for col in numeric_cols:
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce").astype("Int64")
    if include_prediction:
        prepared = prepared.dropna(subset=["prediction"])
    return prepared.reset_index(drop=True)


def generate_drift_report(reference_path: str, current_df: pd.DataFrame, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    reference = pd.read_parquet(reference_path)
    include_prediction = (
        "prediction" in reference.columns
        and reference["prediction"].notna().any()
        and "prediction" in current_df.columns
        and current_df["prediction"].notna().any()
    )
    include_target = (
        "target" in reference.columns
        and "target" in current_df.columns
        and current_df["target"].notna().any()
    )
    reference_prepared = _prepare_reference_df(reference, include_prediction=include_prediction)
    current_prepared = _prepare_current_df(current_df, include_prediction=include_prediction)
    usable_numerical = [
        c
        for c in NUMERICAL_FEATURES
        if c in reference_prepared.columns
        and c in current_prepared.columns
        and reference_prepared[c].notna().any()
        and current_prepared[c].notna().any()
    ]
    usable_categorical = [
        c
        for c in CATEGORICAL_FEATURES
        if c in reference_prepared.columns
        and c in current_prepared.columns
        and reference_prepared[c].notna().any()
        and current_prepared[c].notna().any()
    ]
    keep_cols = usable_numerical + usable_categorical
    if include_prediction:
        keep_cols.append("prediction")
    if include_target:
        keep_cols.append("target")
    if not usable_numerical and not usable_categorical:
        raise ValueError("No overlapping non-empty drift features between reference and current datasets.")
    reference_prepared = reference_prepared[keep_cols]
    current_prepared = current_prepared[keep_cols]
    column_mapping = build_column_mapping(
        include_target=include_target,
        include_prediction=include_prediction,
        numerical_features=usable_numerical,
        categorical_features=usable_categorical,
    )
    metrics = [DataDriftPreset(), DataQualityPreset()]
    if include_target:
        metrics.insert(1, TargetDriftPreset())
    report = Report(metrics=metrics)
    report.run(
        reference_data=reference_prepared,
        current_data=current_prepared,
        column_mapping=column_mapping,
    )

    html_path = out / "drift_report.html"
    json_path = out / "drift_report.json"
    report.save_html(str(html_path))
    json_path.write_text(json.dumps(report.as_dict(), default=str), encoding="utf-8")
    return html_path, json_path


def summarize(json_path: Path) -> dict[str, Any]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    dataset_drift = False
    drifted_features = 0
    total_features = 0
    share_drifted = 0.0
    for metric in data.get("metrics", []):
        if metric.get("metric") == "DatasetDriftMetric":
            result = metric.get("result", {})
            dataset_drift = bool(result.get("dataset_drift", False))
            drifted_features = int(result.get("number_of_drifted_columns", 0))
            total_features = int(result.get("number_of_columns", 0))
            share_drifted = float(result.get("share_of_drifted_columns", 0.0))
            break
    return {
        "dataset_drift": dataset_drift,
        "drifted_features": drifted_features,
        "total_features": total_features,
        "share_drifted": round(share_drifted, 3),
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
    }


def run_and_log(reference_path: str, current_df: pd.DataFrame, experiment_name: str = "bike_sharing_drift"):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"drift_{dt.datetime.utcnow():%Y%m%d_%H%M%S}") as run:
        out_dir = Path("artifacts/drift_runs") / run.info.run_id
        html_path, json_path = generate_drift_report(reference_path, current_df, str(out_dir))
        summary = summarize(json_path)

        mlflow.log_artifact(str(html_path), artifact_path="drift")
        mlflow.log_artifact(str(json_path), artifact_path="drift")
        mlflow.log_metric("share_drifted", summary["share_drifted"])
        mlflow.log_metric("drifted_features", summary["drifted_features"])
        mlflow.log_metric("total_features", summary["total_features"])
        mlflow.set_tag("dataset_drift", str(summary["dataset_drift"]))
        mlflow.set_tag("generated_at", summary["generated_at"])

        return {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "html_artifact_uri": f"runs:/{run.info.run_id}/drift/drift_report.html",
            "summary": summary,
        }
