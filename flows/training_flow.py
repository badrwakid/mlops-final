# flows/training_flow.py
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from prefect import flow, get_run_logger, task
from src.config import load_config
from src.data.load import load_raw
from src.data.split import build_splits, inject_drift
from src.features.preprocessor import fit_preprocessor
from src.training.registry import evaluate_candidate_promotion, promote_version_to_production
from src.training.train import main as train_main


@task(retries=1, retry_delay_seconds=5)
def validate_data() -> str:
    cfg = load_config()
    df = load_raw(cfg.paths.raw_csv)
    log = get_run_logger()
    log.info("validate_data: %d rows OK", len(df))
    return cfg.paths.raw_csv


@task
def preprocess(validated_path: str) -> str:
    cfg = load_config()
    df = pd.read_csv(validated_path).drop(columns=cfg.data.drop_columns)
    df.to_parquet(cfg.paths.processed, index=False)
    train, test, reference, year1 = build_splits(
        df,
        split_col=cfg.data.split_column,
        test_size=cfg.data.test_size,
        ref_holdout=cfg.data.reference_holdout,
        random_state=cfg.data.random_state,
    )
    production = inject_drift(
        year1,
        factor_temp=cfg.drift.perturb_temp_factor,
        factor_hum=cfg.drift.perturb_hum_factor,
        std_windspeed=cfg.drift.perturb_windspeed_noise_std,
        seed=cfg.data.random_state,
    )
    train.to_parquet(cfg.paths.train, index=False)
    test.to_parquet(cfg.paths.test, index=False)
    reference.to_parquet(cfg.paths.reference, index=False)
    production.to_parquet(cfg.paths.production, index=False)

    pipe = fit_preprocessor(
        train,
        target=cfg.data.target,
        numeric=cfg.data.numeric_features,
        categorical=cfg.data.categorical_features,
        k=cfg.preprocessing.feature_selection_k,
        numeric_strategy=cfg.preprocessing.numeric_imputer_strategy,
        categorical_strategy=cfg.preprocessing.categorical_imputer_strategy,
    )
    Path(cfg.paths.preprocessor).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, cfg.paths.preprocessor)
    return cfg.paths.train


@task
def train(train_path: str) -> str:
    get_run_logger().info("train: using splits from %s", train_path)
    train_main()
    cfg = load_config()
    return cfg.paths.model


@task
def evaluate(model_path: str) -> dict:
    cfg = load_config()
    get_run_logger().info("evaluate: model artifact %s", model_path)
    with Path(cfg.paths.metrics).open(encoding="utf-8") as f:
        metrics = json.load(f)
    log = get_run_logger()
    log.info("evaluate: r2=%.4f", metrics["test"]["r2"])
    if metrics["test"]["r2"] < cfg.validation.min_test_r2:
        msg = (
            f"r2={metrics['test']['r2']:.3f} below threshold {cfg.validation.min_test_r2}"
        )
        raise ValueError(msg)
    return metrics


@task
def register_model(_metrics: dict) -> int:
    cfg = load_config()
    decision = evaluate_candidate_promotion()
    if not decision.should_promote:
        raise RuntimeError(f"registry promotion rejected: {decision.compare_status}")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI") or cfg.mlflow.tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    version_str = promote_version_to_production(
        client,
        cfg.mlflow.registered_model_name,
        decision.candidate.version,
        decision=decision,
        approval_mode="prefect_flow",
        approved=True,
    )
    log = get_run_logger()
    log.info("register_model: promoted version %s to Production", version_str)
    return int(version_str)


@flow(name="bike-share-training")
def training_flow():
    validated = validate_data()
    preprocessed = preprocess(validated)
    model_path = train(preprocessed)
    metrics = evaluate(model_path)
    version = register_model(metrics)
    return version


if __name__ == "__main__":
    training_flow()
