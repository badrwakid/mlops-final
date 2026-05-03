"""
Ridge regression baseline: same bike_sharing experiment, different algorithm family
(Lecture 04 / rubric: compare across model types). Does not register a production model.
"""

from __future__ import annotations

import os

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import Ridge

from src.config import load_config
from src.evaluation.metrics import compute_metrics


def _resolve_tracking_uri(configured_uri: str) -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or configured_uri


def _feature_columns(cfg) -> list[str]:
    return cfg.data.numeric_features + cfg.data.categorical_features


def main() -> None:
    cfg = load_config()
    tracking_uri = _resolve_tracking_uri(cfg.mlflow.tracking_uri)

    train = pd.read_parquet(cfg.paths.train)
    test = pd.read_parquet(cfg.paths.test)
    preprocessor = joblib.load(cfg.paths.preprocessor)

    feature_columns = _feature_columns(cfg)
    x_train = preprocessor.transform(train[feature_columns])
    y_train = train[cfg.data.target]
    x_test = preprocessor.transform(test[feature_columns])
    y_test = test[cfg.data.target]

    alpha = 1.0
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    with mlflow.start_run(run_name="ridge_baseline"):
        mlflow.set_tags(
            {
                "run_kind": "baseline",
                "model_family": "ridge",
                "description": "Linear baseline for algorithm comparison vs RandomForest HPO",
            }
        )
        mlflow.log_params(
            {
                "model_type": "ridge",
                "alpha": alpha,
                "feature_selection_k": cfg.preprocessing.feature_selection_k,
                "cyclical_hr_mnth": cfg.preprocessing.cyclical_hr_mnth,
            }
        )
        model = Ridge(alpha=alpha, random_state=cfg.data.random_state)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        test_metrics = compute_metrics(y_test, y_pred)
        passed = test_metrics["r2"] >= cfg.validation.min_test_r2
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_param("validation_min_test_r2", cfg.validation.min_test_r2)
        mlflow.log_param("validation_passed", passed)
        mlflow.sklearn.log_model(model, artifact_path="ridge_model")
    print(
        f"train_baseline: ridge alpha={alpha} test_r2={test_metrics['r2']:.6f} "
        f"test_rmse={test_metrics['rmse']:.6f} (no registry promotion)"
    )


if __name__ == "__main__":
    main()
