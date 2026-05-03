from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.config import Config, load_config
from src.evaluation.metrics import compute_metrics
from src.features.preprocessor import build_preprocessor


def _resolve_tracking_uri(configured_uri: str) -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or configured_uri


def _build_baseline_model(random_state: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        n_jobs=-1,
        random_state=random_state,
    )


def _feature_columns(cfg: Config) -> list[str]:
    return cfg.data.numeric_features + cfg.data.categorical_features


def _build_model(params: dict, random_state: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        **params,
        n_jobs=-1,
        random_state=random_state,
    )


def _validation_gate_info(test_metrics: dict[str, float], min_test_r2: float) -> dict:
    return {
        "min_test_r2": float(min_test_r2),
        "passed": bool(test_metrics["r2"] >= min_test_r2),
    }


def _apply_validation_gate(test_metrics: dict[str, float], min_test_r2: float) -> dict:
    gate = _validation_gate_info(test_metrics, min_test_r2)
    if not gate["passed"]:
        raise RuntimeError(
            "Validation gate failed: "
            f"test_r2={test_metrics['r2']:.6f} < min_test_r2={min_test_r2:.6f}"
        )
    return gate


def _persist_artifacts_if_gate_passes(
    test_metrics: dict[str, float],
    min_test_r2: float,
    persist_artifacts: Callable[[], None],
) -> dict:
    gate = _apply_validation_gate(test_metrics, min_test_r2)
    persist_artifacts()
    return gate


def main() -> None:
    import mlflow
    import mlflow.sklearn

    from src.training.hpo import run_hpo

    cfg = load_config()
    tracking_uri = _resolve_tracking_uri(cfg.mlflow.tracking_uri)

    train = pd.read_parquet(cfg.paths.train)
    test = pd.read_parquet(cfg.paths.test)
    preprocessor = joblib.load(cfg.paths.preprocessor)

    feature_columns = _feature_columns(cfg)
    train_features = train[feature_columns]
    test_features = test[feature_columns]
    x_train = preprocessor.transform(train_features)
    y_train = train[cfg.data.target]
    x_test = preprocessor.transform(test_features)
    y_test = test[cfg.data.target]

    def preprocessor_factory():
        return build_preprocessor(
            numeric=cfg.data.numeric_features,
            categorical=cfg.data.categorical_features,
            k=cfg.preprocessing.feature_selection_k,
            numeric_strategy=cfg.preprocessing.numeric_imputer_strategy,
            categorical_strategy=cfg.preprocessing.categorical_imputer_strategy,
            cyclical_hr_mnth=cfg.preprocessing.cyclical_hr_mnth,
        )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    with mlflow.start_run(run_name="parent_hpo"):
        mlflow.log_params({
            "model_type": cfg.training.model_type,
            "cv_folds": cfg.training.cv_folds,
            "n_trials": cfg.training.n_trials,
            "feature_selection_k": cfg.preprocessing.feature_selection_k,
            "cyclical_hr_mnth": cfg.preprocessing.cyclical_hr_mnth,
            "validation_min_test_r2": cfg.validation.min_test_r2,
        })
        hpo_result = run_hpo(
            train_features,
            y_train,
            search_space=cfg.training.hpo_search_space,
            n_trials=cfg.training.n_trials,
            cv_folds=cfg.training.cv_folds,
            random_state=cfg.data.random_state,
            preprocessor_factory=preprocessor_factory,
        )
        mlflow.log_params({f"best_{key}": value for key, value in hpo_result.best_params.items()})
        mlflow.log_metric("best_cv_rmse", hpo_result.best_cv_rmse)

        model = _build_model(hpo_result.best_params, random_state=cfg.data.random_state)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        test_metrics = compute_metrics(y_test, y_pred)
        gate = _validation_gate_info(test_metrics, cfg.validation.min_test_r2)

        metrics_payload = {
            "test": test_metrics,
            "best_params": hpo_result.best_params,
            "best_cv_rmse": hpo_result.best_cv_rmse,
            "gate": gate,
        }

        metrics_path = Path(cfg.paths.metrics)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(metrics_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})
        mlflow.log_metric("validation_min_test_r2", gate["min_test_r2"])
        mlflow.log_param("validation_passed", gate["passed"])

        def persist_artifacts() -> None:
            model_path = Path(cfg.paths.model)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=cfg.mlflow.registered_model_name,
            )
            mlflow.log_artifact(cfg.paths.preprocessor, artifact_path="preprocessor")

        _persist_artifacts_if_gate_passes(
            test_metrics,
            cfg.validation.min_test_r2,
            persist_artifacts,
        )

    print(
        "train: "
        f"tracking_uri={tracking_uri} "
        f"best_cv_rmse={hpo_result.best_cv_rmse:.6f} "
        f"test_r2={test_metrics['r2']:.6f} "
        f"test_rmse={test_metrics['rmse']:.6f} "
        f"gate={'passed' if gate['passed'] else 'failed'}"
    )


if __name__ == "__main__":
    main()
