from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import mlflow
import numpy as np
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class HPOResult:
    best_params: dict[str, Any]
    best_cv_rmse: float
    study: optuna.Study


def run_hpo(
    X,
    y,
    search_space: dict[str, list[Any]],
    n_trials: int,
    cv_folds: int,
    random_state: int = 42,
    preprocessor_factory: Callable[[], Any] | None = None,
) -> HPOResult:
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        trial_params = {
            name: trial.suggest_categorical(name, choices)
            for name, choices in search_space.items()
        }
        model = RandomForestRegressor(
            **trial_params,
            n_jobs=-1,
            random_state=random_state,
        )
        estimator = model
        if preprocessor_factory is not None:
            estimator = Pipeline([
                ("preprocessor", preprocessor_factory()),
                ("model", model),
            ])
        scores = cross_val_score(
            estimator,
            X,
            y,
            cv=cv_folds,
            scoring="neg_mean_squared_error",
            n_jobs=None,
        )
        cv_rmse = float(np.sqrt(-scores.mean()))

        logged_params = {
            **trial_params,
            "n_jobs": -1,
            "random_state": random_state,
        }
        with mlflow.start_run(run_name=f"hpo_trial_{trial.number}", nested=True):
            mlflow.log_params(logged_params)
            mlflow.log_metrics({"cv_rmse": cv_rmse})
            mlflow.log_metric("cv_rmse_step", cv_rmse, step=trial.number)

        return cv_rmse

    study.optimize(objective, n_trials=n_trials)
    return HPOResult(
        best_params=dict(study.best_params),
        best_cv_rmse=float(study.best_value),
        study=study,
    )
