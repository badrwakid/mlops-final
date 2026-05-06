import numpy as np
from sklearn.preprocessing import StandardScaler
from src.training import hpo as hpo_module
from src.training.hpo import run_hpo


def test_run_hpo_uses_seeded_search_space_and_logs_trials(monkeypatch):
    logged_params = []
    logged_metrics = []
    logged_step_metrics = []
    run_kwargs = []
    trial_params = []

    class DummyRun:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            run_kwargs.append(self.kwargs)
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    def fake_cross_val_score(estimator, X, y, cv, scoring, n_jobs):
        params = estimator.get_params()
        selected = {
            "n_estimators": params["n_estimators"],
            "max_depth": params["max_depth"],
            "min_samples_leaf": params["min_samples_leaf"],
            "random_state": params["random_state"],
            "n_jobs": params["n_jobs"],
        }
        trial_params.append(selected)
        rmse = 2.0
        if selected["n_estimators"] == 20:
            rmse -= 0.5
        if selected["max_depth"] == 4:
            rmse -= 0.25
        return np.array([-(rmse**2), -(rmse**2)])

    monkeypatch.setattr(hpo_module.mlflow, "start_run", lambda **kwargs: DummyRun(**kwargs))
    monkeypatch.setattr(hpo_module.mlflow, "log_params", logged_params.append)
    monkeypatch.setattr(hpo_module.mlflow, "log_metrics", logged_metrics.append)
    monkeypatch.setattr(
        hpo_module.mlflow,
        "log_metric",
        lambda key, value, step=None: logged_step_metrics.append(
            {"key": key, "value": value, "step": step}
        ),
    )
    monkeypatch.setattr(hpo_module, "cross_val_score", fake_cross_val_score)

    search_space = {
        "n_estimators": [10, 20],
        "max_depth": [2, 4],
        "min_samples_leaf": [1, 2],
    }
    X = np.arange(24).reshape(12, 2)
    y = np.arange(12)

    first = run_hpo(X, y, search_space, n_trials=4, cv_folds=2, random_state=123)
    first_sequence = list(trial_params)
    trial_params.clear()
    second = run_hpo(X, y, search_space, n_trials=4, cv_folds=2, random_state=123)

    assert first.best_params == second.best_params
    assert first_sequence == trial_params
    assert first.best_cv_rmse == second.best_cv_rmse
    assert all(run["nested"] for run in run_kwargs)
    assert all(params["random_state"] == 123 for params in logged_params)
    assert [metric["step"] for metric in logged_step_metrics] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert all("cv_rmse" in metrics for metrics in logged_metrics)


def test_run_hpo_uses_pipeline_with_preprocessor_when_factory_is_provided(monkeypatch):
    estimators = []

    class DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    def fake_cross_val_score(estimator, X, y, cv, scoring, n_jobs):
        estimators.append(estimator)
        return np.array([-1.0, -1.0])

    monkeypatch.setattr(hpo_module.mlflow, "start_run", lambda **kwargs: DummyRun())
    monkeypatch.setattr(hpo_module.mlflow, "log_params", lambda params: None)
    monkeypatch.setattr(hpo_module.mlflow, "log_metrics", lambda metrics: None)
    monkeypatch.setattr(hpo_module.mlflow, "log_metric", lambda key, value, step=None: None)
    monkeypatch.setattr(hpo_module, "cross_val_score", fake_cross_val_score)

    run_hpo(
        np.arange(24).reshape(12, 2),
        np.arange(12),
        {"n_estimators": [10], "max_depth": [2], "min_samples_leaf": [1]},
        n_trials=1,
        cv_folds=2,
        random_state=123,
        preprocessor_factory=StandardScaler,
    )

    estimator = estimators[0]
    assert [name for name, _ in estimator.steps] == ["preprocessor", "model"]
    assert isinstance(estimator.named_steps["preprocessor"], StandardScaler)
    assert estimator.named_steps["model"].random_state == 123


def test_run_hpo_supports_hist_gradient_boosting(monkeypatch):
    captured = {}

    class DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    def fake_cross_val_score(estimator, X, y, cv, scoring, n_jobs):
        captured["estimator"] = estimator
        return np.array([-4.0, -4.0])

    monkeypatch.setattr(hpo_module.mlflow, "start_run", lambda **kwargs: DummyRun())
    monkeypatch.setattr(hpo_module.mlflow, "log_params", lambda params: None)
    monkeypatch.setattr(hpo_module.mlflow, "log_metrics", lambda metrics: None)
    monkeypatch.setattr(hpo_module.mlflow, "log_metric", lambda key, value, step=None: None)
    monkeypatch.setattr(hpo_module, "cross_val_score", fake_cross_val_score)

    run_hpo(
        np.arange(24).reshape(12, 2),
        np.arange(12),
        {
            "max_iter": [200],
            "learning_rate": [0.05],
            "max_depth": [8],
            "max_leaf_nodes": [31],
            "min_samples_leaf": [20],
            "l2_regularization": [0.0],
        },
        n_trials=1,
        cv_folds=2,
        model_type="hist_gradient_boosting",
        random_state=123,
    )

    assert captured["estimator"].__class__.__name__ == "HistGradientBoostingRegressor"
