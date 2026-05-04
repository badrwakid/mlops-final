"""Coverage for src.training.train_baseline.main (MLflow mocked)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from src.training.train_baseline import main as train_baseline_main


def _passthrough_to_14(df):  # noqa: ANN001
    """Module-level for joblib pickling."""
    return np.zeros((len(df), 14))


def _bike_frame(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "temp": rng.random(n),
            "atemp": rng.random(n),
            "hum": rng.random(n),
            "windspeed": rng.random(n),
            "hr": rng.integers(0, 24, size=n),
            "mnth": rng.integers(1, 13, size=n),
            "season": rng.integers(1, 5, size=n),
            "holiday": rng.integers(0, 2, size=n),
            "workingday": rng.integers(0, 2, size=n),
            "weathersit": rng.integers(1, 5, size=n),
            "weekday": rng.integers(0, 7, size=n),
            "cnt": rng.integers(1, 200, size=n),
        }
    )


@patch("src.training.train_baseline.mlflow.sklearn.log_model")
@patch("src.training.train_baseline.mlflow.log_metrics")
@patch("src.training.train_baseline.mlflow.log_param")
@patch("src.training.train_baseline.mlflow.log_params")
@patch("src.training.train_baseline.mlflow.set_tags")
@patch("src.training.train_baseline.mlflow.set_experiment")
@patch("src.training.train_baseline.mlflow.set_tracking_uri")
@patch("src.training.train_baseline.mlflow.start_run")
def test_train_baseline_main_runs_ridge_with_mocked_mlflow(
    mock_start_run,
    _mock_set_uri,
    _mock_set_experiment,
    _mock_set_tags,
    _mock_log_params,
    _mock_log_param,
    _mock_log_metrics,
    _mock_log_model,
    tmp_path: Path,
    monkeypatch,
) -> None:
    train = _bike_frame(40)
    test = _bike_frame(20)
    train.to_parquet(tmp_path / "train.parquet", index=False)
    test.to_parquet(tmp_path / "test.parquet", index=False)

    joblib.dump(
        FunctionTransformer(_passthrough_to_14, validate=False),
        tmp_path / "preprocessor.pkl",
    )

    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=ctx)
    ctx.__exit__ = MagicMock(return_value=None)
    mock_start_run.return_value = ctx

    cfg = SimpleNamespace(
        paths=SimpleNamespace(
            train=str(tmp_path / "train.parquet"),
            test=str(tmp_path / "test.parquet"),
            preprocessor=str(tmp_path / "preprocessor.pkl"),
        ),
        data=SimpleNamespace(
            target="cnt",
            random_state=0,
            numeric_features=["temp", "atemp", "hum", "windspeed", "hr", "mnth"],
            categorical_features=[
                "season",
                "holiday",
                "workingday",
                "weathersit",
                "weekday",
            ],
        ),
        preprocessing=SimpleNamespace(feature_selection_k=14, cyclical_hr_mnth=True),
        validation=SimpleNamespace(min_test_r2=-1.0),
        mlflow=SimpleNamespace(
            tracking_uri="file:///tmp/mlruns",
            experiment_name="test_ridge",
        ),
    )
    monkeypatch.setattr("src.training.train_baseline.load_config", lambda: cfg)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "")

    train_baseline_main()

    mock_start_run.assert_called_once()
