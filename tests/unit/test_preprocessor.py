import numpy as np
import pandas as pd
from src.features.preprocessor import build_preprocessor, fit_preprocessor


def _toy_train(n=300):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
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
        "cnt": rng.integers(0, 200, size=n),
    })


def test_build_preprocessor_returns_pipeline_with_correct_steps():
    pipe = build_preprocessor(
        numeric=["temp", "atemp", "hum", "windspeed", "hr", "mnth"],
        categorical=["season", "holiday", "workingday", "weathersit", "weekday"],
        k=10,
    )
    step_names = [s[0] for s in pipe.steps]
    assert "preprocessor" in step_names
    assert "selector" in step_names


def test_fit_then_transform_produces_2d_array_with_no_nans():
    df = _toy_train()
    pipe = fit_preprocessor(df, target="cnt", numeric=[
        "temp", "atemp", "hum", "windspeed", "hr", "mnth",
    ], categorical=[
        "season", "holiday", "workingday", "weathersit", "weekday",
    ], k=10)
    X = df.drop(columns=["cnt"])
    X_t = pipe.transform(X)
    assert X_t.ndim == 2
    assert X_t.shape[0] == len(df)
    assert not np.isnan(X_t).any()


def test_handles_unseen_category_at_transform_time():
    df = _toy_train()
    pipe = fit_preprocessor(df, target="cnt", numeric=[
        "temp", "atemp", "hum", "windspeed", "hr", "mnth",
    ], categorical=[
        "season", "holiday", "workingday", "weathersit", "weekday",
    ], k=10)
    new = df.iloc[:5].copy()
    new["season"] = 99  # unseen category
    X_t = pipe.transform(new.drop(columns=["cnt"]))
    assert X_t.shape == (5, 10)


def test_imputes_missing_numeric_values():
    df = _toy_train()
    df.loc[0:10, "temp"] = np.nan
    pipe = fit_preprocessor(df, target="cnt", numeric=[
        "temp", "atemp", "hum", "windspeed", "hr", "mnth",
    ], categorical=[
        "season", "holiday", "workingday", "weathersit", "weekday",
    ], k=10)
    X_t = pipe.transform(df.drop(columns=["cnt"]))
    assert not np.isnan(X_t).any()
