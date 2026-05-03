import numpy as np
import pandas as pd
from src.data.split import build_splits


def _toy(n=200):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "yr": rng.integers(0, 2, size=n),
        "temp": rng.random(n),
        "hum": rng.random(n),
        "windspeed": rng.random(n),
        "cnt": rng.integers(0, 100, size=n),
        "season": rng.integers(1, 5, size=n),
        "holiday": rng.integers(0, 2, size=n),
        "workingday": rng.integers(0, 2, size=n),
        "weathersit": rng.integers(1, 5, size=n),
        "weekday": rng.integers(0, 7, size=n),
        "atemp": rng.random(n),
        "hr": rng.integers(0, 24, size=n),
        "mnth": rng.integers(1, 13, size=n),
        # Leakage columns from raw UCI schema — must never appear in split outputs
        "casual": rng.integers(0, 50, size=n),
        "registered": rng.integers(0, 80, size=n),
    })


def test_build_splits_returns_four_disjoint_frames():
    df = _toy()
    train, test, reference, production = build_splits(
        df, split_col="yr", test_size=0.2, ref_holdout=0.1, random_state=0,
    )
    total = len(train) + len(test) + len(reference)
    assert total == (df["yr"] == 0).sum()
    assert (production["yr"] == 1).all()
    assert (train["yr"] == 0).all() and (test["yr"] == 0).all() and (reference["yr"] == 0).all()


def test_no_target_leakage_columns_present():
    df = _toy()
    assert "casual" in df.columns and "registered" in df.columns
    train, *_ = build_splits(
        df,
        "yr",
        0.2,
        0.1,
        0,
        drop_columns=["instant", "dteday", "casual", "registered"],
    )
    for c in ("casual", "registered"):
        assert c not in train.columns


def test_drift_perturbation_changes_distribution():
    from src.data.split import inject_drift
    df = _toy()
    yr1 = df[df["yr"] == 1].copy()
    drifted = inject_drift(yr1, factor_temp=1.1, factor_hum=0.85, std_windspeed=0.05, seed=0)
    assert drifted["temp"].mean() > yr1["temp"].mean()
    assert drifted["hum"].mean() < yr1["hum"].mean()
    assert drifted["windspeed"].std() > yr1["windspeed"].std()
