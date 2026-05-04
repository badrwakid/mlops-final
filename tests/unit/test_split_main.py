"""Exercise src.data.split.main."""

from types import SimpleNamespace

import pandas as pd
from src.data.split import main


def _toy_df(n: int = 120) -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(2)
    return pd.DataFrame(
        {
            "yr": rng.integers(0, 2, size=n),
            "temp": rng.random(n),
            "hum": rng.random(n),
            "windspeed": rng.random(n),
            "cnt": rng.integers(1, 100, size=n),
            "season": rng.integers(1, 5, size=n),
            "holiday": rng.integers(0, 2, size=n),
            "workingday": rng.integers(0, 2, size=n),
            "weathersit": rng.integers(1, 5, size=n),
            "weekday": rng.integers(0, 7, size=n),
            "atemp": rng.random(n),
            "hr": rng.integers(0, 24, size=n),
            "mnth": rng.integers(1, 13, size=n),
        }
    )


def test_split_main_writes_split_parquets(tmp_path, monkeypatch):
    proc = tmp_path / "bike_clean.parquet"
    _toy_df().to_parquet(proc, index=False)

    train_p = tmp_path / "train.parquet"
    test_p = tmp_path / "test.parquet"
    ref_p = tmp_path / "reference.parquet"
    prod_p = tmp_path / "production.parquet"

    cfg = SimpleNamespace(
        paths=SimpleNamespace(
            processed=str(proc),
            train=str(train_p),
            test=str(test_p),
            reference=str(ref_p),
            production=str(prod_p),
        ),
        data=SimpleNamespace(
            split_column="yr",
            test_size=0.2,
            reference_holdout=0.1,
            random_state=0,
            drop_columns=None,
        ),
        drift=SimpleNamespace(
            perturb_temp_factor=1.1,
            perturb_hum_factor=0.9,
            perturb_windspeed_noise_std=0.01,
        ),
    )
    monkeypatch.setattr("src.data.split.load_config", lambda: cfg)

    main()

    for p in (train_p, test_p, ref_p, prod_p):
        assert p.exists()
        assert len(pd.read_parquet(p)) > 0
