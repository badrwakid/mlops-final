"""Smoke coverage for src.features.featurize.main (mocked IO)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import src.features.featurize as featurize_mod


def _toy_train(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(1)
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


def test_featurize_main_writes_preprocessor_and_scores(tmp_path, monkeypatch):
    train_df = _toy_train()
    fitted = MagicMock(name="pipeline")

    def fake_fit(*_args, **_kwargs):
        return fitted

    dumped: list[tuple] = []

    def fake_dump(obj, path):
        dumped.append((obj, Path(path)))
        Path(path).write_bytes(b"fake-pkl")

    monkeypatch.setattr(featurize_mod.pd, "read_parquet", lambda _path: train_df)
    monkeypatch.setattr(featurize_mod, "fit_preprocessor", fake_fit)
    monkeypatch.setattr(featurize_mod.joblib, "dump", fake_dump)

    cfg = SimpleNamespace(
        paths=SimpleNamespace(
            train="/does/not/matter/train.parquet",
            preprocessor=str(tmp_path / "preprocessor.pkl"),
        ),
        data=SimpleNamespace(
            target="cnt",
            numeric_features=[
                "temp",
                "atemp",
                "hum",
                "windspeed",
                "hr",
                "mnth",
            ],
            categorical_features=[
                "season",
                "holiday",
                "workingday",
                "weathersit",
                "weekday",
            ],
        ),
        preprocessing=SimpleNamespace(
            feature_selection_k=10,
            numeric_imputer_strategy="median",
            categorical_imputer_strategy="most_frequent",
            cyclical_hr_mnth=True,
            feature_scores_json=None,
        ),
    )

    monkeypatch.setattr(featurize_mod, "load_config", lambda: cfg)

    featurize_mod.main()

    assert dumped and dumped[0][0] is fitted
    assert (tmp_path / "preprocessor.pkl").exists()


def test_featurize_main_calls_export_when_scores_path_set(
    tmp_path: Path, monkeypatch, request
) -> None:
    train_df = _toy_train()
    fitted = MagicMock(name="pipeline")
    called: list[Path] = []

    def fake_fit(*_a, **_k):
        return fitted

    def fake_dump(obj, path):
        Path(path).write_bytes(b"fake-pkl")

    def capture_export(_pipe, path: Path) -> None:
        called.append(Path(path))

    monkeypatch.setattr(featurize_mod.pd, "read_parquet", lambda _path: train_df)
    monkeypatch.setattr(featurize_mod, "fit_preprocessor", fake_fit)
    monkeypatch.setattr(featurize_mod.joblib, "dump", fake_dump)
    monkeypatch.setattr(featurize_mod, "export_selector_feature_scores", capture_export)

    cfg = SimpleNamespace(
        paths=SimpleNamespace(
            train="/x/train.parquet",
            preprocessor=str(tmp_path / "preprocessor.pkl"),
        ),
        data=SimpleNamespace(
            target="cnt",
            numeric_features=[
                "temp",
                "atemp",
                "hum",
                "windspeed",
                "hr",
                "mnth",
            ],
            categorical_features=[
                "season",
                "holiday",
                "workingday",
                "weathersit",
                "weekday",
            ],
        ),
        preprocessing=SimpleNamespace(
            feature_selection_k=10,
            numeric_imputer_strategy="median",
            categorical_imputer_strategy="most_frequent",
            cyclical_hr_mnth=True,
            feature_scores_json="docs/.pytest_feat_scores_tmp.json",
        ),
    )
    monkeypatch.setattr(featurize_mod, "load_config", lambda: cfg)

    featurize_mod.main()

    assert len(called) == 1
    assert called[0].name == ".pytest_feat_scores_tmp.json"

    def cleanup():
        p = called[0]
        if p.exists():
            p.unlink()

    request.addfinalizer(cleanup)
