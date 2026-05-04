"""CyclicalHrMnthEncoder.get_feature_names_out and export_selector_feature_scores."""

from pathlib import Path

import numpy as np
import pandas as pd
from src.features.preprocessor import (
    CyclicalHrMnthEncoder,
    export_selector_feature_scores,
    fit_preprocessor,
)


def _toy(n: int = 200):
    rng = np.random.default_rng(3)
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


def test_get_feature_names_out_requires_input_features():
    import pytest

    enc = CyclicalHrMnthEncoder(enabled=True)
    with pytest.raises(ValueError, match="input_features"):
        enc.get_feature_names_out(None)


def test_get_feature_names_out_expands_hr_mnth_when_enabled():
    enc = CyclicalHrMnthEncoder(enabled=True)
    out = enc.get_feature_names_out(["hr", "mnth", "temp"])
    assert set(out.tolist()) >= {"hr_sin", "hr_cos", "mnth_sin", "mnth_cos", "temp"}


def test_get_feature_names_out_disabled_returns_same_names():
    enc = CyclicalHrMnthEncoder(enabled=False)
    out = enc.get_feature_names_out(["a", "b"])
    assert list(out) == ["a", "b"]


def test_export_selector_feature_scores_writes_json(tmp_path: Path):
    df = _toy()
    pipe = fit_preprocessor(
        df,
        target="cnt",
        numeric=["temp", "atemp", "hum", "windspeed", "hr", "mnth"],
        categorical=["season", "holiday", "workingday", "weathersit", "weekday"],
        k=8,
    )
    out_json = tmp_path / "scores.json"
    export_selector_feature_scores(pipe, out_json)
    text = out_json.read_text(encoding="utf-8")
    assert "f_score" in text and "k_selected" in text


def test_export_selector_raises_on_name_length_mismatch(tmp_path: Path):
    df = _toy()
    pipe = fit_preprocessor(
        df,
        target="cnt",
        numeric=["temp", "atemp", "hum", "windspeed", "hr", "mnth"],
        categorical=["season", "holiday", "workingday", "weathersit", "weekday"],
        k=8,
    )
    pipe.named_steps["selector"].scores_ = np.array([1.0, 2.0])
    import pytest

    with pytest.raises(ValueError, match="match"):
        export_selector_feature_scores(pipe, tmp_path / "bad.json")
