"""Light tests for drift_report helpers (coverage for CI floor)."""

from __future__ import annotations

import pandas as pd
from src.drift_report import (
    _prepare_current_df,
    build_column_mapping,
)


def test_build_column_mapping():
    m = build_column_mapping(
        include_target=True,
        include_prediction=True,
        numerical_features=["temp"],
        categorical_features=["season"],
    )
    assert m.prediction == "prediction"
    assert m.target == "target"
    m2 = build_column_mapping(
        include_target=False,
        include_prediction=False,
        numerical_features=["temp"],
        categorical_features=["season"],
    )
    assert m2.prediction is None
    assert m2.numerical_features == m.numerical_features


def test_prepare_current_df_fills_and_numeric():
    df = pd.DataFrame({"season": [1], "prediction": [1.0]})
    out = _prepare_current_df(df, include_prediction=True)
    assert len(out) >= 1
    assert "prediction" in out.columns
