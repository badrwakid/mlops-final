"""Extra coverage for serving helpers (bins, forest confidence, hash)."""

import hashlib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.serving.app import (
    _feature_hash,
    _predict_with_confidence,
    _prediction_output_class,
)


def test_prediction_output_class_all_bins():
    assert _prediction_output_class(0.0) == "very_low"
    assert _prediction_output_class(99.0) == "very_low"
    assert _prediction_output_class(100.0) == "low"
    assert _prediction_output_class(350.0) == "medium"
    assert _prediction_output_class(500.0) == "high"
    assert _prediction_output_class(900.0) == "very_high"


def test_predict_with_confidence_uses_ensemble_std():
    X = np.random.default_rng(0).random((20, 3))
    y = X[:, 0] * 2
    rf = RandomForestRegressor(n_estimators=8, random_state=0)
    rf.fit(X, y)
    items = _predict_with_confidence(rf, X[:3])
    assert len(items) == 3
    for it in items:
        assert 0.0 <= it.confidence <= 1.0


def test_feature_hash_deterministic():
    df = pd.DataFrame([{"a": 1, "b": 2.0}])
    h1 = _feature_hash(df)
    h2 = _feature_hash(df)
    assert h1 == h2
    assert len(h1) == 64
    payload = df.to_json(orient="records", date_format="iso", double_precision=8)
    assert hashlib.sha256(payload.encode("utf-8")).hexdigest() == h1


def test_feature_hash_changes_with_data():
    df1 = pd.DataFrame([{"a": 1}])
    df2 = pd.DataFrame([{"a": 2}])
    assert _feature_hash(df1) != _feature_hash(df2)
