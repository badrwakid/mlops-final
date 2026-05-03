# src/features/preprocessor.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _effective_numeric(numeric: list[str], cyclical_hr_mnth: bool) -> list[str]:
    """Map config numeric feature names to post-encoder names when cyclical is on."""
    if not cyclical_hr_mnth:
        return list(numeric)
    out: list[str] = []
    for c in numeric:
        if c == "hr":
            out.extend(["hr_sin", "hr_cos"])
        elif c == "mnth":
            out.extend(["mnth_sin", "mnth_cos"])
        else:
            out.append(c)
    return out


class CyclicalHrMnthEncoder(BaseEstimator, TransformerMixin):
    """sin/cos encoding for hour (0–23) and month (1–12) per cyclical time features (Lecture 03b)."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)

    def fit(self, X, y=None):  # noqa: ANN001
        return self

    def transform(self, X):  # noqa: ANN001
        if not self.enabled:
            return X
        out = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if "hr" in out.columns:
            out["hr_sin"] = np.sin(2.0 * np.pi * out["hr"] / 24.0)
            out["hr_cos"] = np.cos(2.0 * np.pi * out["hr"] / 24.0)
            out = out.drop(columns=["hr"])
        if "mnth" in out.columns:
            ang = 2.0 * np.pi * (out["mnth"] - 1) / 12.0
            out["mnth_sin"] = np.sin(ang)
            out["mnth_cos"] = np.cos(ang)
            out = out.drop(columns=["mnth"])
        return out

    def get_feature_names_out(self, input_features=None):  # noqa: ANN001
        if input_features is None:
            raise ValueError("input_features is required")
        feats = list(input_features)
        if not self.enabled:
            return np.asarray(feats, dtype=object)
        out: list[str] = []
        for f in feats:
            if f == "hr":
                out.extend(["hr_sin", "hr_cos"])
            elif f == "mnth":
                out.extend(["mnth_sin", "mnth_cos"])
            else:
                out.append(f)
        return np.asarray(out, dtype=object)


def build_preprocessor(
    numeric: list[str],
    categorical: list[str],
    k: int,
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
    cyclical_hr_mnth: bool = True,
) -> Pipeline:
    effective_numeric = _effective_numeric(numeric, cyclical_hr_mnth)
    cyclical = CyclicalHrMnthEncoder(enabled=cyclical_hr_mnth)
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=numeric_strategy)),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=categorical_strategy)),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    column_transformer = ColumnTransformer(
        [
            ("num", numeric_pipe, effective_numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("cyclical", cyclical),
        ("column_prep", column_transformer),
        ("selector", SelectKBest(score_func=f_regression, k=k)),
    ])


def fit_preprocessor(
    df: pd.DataFrame,
    target: str,
    numeric: list[str],
    categorical: list[str],
    k: int,
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
    cyclical_hr_mnth: bool = True,
) -> Pipeline:
    X = df[numeric + categorical]
    y = df[target]
    pipe = build_preprocessor(
        numeric,
        categorical,
        k=k,
        numeric_strategy=numeric_strategy,
        categorical_strategy=categorical_strategy,
        cyclical_hr_mnth=cyclical_hr_mnth,
    )
    pipe.fit(X, y)
    return pipe


def export_selector_feature_scores(pipe: Pipeline, out_path: Path) -> None:
    """Write SelectKBest f-scores (pre-selection) to JSON for documentation."""
    column_prep = pipe.named_steps["column_prep"]
    feature_names = list(column_prep.get_feature_names_out())
    scores = np.asarray(pipe.named_steps["selector"].scores_, dtype=float)
    if len(feature_names) != len(scores):
        raise ValueError("feature name count does not match selector scores")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"feature": str(name), "f_score": float(s)}
        for name, s in zip(feature_names, scores, strict=True)
    ]
    top_k = int(pipe.named_steps["selector"].k)
    # order by score desc for readability
    rows.sort(key=lambda r: r["f_score"], reverse=True)
    payload = {
        "k_selected": top_k,
        "all_features_ranked_by_f_score": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
