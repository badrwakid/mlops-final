# src/features/preprocessor.py
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric: list[str],
    categorical: list[str],
    k: int,
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
) -> Pipeline:
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
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocessor", column_transformer),
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
) -> Pipeline:
    X = df[numeric + categorical]
    y = df[target]
    pipe = build_preprocessor(
        numeric, categorical, k=k,
        numeric_strategy=numeric_strategy,
        categorical_strategy=categorical_strategy,
    )
    pipe.fit(X, y)
    return pipe
