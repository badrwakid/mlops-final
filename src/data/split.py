# src/data/split.py
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config
from src.data.columns import drop_configured_columns


def inject_drift(
    df: pd.DataFrame,
    factor_temp: float,
    factor_hum: float,
    std_windspeed: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    if "temp" in out.columns:
        out["temp"] = (out["temp"] * factor_temp).clip(0, 1)
    if "hum" in out.columns:
        out["hum"] = (out["hum"] * factor_hum).clip(0, 1)
    if "windspeed" in out.columns:
        out["windspeed"] = (out["windspeed"] + rng.normal(0, std_windspeed, len(out))).clip(0, 1)
    return out


def build_splits(
    df: pd.DataFrame,
    split_col: str,
    test_size: float,
    ref_holdout: float,
    random_state: int,
    drop_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if drop_columns:
        df = drop_configured_columns(df, drop_columns)
    year0 = df[df[split_col] == 0].copy()
    year1 = df[df[split_col] == 1].copy()
    # Temporal split: train/test/reference are year 0 only; production is raw year 1.
    # Random splits below are within year 0 — not stratified on cnt (regression); stratify=
    # is undefined for continuous targets without binning; quantile stratification is optional.
    year0_main, reference = train_test_split(
        year0, test_size=ref_holdout, random_state=random_state,
    )
    train, test = train_test_split(
        year0_main, test_size=test_size, random_state=random_state,
    )
    return train, test, reference, year1


def main() -> None:
    cfg = load_config()
    df = pd.read_parquet(cfg.paths.processed)
    train, test, reference, year1 = build_splits(
        df,
        split_col=cfg.data.split_column,
        test_size=cfg.data.test_size,
        ref_holdout=cfg.data.reference_holdout,
        random_state=cfg.data.random_state,
        drop_columns=cfg.data.drop_columns,
    )
    production = year1.copy()
    Path(cfg.paths.train).parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(cfg.paths.train, index=False)
    test.to_parquet(cfg.paths.test, index=False)
    reference.to_parquet(cfg.paths.reference, index=False)
    production.to_parquet(cfg.paths.production, index=False)
    print(
        f"split: train={len(train):,} test={len(test):,} "
        f"reference={len(reference):,} production={len(production):,}"
    )


if __name__ == "__main__":
    main()
