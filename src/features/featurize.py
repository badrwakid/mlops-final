# src/features/featurize.py
from pathlib import Path

import joblib
import pandas as pd

from src.config import load_config
from src.features.preprocessor import fit_preprocessor


def main() -> None:
    cfg = load_config()
    train = pd.read_parquet(cfg.paths.train)
    pipe = fit_preprocessor(
        train,
        target=cfg.data.target,
        numeric=cfg.data.numeric_features,
        categorical=cfg.data.categorical_features,
        k=cfg.preprocessing.feature_selection_k,
        numeric_strategy=cfg.preprocessing.numeric_imputer_strategy,
        categorical_strategy=cfg.preprocessing.categorical_imputer_strategy,
    )
    out = Path(cfg.paths.preprocessor)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out)
    print(f"featurize: saved fitted preprocessor to {out}")


if __name__ == "__main__":
    main()
