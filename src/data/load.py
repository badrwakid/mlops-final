# src/data/load.py
from pathlib import Path

import pandas as pd

from src.data.schema import raw_schema


def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    raw_schema.validate(df)
    return df
