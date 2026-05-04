"""Coverage for src.data.load.load_raw."""

from pathlib import Path

import pandas as pd
from src.data.load import load_raw


def test_load_raw_reads_and_validates_sample(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    sample = repo_root / "tests" / "data" / "sample_hour.csv"
    df = load_raw(sample)
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 10
    assert "cnt" in df.columns
