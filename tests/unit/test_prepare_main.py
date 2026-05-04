"""Exercise src.data.prepare.main for coverage."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from src.data.prepare import main


def test_prepare_main_writes_parquet(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    sample = repo_root / "tests" / "data" / "sample_hour.csv"
    raw_dest = tmp_path / "hour.csv"
    raw_dest.write_bytes(sample.read_bytes())
    out_parquet = tmp_path / "bike_clean.parquet"

    cfg = SimpleNamespace(
        paths=SimpleNamespace(
            raw_csv=str(raw_dest),
            processed=str(out_parquet),
        ),
        data=SimpleNamespace(
            drop_columns=["instant", "dteday", "casual", "registered"],
        ),
    )
    monkeypatch.setattr("src.data.prepare.load_config", lambda: cfg)

    main()

    assert out_parquet.exists()
    df = pd.read_parquet(out_parquet)
    assert len(df) > 0
