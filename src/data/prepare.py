# src/data/prepare.py
from pathlib import Path

from src.config import load_config
from src.data.load import load_raw


def main() -> None:
    cfg = load_config()
    df = load_raw(cfg.paths.raw_csv)
    df = df.drop(columns=cfg.data.drop_columns)
    out = Path(cfg.paths.processed)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"prepare: wrote {len(df):,} rows to {out}")


if __name__ == "__main__":
    main()
