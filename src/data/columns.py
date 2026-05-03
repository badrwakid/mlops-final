"""Shared column helpers for prepare/split stages."""

from __future__ import annotations

import pandas as pd


def drop_configured_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Drop any of ``columns`` that exist on ``df`` (no-op if none present)."""
    to_drop = [c for c in columns if c in df.columns]
    return df.drop(columns=to_drop) if to_drop else df
