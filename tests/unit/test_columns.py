import pandas as pd
from src.data.columns import drop_configured_columns


def test_drop_configured_columns_removes_only_present():
    df = pd.DataFrame({"a": [1], "b": [2], "casual": [3]})
    out = drop_configured_columns(df, ["casual", "missing"])
    assert list(out.columns) == ["a", "b"]
    assert "casual" not in out.columns
