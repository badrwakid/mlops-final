import pandas as pd
import pytest
from src.data.schema import raw_schema


def _valid_row():
    return {
        "instant": 1, "dteday": "2011-01-01", "season": 1, "yr": 0, "mnth": 1,
        "hr": 0, "holiday": 0, "weekday": 6, "workingday": 0, "weathersit": 1,
        "temp": 0.24, "atemp": 0.288, "hum": 0.81, "windspeed": 0.0,
        "casual": 3, "registered": 13, "cnt": 16,
    }


def test_schema_accepts_valid_row():
    df = pd.DataFrame([_valid_row()])
    raw_schema.validate(df)


def test_schema_rejects_bad_season():
    bad = _valid_row()
    bad["season"] = 7
    df = pd.DataFrame([bad])
    with pytest.raises(Exception):
        raw_schema.validate(df)


def test_schema_rejects_zero_or_negative_count():
    bad = _valid_row()
    bad["cnt"] = 0
    df = pd.DataFrame([bad])
    with pytest.raises(Exception):
        raw_schema.validate(df)
