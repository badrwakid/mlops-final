import pandera as pa
from pandera import Column, DataFrameSchema

raw_schema = DataFrameSchema(
    {
        "instant": Column(int, unique=True),
        "dteday": Column(str),
        "season": Column(int, pa.Check.isin([1, 2, 3, 4])),
        "yr": Column(int, pa.Check.isin([0, 1])),
        "mnth": Column(int, pa.Check.in_range(1, 12)),
        "hr": Column(int, pa.Check.in_range(0, 23)),
        "holiday": Column(int, pa.Check.isin([0, 1])),
        "weekday": Column(int, pa.Check.in_range(0, 6)),
        "workingday": Column(int, pa.Check.isin([0, 1])),
        "weathersit": Column(int, pa.Check.isin([1, 2, 3, 4])),
        "temp": Column(float, pa.Check.in_range(0.0, 1.0)),
        "atemp": Column(float, pa.Check.in_range(0.0, 1.0)),
        "hum": Column(float, pa.Check.in_range(0.0, 1.0)),
        "windspeed": Column(float, pa.Check.in_range(0.0, 1.0)),
        "casual": Column(int, pa.Check.ge(0)),
        "registered": Column(int, pa.Check.ge(0)),
        "cnt": Column(int, pa.Check.ge(1)),
    },
    strict=True,
    coerce=True,
)
