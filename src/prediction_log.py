from __future__ import annotations

import csv
import datetime as dt
import threading
from pathlib import Path

LOG_PATH = Path("artifacts/prediction_log.csv")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
_LOCK = threading.Lock()

FEATURE_COLS = [
    "season",
    "yr",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
    "temp",
    "atemp",
    "hum",
    "windspeed",
]
FIELDS = ["timestamp"] + FEATURE_COLS + ["prediction"]


def log_prediction(features: dict, prediction: float) -> None:
    row = {"timestamp": dt.datetime.utcnow().isoformat(), "prediction": float(prediction)}
    for key in FEATURE_COLS:
        row[key] = features.get(key)
    with _LOCK:
        new_file = not LOG_PATH.exists()
        with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
