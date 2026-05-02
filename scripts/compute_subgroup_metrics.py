from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import joblib
import pandas as pd
from src.config import load_config
from src.evaluation.metrics import compute_metrics

GROUP_FIELDS = ["season", "weathersit", "workingday"]
MIN_GROUP_N = 30
OUTPUT_PATH = Path("docs") / "subgroup_metrics.json"


def feature_columns(cfg) -> list[str]:
    return cfg.data.numeric_features + cfg.data.categorical_features


def build_subgroup_payload(
    df: pd.DataFrame,
    y_true: Sequence[float],
    y_pred: Sequence[float],
    group_fields: Sequence[str] = GROUP_FIELDS,
    min_n: int = MIN_GROUP_N,
) -> dict:
    scored = df.copy()
    scored["_y_true"] = list(y_true)
    scored["_y_pred"] = list(y_pred)

    payload = {
        "overall": compute_metrics(scored["_y_true"], scored["_y_pred"]),
        "subgroups": {},
    }

    for field in group_fields:
        field_groups = []
        for value, group in scored.groupby(field, sort=True):
            n = int(len(group))
            if n < min_n:
                continue
            metrics = compute_metrics(group["_y_true"], group["_y_pred"])
            field_groups.append({
                "value": value.item() if hasattr(value, "item") else value,
                "n": n,
                **metrics,
            })
        payload["subgroups"][field] = field_groups

    return payload


def write_subgroup_metrics(payload: dict, output_path: Path = OUTPUT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compute_and_write_subgroup_metrics(output_path: Path = OUTPUT_PATH) -> dict:
    cfg = load_config()
    model = joblib.load(cfg.paths.model)
    preprocessor = joblib.load(cfg.paths.preprocessor)
    test = pd.read_parquet(cfg.paths.test)

    X_test = test[feature_columns(cfg)]
    y_true = test[cfg.data.target]
    y_pred = model.predict(preprocessor.transform(X_test))

    payload = build_subgroup_payload(test, y_true, y_pred)
    write_subgroup_metrics(payload, output_path)
    print(f"saved subgroup metrics to {output_path}")
    return payload


def main() -> int:
    compute_and_write_subgroup_metrics()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
