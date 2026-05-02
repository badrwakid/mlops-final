# monitoring/run_monitoring.py
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    RegressionPreset,
)
from evidently.report import Report
from src.config import load_config

from monitoring.drift_logic import drift_alert

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _add_predictions(df: pd.DataFrame, model, preprocessor, feature_cols: list[str], target: str) -> pd.DataFrame:
    X = preprocessor.transform(df[feature_cols])
    df = df.copy()
    df["prediction"] = model.predict(X)
    df = df.rename(columns={target: "target"})
    return df


def _drift_per_feature_from_report(report: Report) -> dict[str, bool]:
    result = report.as_dict()
    out: dict[str, bool] = {}
    for metric in result["metrics"]:
        if metric.get("metric") == "DataDriftTable":
            by_col = metric["result"].get("drift_by_columns", {})
            for col, info in by_col.items():
                out[col] = bool(info.get("drift_detected", False))
            break
    return out


def main() -> None:
    cfg = load_config()
    model = joblib.load(cfg.paths.model)
    preprocessor = joblib.load(cfg.paths.preprocessor)
    reference = pd.read_parquet(cfg.paths.reference)
    production = pd.read_parquet(cfg.paths.production)

    # baseline: first half of reference as ref, second half as clean current
    half = len(reference) // 2
    production_clean = reference.iloc[half:].copy()
    ref = reference.iloc[:half].copy()

    feature_cols = cfg.data.numeric_features + cfg.data.categorical_features
    ref = _add_predictions(ref, model, preprocessor, feature_cols, cfg.data.target)
    prod_clean = _add_predictions(production_clean, model, preprocessor, feature_cols, cfg.data.target)
    prod_drift = _add_predictions(production, model, preprocessor, feature_cols, cfg.data.target)

    out_dir = Path("monitoring/evidently_reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    presets = [DataDriftPreset(), DataQualityPreset(), RegressionPreset()]

    baseline_report = Report(metrics=presets)
    baseline_report.run(reference_data=ref, current_data=prod_clean)
    baseline_report.save_html(str(out_dir / "baseline.html"))
    log.info("baseline report saved → %s", out_dir / "baseline.html")

    drift_report = Report(metrics=presets)
    drift_report.run(reference_data=ref, current_data=prod_drift)
    drift_report.save_html(str(out_dir / "drift.html"))
    log.info("drift report saved → %s", out_dir / "drift.html")

    drift_map = _drift_per_feature_from_report(drift_report)
    result = drift_alert(drift_map, threshold_share=cfg.drift.drift_threshold_share)
    summary = {
        "alert": result.alert,
        "drift_share": result.drift_share,
        "drifted_features": result.drifted_features,
        "threshold": cfg.drift.drift_threshold_share,
    }
    with open(out_dir / "drift_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("drift summary: %s", summary)

    if result.alert:
        log.warning(
            "DRIFT DETECTED: %.0f%% of features drifted (>%.0f%% threshold). Drifted: %s",
            result.drift_share * 100,
            cfg.drift.drift_threshold_share * 100,
            ", ".join(result.drifted_features),
        )
    else:
        log.info("no drift alert (share=%.2f)", result.drift_share)


if __name__ == "__main__":
    main()
