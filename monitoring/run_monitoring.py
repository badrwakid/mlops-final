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
from src.serving.metrics import DRIFT_PSI

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


def _feature_weights_from_scores(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("all_features_ranked_by_f_score", [])
    raw: dict[str, float] = {}
    for row in rows:
        name = str(row.get("feature", ""))
        score = float(row.get("f_score", 0.0))
        # Convert SelectKBest names back to original feature names where possible.
        if name.startswith("num__"):
            clean = name.removeprefix("num__")
            clean = clean.removesuffix("_sin").removesuffix("_cos")
        elif name.startswith("cat__"):
            clean = name.removeprefix("cat__").split("_", 1)[0]
        else:
            clean = name
        raw[clean] = max(raw.get(clean, 0.0), score)
    if not raw:
        return {}
    max_score = max(raw.values()) or 1.0
    # Normalize into [0.2, 1.0] to avoid zeroing less important features.
    return {k: 0.2 + 0.8 * (v / max_score) for k, v in raw.items()}


def main() -> None:
    cfg = load_config()
    out_dir = Path("monitoring/evidently_reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    required = {
        "model": Path(cfg.paths.model),
        "preprocessor": Path(cfg.paths.preprocessor),
        "reference": Path(cfg.paths.reference),
        "production": Path(cfg.paths.production),
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        summary = {
            "skipped": True,
            "reason": "missing_required_artifacts",
            "missing": missing,
            "hint": "Run `dvc pull` or `dvc repro` before monitoring.",
        }
        with open(out_dir / "drift_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log.warning("monitoring skipped: missing artifacts: %s", ", ".join(missing))
        return

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
    exclude = {"yr", "target", "prediction"}
    score_weights = _feature_weights_from_scores(Path("docs/feature_scores.json"))
    result = drift_alert(
        drift_map,
        threshold_share=cfg.drift.drift_threshold_share,
        exclude_cols=exclude,
        feature_weights=score_weights,
    )
    input_drift_map = {k: v for k, v in drift_map.items() if k not in exclude}
    for feature, is_drifted in input_drift_map.items():
        DRIFT_PSI.labels(feature=feature).set(1.0 if is_drifted else 0.0)
    summary = {
        "alert": result.alert,
        "drift_share": result.drift_share,
        "drift_share_inputs_only": (
            (sum(1 for v in input_drift_map.values() if v) / len(input_drift_map))
            if input_drift_map
            else 0.0
        ),
        "drifted_features": result.drifted_features,
        "excluded_from_share": sorted(exclude),
        "threshold": cfg.drift.drift_threshold_share,
        "severity": "P2" if result.alert else "P3",
        "policy": (
            "Trigger retraining when input drift_share exceeds threshold in 2 consecutive windows; "
            "investigate feature pipelines immediately."
        ),
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
