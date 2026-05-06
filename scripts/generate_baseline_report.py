from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    err = y_true.to_numpy(dtype=float) - y_pred.to_numpy(dtype=float)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(np.square(err))))
    y = y_true.to_numpy(dtype=float)
    ss_res = float(np.sum(np.square(err)))
    ss_tot = float(np.sum(np.square(y - np.mean(y))))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    denom = (np.abs(y) + np.abs(y_pred.to_numpy(dtype=float))) / 2.0
    smape = float(np.mean(np.where(denom > 0, np.abs(err) / denom, 0.0)) * 100.0)
    return {"mae": mae, "rmse": rmse, "r2": r2, "smape": smape}


def _persistence_baseline(train: pd.DataFrame, test: pd.DataFrame, lag_hours: int = 168) -> pd.Series:
    full = pd.concat([train, test], axis=0).reset_index(drop=True)
    full["yhat"] = full["cnt"].shift(lag_hours)
    yhat = full.iloc[len(train) :]["yhat"].copy()
    return yhat.fillna(train["cnt"].mean())


def _hourly_mean_baseline(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    table = train.groupby(["hr", "weekday", "season"], dropna=False)["cnt"].mean()
    fallback = float(train["cnt"].mean())
    return test.apply(lambda r: float(table.get((r["hr"], r["weekday"], r["season"]), fallback)), axis=1)


def main() -> None:
    train_path = Path("data/splits/train.parquet")
    test_path = Path("data/splits/test.parquet")
    metrics_path = Path("data/splits/metrics.json")
    out_path = Path("docs/baseline_report.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    y_true = test["cnt"].astype(float)
    persistence = _metrics(y_true, _persistence_baseline(train, test))
    hourly_mean = _metrics(y_true, _hourly_mean_baseline(train, test))

    model_metrics = {}
    if metrics_path.exists():
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        model_metrics = payload.get("test", {})

    model_rmse = float(model_metrics.get("rmse", np.nan))
    lift = (
        ((persistence["rmse"] - model_rmse) / persistence["rmse"]) * 100.0
        if np.isfinite(model_rmse) and persistence["rmse"] > 0
        else np.nan
    )

    out_path.write_text(
        "\n".join(
            [
                "# Baseline report",
                "",
                "Comparison on the held-out test split.",
                "",
                "| Model | MAE | RMSE | R² | sMAPE |",
                "| --- | ---: | ---: | ---: | ---: |",
                f"| Persistence (t-168) | {persistence['mae']:.3f} | {persistence['rmse']:.3f} | {persistence['r2']:.3f} | {persistence['smape']:.2f}% |",
                f"| Hourly mean | {hourly_mean['mae']:.3f} | {hourly_mean['rmse']:.3f} | {hourly_mean['r2']:.3f} | {hourly_mean['smape']:.2f}% |",
                f"| Trained model (v1) | {float(model_metrics.get('mae', np.nan)):.3f} | {model_rmse:.3f} | {float(model_metrics.get('r2', np.nan)):.3f} | n/a |",
                "",
                (
                    f"Lift over persistence (RMSE): {lift:.2f}%."
                    if np.isfinite(lift)
                    else "Lift over persistence (RMSE): unavailable (missing model metrics)."
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
