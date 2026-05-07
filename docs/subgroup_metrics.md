# Subgroup Metrics

This file summarizes subgroup evaluation for `bike_share_regressor` using `docs/subgroup_metrics.json`.

## Overall

| Split | N | RMSE | MAE | R2 |
| --- | ---: | ---: | ---: | ---: |
| Overall holdout | 1,556 | 66.71 | 45.06 | 0.757 |

## Current subgroup breakdown

| Group | Value | Label (project semantics) | N | RMSE | MAE | R2 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| season | 1 | Spring | 347 | 38.54 | 24.68 | 0.707 |
| season | 2 | Summer | 408 | 68.82 | 47.52 | 0.722 |
| season | 3 | Fall | 410 | 74.44 | 53.14 | 0.760 |
| season | 4 | Winter | 391 | 75.10 | 52.12 | 0.709 |
| weathersit | 1 | Clear/few clouds | 1,051 | 68.74 | 46.72 | 0.769 |
| weathersit | 2 | Mist/cloudy | 351 | 65.46 | 43.79 | 0.665 |
| weathersit | 3 | Light rain/snow | 153 | 54.49 | 36.80 | 0.684 |
| workingday | 0 | Non-working day | 504 | 82.64 | 58.87 | 0.602 |
| workingday | 1 | Working day | 1,052 | 57.54 | 38.45 | 0.824 |

Note: `weathersit=4` is absent in the current subgroup artifact because the evaluated holdout split has insufficient/no rows for that category.

## Template for future runs

Use this compact template when subgroup metrics are recomputed from a new run:

| Group | Value | Label | N | RMSE | MAE | R2 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| season | 1 | Spring | 0 | 0.00 | 0.00 | 0.000 | fill after evaluation |
| weathersit | 1 | Clear/few clouds | 0 | 0.00 | 0.00 | 0.000 | fill after evaluation |
| workingday | 1 | Working day | 0 | 0.00 | 0.00 | 0.000 | fill after evaluation |
