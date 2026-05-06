## MLflow naming convention — Bike Sharing project

### Experiment names
Format: `bike_share_<purpose>`
Examples:
- bike_share_training       — model training runs
- bike_share_tuning         — hyperparameter sweeps
- bike_share_drift          — Evidently drift reports
- bike_share_baselines      — persistence and hourly mean baselines

### Run names
Format: `<stage>__<algo_or_purpose>__<short_id>`

Where:
- stage: one of [train, tune, baseline, eval, drift, debug]
- algo: short model identifier (rf, xgb, lgbm, lr, persistence, hourly_mean)
  or purpose for non-training runs (drift_check, sanity)
- short_id: 8-char timestamp YYMMDDHH or version tag (v1, v2, exp03)

Examples:
- train__xgb__25050610
- baseline__persistence__25050610
- tune__rf__exp03
- drift__check__25050612
- eval__xgb_v1__final_test

### Required tags on every run
- stage          — same as in the name
- algo           — same as in the name
- dataset_hash   — first 12 chars of SHA-256 of the input data
- author         — student's identifier (e.g. "ali")
- git_commit     — short hash, if available

### Run descriptions
Every run gets a one-line `mlflow.note.content` describing what was tested.
Bad: "first try"
Good: "XGBoost 500 trees, max_depth=8, lr=0.05, all features. Beats persistence by 23% RMSE."
