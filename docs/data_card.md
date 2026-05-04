# Data Card

## Dataset
- **Name:** Bike Sharing Dataset (hour-level)
- **Raw file pointer:** `data/raw/hour.csv.dvc`
- **Versioning:** DVC + Git metadata (`dvc.yaml`, `dvc.lock`)

## Schema & Validation
- Raw schema is validated with Pandera (`src/data/schema.py`).
- CI includes data validation tests in `tests/data/`.

## Processing Pipeline
- `prepare` → `preprocess` → `featurize` → `train` (DVC stages).
- Outputs include cleaned data, train/test/reference/production splits, and fitted artifacts.

## Features
- Numeric and categorical features defined in `configs/params.yaml`.
- Includes cyclical encoding for temporal features (`hr`, `mnth`).

## Splits
- Temporal split logic implemented in `src/data/split.py`.
- Separate reference and production windows used for monitoring/drift analysis.

## Known Constraints
- Artifacts are generated and tracked via DVC flow; large binaries are not directly committed.
- Data drift is expected and monitored with threshold-based alerts.

