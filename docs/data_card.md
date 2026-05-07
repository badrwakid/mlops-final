# Data Card

## Source
- **Dataset provenance:** UCI Bike Sharing Dataset (`hour.csv`) from [UCI dataset archive](https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip), fetched by `scripts/fetch_uci_hour_csv.py`.
- **In-repo provenance artifact:** DVC pointer `data/raw/hour.csv.dvc` (raw file path `hour.csv`, tracked hash metadata).
- **Versioning:** Pipeline/data lineage recorded in `dvc.yaml` and `dvc.lock`.
- **Scope:** The source bundle includes both `hour.csv` and `day.csv` (see `docs/source_readme.txt`), but this pipeline is intentionally configured for hourly modeling via `configs/params.yaml` (`data/raw/hour.csv`).

## Schema
- Raw data contract is defined as a Pandera `DataFrameSchema` (`raw_schema`) in `src/data/schema.py`.
- Validation is applied in data loading (`src/data/load.py`) and covered by tests in `tests/data/test_data_validation.py`.

## Preprocessing Decisions
- Data flow follows DVC stages (`prepare` -> `preprocess` -> `featurize` -> `train`) as defined in `dvc.yaml`.
- Feature settings (numeric/categorical and time-related transforms) are configured in `configs/params.yaml`.
- Temporal split behavior is implemented in `src/data/split.py` for train/reference/production style windows.

## Known Biases
- This dataset represents bike demand under historical local conditions; patterns may not generalize across cities/seasons.
- Hour, weather, and calendar effects can dominate outcomes, so underrepresented conditions may have higher error.

## Privacy
- The tracked tabular dataset used here is operational demand/weather style data and does not include direct personal identifiers in project artifacts (`data/raw/hour.csv.dvc`).
- Repository policy avoids committing large/raw data blobs directly; data artifacts are tracked through DVC metadata.

## Licensing
- **Status:** Upstream license is not independently re-verified in this repository.
- **Source-owner note:** The project uses the UCI-provided dataset package and mirrors its accompanying source notes in `docs/source_readme.txt` (including citation/license text from the dataset owner/distributor).
- This repository tracks a pointer (`data/raw/hour.csv.dvc`) rather than storing the raw `hour.csv` in Git.

