from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_LOG_PATH = REPO_ROOT / "docs" / "experiment_log.csv"
REQUIRED_COLUMNS = ("run_id", "experiment_id", "status")


def verify_experiment_log(csv_path: Path = EXPERIMENT_LOG_PATH) -> list[str]:
    errors: list[str] = []

    if not csv_path.exists():
        return [f"Missing file: {csv_path}"]

    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            fieldnames = reader.fieldnames or []
            first_row = next(reader, None)
    except OSError as exc:
        return [f"Failed to read file: {csv_path} ({exc.__class__.__name__})"]
    except UnicodeDecodeError:
        return [f"Failed to decode CSV as UTF-8: {csv_path}"]
    except csv.Error:
        return [f"Invalid CSV format: {csv_path}"]

    if first_row is None:
        errors.append(f"No data rows found in {csv_path}")

    missing_required = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing_required:
        errors.append(f"Missing required columns: {', '.join(missing_required)}")

    if not any(column.startswith("params.") for column in fieldnames):
        errors.append("Missing params.* columns")

    if not any(column.startswith("metrics.") for column in fieldnames):
        errors.append("Missing metrics.* columns")

    return errors


def main() -> int:
    errors = verify_experiment_log()
    if errors:
        for error in errors:
            print(f"verify_docs_repro: {error}", file=sys.stderr)
        return 1

    docs_contract_cmd = [sys.executable, "-m", "pytest", "tests/docs/test_docs_contract.py", "-v"]
    docs_contract_result = subprocess.run(docs_contract_cmd, cwd=REPO_ROOT)
    if docs_contract_result.returncode != 0:
        print("verify_docs_repro: docs contract tests failed", file=sys.stderr)
        return docs_contract_result.returncode

    print(f"verify_docs_repro: ok ({EXPERIMENT_LOG_PATH})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
