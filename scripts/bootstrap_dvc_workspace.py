#!/usr/bin/env python3
"""Bootstrap DVC pipeline outputs for a fresh clone (no paid remote required).

Mirrors the free CI path in ``.github/workflows/ci.yml``:

1. Ensure ``data/raw/hour.csv`` exists via :mod:`scripts.fetch_uci_hour_csv` (MD5-checked).
2. If ``data/splits/model.pkl`` is missing, run ``dvc repro`` with
   ``MLFLOW_TRACKING_URI=file:./mlruns`` so training does not need a running MLflow server.

If you use DagsHub (or any DVC remote), prefer publishing and pulling artifacts instead::

    dvc repro
    dvc push

    # other machine / fresh clone
    dvc remote modify --local localremote password <DAGSHUB_TOKEN>
    dvc pull

Run from the repository root::

    python scripts/bootstrap_dvc_workspace.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    os.chdir(ROOT)
    model = ROOT / "data" / "splits" / "model.pkl"
    raw = ROOT / "data" / "raw" / "hour.csv"
    if not raw.exists():
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "fetch_uci_hour_csv.py")],
            check=True,
        )
    if model.exists():
        print(
            "OK: data/splits/model.pkl present — workspace already has trained artifacts. "
            "Run `python -m dvc status` if you expected changes."
        )
        return 0
    env = os.environ.copy()
    env.setdefault("MLFLOW_TRACKING_URI", "file:./mlruns")
    subprocess.run([sys.executable, "-m", "dvc", "repro"], check=True, env=env)
    print("OK: `dvc repro` finished; outputs should match dvc.lock.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
