# Critical issues blocking a confident 120/120 claim

Each item can cap the posted score until resolved. IDs are stable for the scorecard verdict.

## C1 — Submission screenshots incomplete

- **Severity:** High (rubric §6.1 component 5 — branch protection + green CI evidence)
- **Evidence:** `docs/screenshots/` contains only `.gitkeep` (no PNGs committed at audit time).
- **Fix:** Capture and commit the files already named in `README.md` (`pytest_coverage_report.png`, `branch_protection_main.png`, `dvc_repro_deterministic.png`, Bonus A/B PNGs as listed).
- **Validation:** `git ls-files docs/screenshots/*.png` returns at least the CI + coverage pair.

## C2 — Local DVC CLI may break on system Python (`pathspec`)

- **Severity:** Medium (blocks “run these commands on your laptop” story for §6 component 1 evidence)
- **Evidence:** `python -m dvc dag` failed with `_DIR_MARK` / `pathspec` when using a user-wide Python 3.12 site-packages that pins `pathspec` 1.x.
- **Fix:** Use the project venv and pinned `requirements.txt` (`pathspec==0.11.2`). Run `python -m dvc …` from that environment only; document in checklist if graders use a fresh venv.
- **Validation:** `.venv\Scripts\python.exe -m dvc dag` prints the DAG (or `python -m dvc` after `pip install -r requirements.txt` in a clean venv).

## C3 — `dvc repro` not re-run during this audit session

- **Severity:** Medium-high for component 1 narrative until proven on submitter machine
- **Evidence:** Audit session did not execute a full `dvc repro` (data/MLflow prerequisites vary by machine).
- **Fix:** On a clean checkout: fetch `hour.csv` (see `README.md`), then run `python -m dvc repro` and attach `docs/screenshots/dvc_repro_deterministic.png`.
- **Validation:** `dvc.lock` updates only when stages change; graders expect a clean repro log.
