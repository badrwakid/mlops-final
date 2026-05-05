# Numeric scorecard (120-point scale)

| Component | Max | Earned | Evidence / notes |
|-----------|-----|--------|-------------------|
| 1 — DVC | 10 | 7* | Pipeline + remote documented; **`dvc repro` not rerun** this audit session; proves on your machine → 10 |
| 2 — Preprocessing | 12 | 12 | `dvc.yaml` featurize; `configs/params.yaml`; unit tests exist |
| 3 — Experiments & registry | 15 | 12* | Code + MLflow plumbing present; **export/screenshots pending** → full 15 with artifacts |
| 4 — Serving | 15 | 15 | FastAPI endpoints + tests GREEN |
| 5 — CI/CD | 13 | 9* | Workflow + coverage wired; **branch protection PNG + green CI PNG** pending |
| 6 — Monitoring & drift | 15 | 15 | Evidently HTML + Prometheus + drift logic documented |
| 7 — Documentation | 10 | 9* | README + cards; **exported MLflow CSV** pending if not committed |
| 8 — Reproducibility | 10 | 10 | Pinned reqs, params YAML, README |
| **Subtotal base** | **100** | **89*** | Items with `*` unblock to 100 |
| Bonus A — Docker | +10 | ? | Needs live Docker verification + screenshots |
| Bonus B — Prefect | +10 | ? | Code + YAML strong; Needs UI failure/success screenshots for full assurance |
| **Total** | **120** | **89+ ?** | **Verdict:** Not submission-final until **C1–C3** in `critical-issues.md` closed |

## Verdict

- **Ready for submission:** **NO** until PNG evidence + `dvc repro` log on a healthy venv are attached and `docs/experiment_log.csv` (or equivalent) proves ≥3 MLflow runs.
- **Blocking IDs:** **C1**, **C3** (screenshots / repro proof); optionally **C2** if graders hit `pathspec` on wrong interpreter.
