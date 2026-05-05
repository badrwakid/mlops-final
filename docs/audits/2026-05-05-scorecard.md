# Numeric scorecard (120-point scale)

| Component | Max | Earned | Evidence / notes |
|-----------|-----|--------|-------------------|
| 1 — DVC | 10 | 7* | Pipeline + remote + lock; **C2/C3** still block “fully proven” until venv + `dvc repro` evidence. **ZIP‑9** done (`registry.py` in `train` deps / `dvc.lock`). |
| 2 — Preprocessing | 12 | 12 | Unchanged. |
| 3 — Experiments & registry | 15 | 12* | **ZIP‑5 / ZIP‑8** tighten MLflow error handling; still need **MLflow export / ≥3 runs** for full marks. |
| 4 — Serving | 15 | 15 | **ZIP‑5** implemented; tests green. |
| 5 — CI/CD | 13 | 9* | **ZIP‑2** drift job now **bootstraps + caches** artifacts; still **C1** screenshots. |
| 6 — Monitoring & drift | 15 | 15 | **ZIP‑6** exit codes; drift workflow seeds data / DVC repro / monitoring on Actions. |
| 7 — Documentation | 10 | 9* | README + cards; export CSV optional until runs exist. |
| 8 — Reproducibility | 10 | 10 | **ZIP‑1** deferred (hashes not required by PDF). |
| **Subtotal base** | **100** | **90*** | Raise with PNG + **`dvc repro`** proof + MLflow export. |
| Bonus A — Docker | +10 | ? | Unchanged (live proof). |
| Bonus B — Prefect | +10 | ? | Unchanged (UI evidence). |
| **Total** | **120** | **see above** | See **ZIP merge notes** below. |

## ZIP merge implementation notes (2026-05-06)

| Item | Outcome |
|------|---------|
| ZIP‑2 | **Done** — `continue-on-error` removed; workflow fetches raw data, caches DVC outputs, `dvc repro` on cache miss with `file:./mlruns`. |
| ZIP‑5 | **Done** — `src/serving/app.py` catches `MlflowException` only; warning before pickle fallback. |
| ZIP‑6 | **Done** — `monitoring/run_monitoring.py` → `SystemExit(2)` when artifacts missing + test. |
| ZIP‑8 | **Done** — `scripts/validate_model.py` narrows registry load errors. |
| ZIP‑9 | **Done** — `dvc.yaml` + **`dvc.lock`** include `src/training/registry.py`. |
| ZIP‑1 | **DEFER** — optional hash-locked installs; cite if instructor asks beyond pinned `requirements.txt`. |
| ZIP‑7 | **DEFER** — keep `[tool.coverage.run] omit = …` until tests cover omitted modules and keep ≥70%. |
| ZIP‑10 | **N/A** — serving via `docker/api.Dockerfile`. |

## Verdict

- **Ready for submission:** **NO** until **C1**, **C3**, and MLflow run evidence satisfy §6 — same as prior verdict; codebase side of **MLOPS_AUDITS ZIP backlog** is **mostly closed**.
- **Blocking IDs:** **C1**, **C3**; **C2** if graders use broken global Python.
