# Complete Project Audit + 120/120 Rubric Submission Plan

> **Deprecation:** The Cursor **`@write-plan` command is deprecated** and will be removed in a future Cursor release. For new plans, ask the assistant to apply the **`writing-plans`** skill (`C:\Users\bigbo\.cursor\skills\writing-plans\SKILL.md`): bite-sized checkbox tasks, concrete commands, no vague steps.

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a complete, evidence-based audit package **and** close every graded requirement so the submission can defensibly score **120/120** — **100** base rubric plus **Bonus A (+10)** Docker and **Bonus B (+10)** orchestration — with reproducible commands and screenshots/docs as proof.

**Architecture:** Phase A preserves the deterministic audit toolchain (inventory → skeleton → filled audit → checklist → backlog → scorecard). Phase B is a rubric-aligned “close-out” pass: map each syllabus criterion to artifacts in-repo, run verification commands, capture evidence (`docs/screenshots/`, exported MLflow artifacts), fix gaps before final submission.

**Tech Stack:** Python 3.11, scikit-learn, DVC, MLflow, FastAPI, GitHub Actions, Evidently, Prometheus, pytest, Prefect, Docker Compose, ruff.

---

## Grading Rubric Coverage Matrix (→ 120/120)

| Component | Points | Grading criteria (from syllabus) | Primary evidence targets in this repo |
|-----------|-------:|-----------------------------------|---------------------------------------|
| 1 — Data versioning | 10 | Pipeline runs cleanly; **three artifacts** tracked; `dvc repro` deterministic; **remote configured & documented** | `dvc.yaml`, `.dvc/config` or docs for remote URL, `dvc.lock`; outs e.g. `data/splits/preprocessor.pkl`, `data/splits/model.pkl`, parquet splits tracked via `.dvc` where applicable |
| 2 — Preprocessing | 12 | **sklearn `Pipeline`** complete & serialized; **all steps config-driven**; preprocessor **DVC-tracked**; **≥3 unit tests** passing | `src/features/preprocessor.py`, `configs/params.yaml`, `tests/unit/test_preprocessor*.py`, `tests/unit/test_featurize_main.py` |
| 3 — Experiments & registry | 15 | **≥3 experiments** fully logged; **HPO applied**; best model **registered** and **promoted to Production via API**; metrics & artifacts complete | `src/training/train.py`, `src/training/hpo.py`, `src/training/registry.py`, MLflow UI or `scripts/export_runs.py` output |
| 4 — Serving | 15 | `/health` & `/predict`; model loads **at startup**; **input validation**; **test script passes** | `src/serving/app.py`, `src/serving/schemas.py`, `tests/test_api.py` or `tests/unit/test_serving.py`, `scripts/validate_model.py` if applicable |
| 5 — CI/CD | 13 | **All four CI stages passing**; **coverage ≥70%**; **branch protection active**; **green CI screenshot** | `.github/workflows/ci.yml` (`lint`, `test`, `data-validation`, `model-validation`); README screenshot path; repo settings screenshot |
| 6 — Monitoring & drift | 15 | **Two Evidently reports**; **drift threshold logic**; **drift simulation documented**; **Prometheus metrics operational** | `monitoring/evidently_reports/baseline.html`, `drift.html`; `monitoring/drift_logic.py`; `monitoring/prometheus/prometheus.yml`; `docker-compose.yml` service wiring |
| 7 — Documentation | 10 | README **Quickstart**; **Model Card** & **Data Card**; **MLflow experiment log exported** | `README.md`, `docs/model_card.md`, `docs/data_card.md`, exported CSV/JSON from MLflow (`scripts/export_runs.py` or documented export) |
| 8 — Setup & reproducibility | 10 | **Pinned `requirements.txt`**; **all params** in `configs/params.yaml`; **`.gitignore` correct**; **clean-install Quickstart verified** | `requirements.txt`, `configs/params.yaml`, `.gitignore`; README steps from fresh venv |
| **Bonus A** — Docker | +10 | **Dockerfile builds**; **Compose starts serving app**; `/health` responds | `docker/api.Dockerfile` (and `docker/mlflow.Dockerfile` if used), root `docker-compose.yml` |
| **Bonus B** — Orchestration | +10 | **DAG ≥5 tasks**; **schedulable**; **failure handling shown** | `flows/training_flow.py`, `prefect.yaml`, screenshots of failed run after threshold tweak (`configs/params.yaml` `validation.min_test_r2`) |

**Self-review (rubric sanity):** Every row above MUST have either (a) a “PASS with evidence location” filled in Phase B, or (b) a dedicated close-out task with an owner and command.

---

## File Structure Map (Locked Before Execution)

- `docs/plan/2026-05-05-complete-project-audit.md` — this plan (canonical path for this workspace).
- `docs/audits/2026-05-05-file-by-file-audit.md` — file-by-file audit (generated + filled).
- `docs/audits/2026-05-05-project-checklist.md` — checklist including **§6 Rubric**.
- `docs/audits/2026-05-05-critical-issues.md` — blocking issues vs 120/120.
- `docs/audits/2026-05-05-improvements.md` — non-blocking polish.
- `docs/audits/2026-05-05-scorecard.md` — **numeric /120**.
- `docs/screenshots/` — CI green run, Prefect schedules/failures, optional MLflow/evidently thumbnails (names documented in README).
- `scripts/audit/build_inventory.py` — deterministic inventory generator.
- `scripts/audit/generate_audit_skeleton.py` — scaffold sections.
- `tests/scripts/audit/test_inventory.py`, `tests/scripts/audit/test_skeleton.py`.

---

### Task 1: Create Deterministic File Inventory

**Files:**
- Create: `scripts/audit/build_inventory.py`
- Create: `tests/scripts/audit/test_inventory.py`
- Create: `docs/audits/2026-05-05-inventory.txt`

- [ ] **Step 1: Write the failing test**

```python
from scripts.audit.build_inventory import collect_files

def test_collect_files_is_sorted_and_unique(tmp_path):
    (tmp_path / "b.py").write_text("print('b')", encoding="utf-8")
    (tmp_path / "a.py").write_text("print('a')", encoding="utf-8")
    files = collect_files(tmp_path)
    assert files == sorted(set(files))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/scripts/audit/test_inventory.py::test_collect_files_is_sorted_and_unique -v`  
Expected: FAIL with `ModuleNotFoundError` or missing `collect_files`.

- [ ] **Step 3: Write minimal implementation**

```python
from pathlib import Path

EXCLUDED_DIRS = {".git", ".dvc/cache", ".venv", ".venv_strict", "__pycache__", ".pytest_cache", ".ruff_cache"}

def collect_files(root: Path) -> list[str]:
    items: list[str] = []
    for path in root.rglob("*"):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            if any(part in EXCLUDED_DIRS for part in rel.split("/")):
                continue
            items.append(rel)
    return sorted(set(items))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/scripts/audit/test_inventory.py -v`  
Expected: PASS.

- [ ] **Step 5: Generate inventory artifact**

Run: `python -m scripts.audit.build_inventory --root . --out docs/audits/2026-05-05-inventory.txt`  
Expected: file exists; excludes cache dirs listed in `EXCLUDED_DIRS`.

- [ ] **Step 6: Commit**

```bash
git add scripts/audit/build_inventory.py tests/scripts/audit/test_inventory.py docs/audits/2026-05-05-inventory.txt
git commit -m "chore(audit): add deterministic repository inventory generator"
```

### Task 2: Generate Audit Skeleton So No File Is Missed

**Files:**
- Create: `scripts/audit/generate_audit_skeleton.py`
- Create: `tests/scripts/audit/test_skeleton.py`
- Create: `docs/audits/2026-05-05-file-by-file-audit.md`

- [ ] **Step 1: Write the failing test**

```python
from scripts.audit.generate_audit_skeleton import render_section

def test_render_section_contains_required_headings():
    section = render_section("src/data/prepare.py")
    assert "## src/data/prepare.py" in section
    assert "### Purpose" in section
    assert "### Line-by-line findings" in section
    assert "### Exact code fixes" in section
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/scripts/audit/test_skeleton.py::test_render_section_contains_required_headings -v`  
Expected: FAIL due to missing module/function.

- [ ] **Step 3: Write minimal implementation**

```python
def render_section(path: str) -> str:
    return (
        f"## {path}\n\n"
        "### Purpose\n"
        "(fill)\n\n"
        "### Line-by-line findings\n"
        "(fill)\n\n"
        "### Exact code fixes\n"
        "(fill)\n"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/scripts/audit/test_skeleton.py -v`  
Expected: PASS.

- [ ] **Step 5: Generate full skeleton**

Run: `python -m scripts.audit.generate_audit_skeleton --inventory docs/audits/2026-05-05-inventory.txt --out docs/audits/2026-05-05-file-by-file-audit.md`  
Expected: one section per inventory path.

- [ ] **Step 6: Commit**

```bash
git add scripts/audit/generate_audit_skeleton.py tests/scripts/audit/test_skeleton.py docs/audits/2026-05-05-file-by-file-audit.md
git commit -m "chore(audit): scaffold file-by-file audit document"
```

### Task 3: Perform Complete File-by-File Audit

**Files:**
- Modify: `docs/audits/2026-05-05-file-by-file-audit.md`

- [ ] **Step 1: Audit infrastructure/config first**

Audit order:

1. Root configs: `.gitignore`, `requirements.txt`, `dvc.yaml`, `dvc.lock`, `docker-compose.yml`, `configs/params.yaml`, `prefect.yaml`
2. CI: `.github/workflows/ci.yml`, `.github/workflows/monitoring-drift.yml`
3. DVC-tracked outputs: `.dvc` files under `data/`

- [ ] **Step 2: For each file, fill subsections**

```markdown
## <path>
### Purpose
### Line-by-line findings
### Exact code fixes
```

- [ ] **Step 3: Verify completeness**

Run: `python -m scripts.audit.generate_audit_skeleton --verify-complete docs/audits/2026-05-05-file-by-file-audit.md`  
Expected: PASS when automation exists; otherwise peer-review checklist confirms no skipped paths.

- [ ] **Step 4: Commit**

```bash
git add docs/audits/2026-05-05-file-by-file-audit.md
git commit -m "docs(audit): complete file-by-file technical audit with code-level fixes"
```

### Task 4: Project Checklist Including Full §6 Rubric

**Files:**
- Create: `docs/audits/2026-05-05-project-checklist.md`

- [ ] **Step 1: Copy this checklist into `docs/audits/2026-05-05-project-checklist.md` and set PASS/FAIL per line**

```markdown
## §6.1 Rubric component 1 — Data versioning (10)
- [ ] `python -m dvc repro` (or `dvc repro`) completes without error
- [ ] Three (or more) distinct versioned artifacts demonstrably tracked (.dvc or `outs` reproducible hashes)
- [ ] Repeated `dvc repro` stable (deterministic deps; document seed in `configs/params.yaml`)
- [ ] Remote storage configured: show `dvc remote list` OR `.dvc/config` + README “DVC Remote” subsection

## §6.1 Rubric component 2 — Preprocessing pipeline (12)
- [ ] Fitted preprocessing is a sklearn Pipeline object serialized to tracked path (`data/splits/preprocessor.pkl`)
- [ ] Imputer/feature steps pull parameters only from config (`configs/params.yaml`) — no stray literals
- [ ] Artifact appears in correct DVC stage (`dvc.yaml` `featurize` outs)
- [ ] Unit tests ≥3 touching preprocessor/pipeline behaviors pass: e.g.
  `pytest tests/unit/test_preprocessor.py tests/unit/test_featurize_main.py tests/unit/test_prepare_main.py -q`

## §6.1 Rubric component 3 — Experiments & registry (15)
- [ ] MLflow logs ≥3 distinguishable experiments or runs with params + metrics (`MLFLOW_TRACKING_URI` documented)
- [ ] HPO code path exercised (`src/training/hpo.py`) — capture run IDs or screenshots
- [ ] Model registered AND Production promotion via Tracking/Registry API (`src/training/registry.py`): document CLI or screenshot

## §6.1 Rubric component 4 — Serving (15)
- [ ] `/health` returns success when model loaded (`src/serving/app.py`)
- [ ] `/predict` validates payloads (`src/serving/schemas.py`) and runs inference
- [ ] Automated tests green: `pytest tests/test_api.py tests/unit/test_serving.py -q`

## §6.1 Rubric component 5 — CI/CD (13)
- [ ] Four jobs passing on default branch PR/push: `lint`, `test`, `data-validation`, `model-validation` (see `ci.yml`)
- [ ] Coverage gate ≥70% enforced (`--cov-fail-under=70`)
- [ ] Branch protection screenshot (Rulesets or classic branch protection requiring CI) saved as `docs/screenshots/ci-branch-protection.png`
- [ ] Green workflow screenshot `docs/screenshots/ci-green-main.png`

## §6.1 Rubric component 6 — Monitoring & drift (15)
- [ ] Two Evidently HTML artifacts committed or regenerated reproducibly: `monitoring/evidently_reports/baseline.html`, `monitoring/evidently_reports/drift.html`
- [ ] Drift threshold semantics documented + implemented (`monitoring/drift_logic.py` + README “Drift simulation”)
- [ ] Prometheus config scrapes API (`monitoring/prometheus/prometheus.yml`) matches `docker-compose.yml` network names

## §6.1 Rubric component 7 — Documentation (10)
- [ ] README Quickstart ≤ actionable steps install → train/serve smoke
- [ ] Model card (`docs/model_card.md`) & data card (`docs/data_card.md`) complete sections for intended use/limitations
- [ ] MLflow export attached: run `python scripts/export_runs.py` (if that is repo standard) OR export UI table to `docs/mlflow/export.md` — link from README

## §6.1 Rubric component 8 — Reproducibility (10)
- [ ] Fully pinned/third-party installs from `requirements.txt` in README commands
- [ ] No orphaned tunables outside `configs/params.yaml`
- [ ] `.gitignore` excludes venv/cache/MLflow clutter but retains required pickles per course policy — verify `git status` clean policy

## Bonus A — Docker (+10)
- [ ] `docker compose build api` succeeds
- [ ] `docker compose up api` → `curl http://localhost:8000/health` (or documented port)

## Bonus B — Orchestration (+10)
- [ ] Prefect flow has ≥5 tasks (this repo: `validate_data`, `preprocess`, `train`, `evaluate`, `register_model` in `flows/training_flow.py`)
- [ ] Deployment schedulable: `prefect deploy --prefect-file prefect.yaml` (after work pool creation) documented
- [ ] Failure handling demo: intentionally fail `evaluate` via `configs/params.yaml` `validation.min_test_r2`; screenshot Prefect/UI state as `docs/screenshots/prefect-failure-handling.png`; restore sane threshold afterward
```

- [ ] **Step 2: Run local verification sweep and paste command excerpts into checklist**

Commands (Windows-friendly):

```powershell
python -m ruff check src/ tests/
python -m pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=70
python -m dvc dag
python -m dvc repro
```

Expected: transcripts or summarized PASS next to corresponding checklist rows.

- [ ] **Step 3: Commit**

```bash
git add docs/audits/2026-05-05-project-checklist.md
git commit -m "docs(audit): add full §6 rubric checklist targeting 120/120"
```

### Task 5: Critical Issues vs 120 and Improvement Backlog

**Files:**
- Create: `docs/audits/2026-05-05-critical-issues.md`
- Create: `docs/audits/2026-05-05-improvements.md`

- [ ] **Step 1: List top issues that would cap score below stated row** (examples: MLflow unreachable in CI blocking registry proof; Evidently reports missing; Dockerfile build fails on grader laptop).

- [ ] **Step 2: List improvements worth ≤10 points combined** — polish after blockers cleared.

- [ ] **Step 3: Commit**

### Task 6: Scorecard (/120)

**Files:**
- Create: `docs/audits/2026-05-05-scorecard.md`

Score section format:

```markdown
| Component | Max | Earned | Evidence |
|-----------|-----|--------|----------|
| 1 DVC | 10 | | |
...
| Bonus A Docker | +10 | | |
| Bonus B Prefect | +10 | | |
| **Total** | **120** | | |

## Verdict
- Ready for submission YES/NO — blocking IDs from critical list if NO
```

### Task 7: Quality Gate (“No drift in audit docs themselves”)

- [ ] `rg "implement later|TBD \\(fill pending\\)" docs/audits/` returns no accidental placeholders in **completed** narratives (automated scaffolding may intentionally mark unfilled segments until Task 3 finishes).
- [ ] Scorecard totals agree with checklist all-PASS assumptions.
- [ ] README cross-links screenshots and checklist path `docs/audits/2026-05-05-project-checklist.md`.

```bash
git add docs/audits/
git commit -m "docs(audit): finalize scorecard + quality gate for 120/120 readiness"
```

### Task 8: DVC Remote & Artifact Narrative (closes Component 1)

**Files:** README subsection, optionally `.dvc/config`

- [ ] Demonstrate exactly **three named artifacts graders can grep for** (for example): `bike_clean.parquet` pipeline output, `preprocessor.pkl`, `model.pkl`.
- [ ] Document `dvc remote add`/`dvc remote default` and `dvc push` expectation for reproducibility bonus if course requires remote (S3/Azure/GDrive).

Commit when remote URL sanitized (no secrets in repo).

### Task 9: MLflow Experiment & Registry Evidence (closes Component 3)

**Files:** `README.md`, `scripts/export_runs.py` output artifact

- [ ] Capture **≥3 runs** screenshots or CSV export listing `run_id`, params, metrics.
- [ ] Capture **staging → production** promotion (UI or scripted `MlflowClient` transcript).

### Task 10: CI/CD & Branch Protection Evidence (closes Component 5)

**Files:** `docs/screenshots/`, README

- [ ] Attach **branch protection** evidence (Classic or Rulesets UI) — graders often checklist this explicitly.

### Task 11: Monitoring Stack Smoke (closes Component 6 + ties Bonus A Compose)

**Files:** README monitoring section only if gaps found

```powershell
docker compose up prometheus api
# verify metrics endpoint behaves as README states
```

- [ ] Re-run evidently report generation commands documented in README if HTML drifted.

### Task 12: Submission README Pass (“single source of truth”)

**Files:** `README.md`

- [ ] Confirm Quickstart installs from clean venv (`python -m venv .venv`, `pip install -r requirements.txt`, then train/serve/docker/prefect optional blocks).
- [ ] Embed table mapping **screenshot filenames** ↔ **§6 criterion**.

---

## Self-Review (against §6 Rubric spec)

### 1) Spec coverage
- Each **§6.1 row** mapped in the matrix plus Task 4 checklist and Task 8–12 closes common proof gaps (**remote**, **MLflow export**, **branch protection**, **Compose monitoring**).

### 2) Placeholder scan
- Executable steps cite real paths (`docker-compose.yml`, `flows/training_flow.py`, `monitoring/evidently_reports/*.html`).
- Skeleton generator uses `(fill)` not open-ended prose.

### 3) Signature/path consistency
- Inventory and skeleton modules remain canonical for Phase A verification.

---

Plan complete (`docs/plan/2026-05-05-complete-project-audit.md`). **Execution options:**

1. **Subagent-driven** — one subagent per task; review between tasks (**uses `superpowers:subagent-driven-development`**).

2. **Inline** — run tasks in-session with checkpoints (**uses `superpowers:executing-plans`**).

Which approach do you want?
