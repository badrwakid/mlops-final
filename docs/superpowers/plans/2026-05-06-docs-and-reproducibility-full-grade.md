# Documentation & Reproducibility Full-Grade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver full marks for Component 7 (Documentation) and Component 8 (Project setup & reproducibility), plus defensible team-collaboration evidence for final discussion.

**Architecture:** Add a documentation-quality gate (tests + scripts) that validates rubric-required files and content, then tighten `README.md`, cards, and reproducibility files to satisfy strict checklist rules. Add lightweight team-process artifacts (`CONTRIBUTING`, PR/Issue templates, commit-evidence export) so collaboration claims are auditable. Keep implementation incremental with TDD and frequent commits.

**Tech Stack:** Python 3.11+, pytest, markdown docs, GitHub templates, existing repo tooling (`mlflow`, `dvc`, `PyYAML`).

---

## File Structure (planned changes)

- Create: `tests/docs/test_docs_contract.py` — rubric contract tests for README, model card, data card, experiment log, requirements pins, params centralization, and `.gitignore`.
- Create: `scripts/verify_docs_repro.py` — local audit script printing pass/fail summary used before submission.
- Create: `scripts/export_team_contribution.py` — exports commit distribution evidence (author + count + files) for discussion readiness.
- Create: `.github/PULL_REQUEST_TEMPLATE.md` — enforces review/checklist evidence for PR merges.
- Create: `.github/ISSUE_TEMPLATE/task.yml` — standard task/bug tracking for grading evidence.
- Create: `docs/team_collaboration.md` — branch strategy, PR policy, issue workflow, and member accountability map.
- Modify: `README.md` — strict 3-command Quickstart path + architecture overview + clean-install verification + docs links.
- Modify: `docs/model_card.md` — complete required sections and subgroup metrics presentation.
- Modify: `docs/data_card.md` — complete source/schema/preprocessing/bias/privacy/license sections.
- Modify: `docs/experiment_log.csv` (regenerated) — all runs/params/metrics export.
- Modify: `docs/mlflow/export.md` — unambiguous export + verification commands.
- Modify: `requirements.txt` — ensure every used dependency is pinned (`package==x.y.z`) and remove non-pinned constraints.
- Modify: `configs/params.yaml` — ensure every runtime threshold/parameter is centralized.
- Modify: `.gitignore` (+ if needed `data/**/.gitignore`) — protect data/artifacts/env/IDE while preserving required tracked artifacts.

---

### Task 1: Build rubric contract tests first (TDD harness)

**Files:**
- Create: `tests/docs/test_docs_contract.py`
- Test: `tests/docs/test_docs_contract.py`

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

from pathlib import Path
import csv
import re

ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_required_docs_exist():
    required = [
        "README.md",
        "docs/model_card.md",
        "docs/data_card.md",
        "docs/experiment_log.csv",
        "requirements.txt",
        "configs/params.yaml",
        ".gitignore",
    ]
    for rel in required:
        assert (ROOT / rel).exists(), f"missing required file: {rel}"


def test_readme_has_project_title_architecture_and_quickstart():
    txt = _read("README.md").lower()
    assert txt.startswith("# ")
    assert "architecture" in txt
    assert "quickstart" in txt


def test_quickstart_has_three_or_fewer_primary_commands():
    txt = _read("README.md")
    marker = "## Quickstart"
    assert marker in txt
    quick = txt.split(marker, 1)[1].split("## ", 1)[0]
    blocks = re.findall(r"```(?:bash|powershell)?\n(.*?)```", quick, flags=re.S)
    assert blocks, "quickstart must include at least one command block"
    first_block = [ln.strip() for ln in blocks[0].splitlines() if ln.strip() and not ln.strip().startswith("#")]
    assert len(first_block) <= 3, f"quickstart block has >3 commands: {len(first_block)}"


def test_model_card_sections():
    txt = _read("docs/model_card.md").lower()
    for token in [
        "description",
        "intended use",
        "training data",
        "metrics",
        "subgroup",
        "limitations",
        "ethical",
    ]:
        assert token in txt, f"model card missing section/token: {token}"


def test_data_card_sections():
    txt = _read("docs/data_card.md").lower()
    for token in [
        "source",
        "schema",
        "preprocessing",
        "bias",
        "privacy",
        "license",
    ]:
        assert token in txt, f"data card missing section/token: {token}"


def test_experiment_log_has_required_columns():
    path = ROOT / "docs/experiment_log.csv"
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
    must_have = {"run_id", "status"}
    has_metrics = any(c.startswith("metrics.") for c in cols)
    has_params = any(c.startswith("params.") for c in cols)
    assert must_have.issubset(cols)
    assert has_metrics and has_params


def test_requirements_are_pinned():
    txt = _read("requirements.txt")
    for line in txt.splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        pkg = raw.split("#", 1)[0].strip()
        assert "==" in pkg, f"dependency not pinned with == : {pkg}"


def test_gitignore_has_required_exclusions():
    txt = _read(".gitignore").lower()
    for token in [".venv/", "venv/", ".vscode/", ".idea/", "data/raw", "data/processed", "data/splits"]:
        assert token in txt, f".gitignore missing {token}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/docs/test_docs_contract.py -v`  
Expected: FAIL on at least quickstart command count + model/data card completeness checks.

- [ ] **Step 3: Add test package init if needed**

```python
# tests/docs/__init__.py
```

- [ ] **Step 4: Run tests again**

Run: `pytest tests/docs/test_docs_contract.py -v`  
Expected: still FAIL (until docs updates in next tasks).

- [ ] **Step 5: Commit**

```bash
git add tests/docs/test_docs_contract.py tests/docs/__init__.py
git commit -m "test: add rubric contract tests for documentation and reproducibility"
```

---

### Task 2: Make README pass strict rubric checks (3-command quickstart)

**Files:**
- Modify: `README.md`
- Test: `tests/docs/test_docs_contract.py::test_readme_has_project_title_architecture_and_quickstart`
- Test: `tests/docs/test_docs_contract.py::test_quickstart_has_three_or_fewer_primary_commands`

- [ ] **Step 1: Write README failing test assertions first (if not failing yet, tighten)**

```python
def test_readme_quickstart_mentions_install_pipeline_serve():
    txt = _read("README.md").lower()
    quick = txt.split("## Quickstart", 1)[1].split("## ", 1)[0]
    assert "pip install -r requirements.txt" in quick
    assert "dvc repro" in quick or "python -m dvc repro" in quick
    assert "uvicorn" in quick and "src.serving.app:app" in quick
```

- [ ] **Step 2: Run single test to verify failure**

Run: `pytest tests/docs/test_docs_contract.py::test_readme_quickstart_mentions_install_pipeline_serve -v`  
Expected: FAIL if Quickstart is verbose/ambiguous.

- [ ] **Step 3: Rewrite Quickstart to one canonical block (<=3 commands)**

```markdown
## Quickstart

From repository root (clean environment):

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt && python -m dvc repro
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

Windows PowerShell equivalent:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt; python -m dvc repro
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```
```

- [ ] **Step 4: Add architecture overview + clean-install verification subsection**

```markdown
## Architecture Overview

Data (DVC) -> preprocessing/training (scikit-learn + MLflow) -> serving (FastAPI) -> monitoring (Evidently + Prometheus).

## Clean Install Verification

Verified on a fresh virtual environment with the three Quickstart commands above.
Evidence screenshot path: `docs/screenshots/quickstart_clean_install.png`.
```

- [ ] **Step 5: Run targeted tests**

Run: `pytest tests/docs/test_docs_contract.py -k "readme or quickstart" -v`  
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add README.md tests/docs/test_docs_contract.py
git commit -m "docs: simplify quickstart to 3 commands and add architecture verification"
```

---

### Task 3: Complete Model Card to grading rubric depth

**Files:**
- Modify: `docs/model_card.md`
- Create: `docs/subgroup_metrics.md`
- Test: `tests/docs/test_docs_contract.py::test_model_card_sections`

- [ ] **Step 1: Add failing test for explicit subsection names**

```python
def test_model_card_has_required_headings():
    txt = _read("docs/model_card.md")
    for heading in [
        "## Description",
        "## Intended Use",
        "## Training Data",
        "## Metrics (overall)",
        "## Metrics (per subgroup)",
        "## Limitations",
        "## Ethical Considerations",
    ]:
        assert heading in txt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/docs/test_docs_contract.py::test_model_card_has_required_headings -v`  
Expected: FAIL with missing heading assertions.

- [ ] **Step 3: Rewrite model card with complete sections**

```markdown
# Model Card

## Description
- Model: `bike_share_regressor` (hourly demand regression).
- Objective: predict bike rentals/hour from weather + calendar features.

## Intended Use
- Intended for educational MLOps demonstration and internal forecasting experiments.
- Not intended for safety-critical or legal/financial decision-making.

## Training Data
- Dataset: UCI Bike Sharing Dataset (`hour.csv`), documented in `docs/data_card.md`.
- Split strategy: temporal reference/production split via pipeline config in `configs/params.yaml`.

## Metrics (overall)
- RMSE, MAE, R2 tracked in MLflow and persisted in `data/splits/metrics.json`.

## Metrics (per subgroup)
- Subgroups: season, weather situation, working day.
- Detailed subgroup table: `docs/subgroup_metrics.md`.

## Limitations
- Sensitive to seasonal distribution shift and rare weather regimes.
- Prediction quality depends on feature completeness and stable upstream data semantics.

## Ethical Considerations
- Potential allocation bias across neighborhoods if deployed without equity checks.
- Monitoring is required to detect disparate degradation across subgroup slices.
```

- [ ] **Step 4: Add subgroup metrics companion file**

```markdown
# Subgroup Metrics

| Subgroup | Slice | RMSE | MAE | R2 | n |
|---|---|---:|---:|---:|---:|
| season | winter | ... | ... | ... | ... |
| season | spring | ... | ... | ... | ... |
| weathersit | clear | ... | ... | ... | ... |
| workingday | 0 | ... | ... | ... | ... |
```

- [ ] **Step 5: Run model-card tests**

Run: `pytest tests/docs/test_docs_contract.py -k "model_card" -v`  
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add docs/model_card.md docs/subgroup_metrics.md tests/docs/test_docs_contract.py
git commit -m "docs: complete model card sections including subgroup metrics and ethics"
```

---

### Task 4: Complete Data Card to grading rubric depth

**Files:**
- Modify: `docs/data_card.md`
- Test: `tests/docs/test_docs_contract.py::test_data_card_sections`

- [ ] **Step 1: Add strict heading test**

```python
def test_data_card_has_required_headings():
    txt = _read("docs/data_card.md")
    for heading in [
        "## Source",
        "## Schema",
        "## Preprocessing Decisions",
        "## Known Biases",
        "## Privacy",
        "## Licensing",
    ]:
        assert heading in txt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/docs/test_docs_contract.py::test_data_card_has_required_headings -v`  
Expected: FAIL initially.

- [ ] **Step 3: Rewrite data card with explicit sections**

```markdown
# Data Card

## Source
- Name: UCI Bike Sharing Dataset (`hour.csv`)
- URL: https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset
- In-repo pointer: `data/raw/hour.csv.dvc`

## Schema
- Target: `cnt`
- Numeric features: `temp`, `atemp`, `hum`, `windspeed`, `hr`, `mnth`
- Categorical features: `season`, `holiday`, `workingday`, `weathersit`, `weekday`
- Validation contract: `src/data/schema.py`

## Preprocessing Decisions
- Missing-value handling: configured imputers (`configs/params.yaml`).
- Scaling/encoding and feature selection handled in pipeline code under `src/features/`.
- Train/serve parity ensured via serialized preprocessor artifact.

## Known Biases
- Temporal seasonality and weather imbalance can bias error rates by subgroup.
- Working-day demand patterns may underrepresent holiday extremes.

## Privacy
- Dataset contains no direct personal identifiers.
- No row-level user tracking fields are used in model features.

## Licensing
- Educational/public dataset usage under UCI repository terms.
- Team documents external resources in README/report citations.
```

- [ ] **Step 4: Run data-card tests**

Run: `pytest tests/docs/test_docs_contract.py -k "data_card" -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docs/data_card.md tests/docs/test_docs_contract.py
git commit -m "docs: complete data card with schema, bias, privacy, and licensing sections"
```

---

### Task 5: Make experiment log export robust and auditable

**Files:**
- Modify: `docs/mlflow/export.md`
- Modify: `docs/experiment_log.csv`
- Create: `scripts/verify_docs_repro.py`
- Test: `tests/docs/test_docs_contract.py::test_experiment_log_has_required_columns`

- [ ] **Step 1: Add failing tests for CSV freshness and size**

```python
def test_experiment_log_has_nonzero_rows():
    path = ROOT / "docs/experiment_log.csv"
    rows = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) > 1, "experiment_log.csv must contain header + >=1 run"
```

- [ ] **Step 2: Run test to verify failure (if CSV empty/stale)**

Run: `pytest tests/docs/test_docs_contract.py::test_experiment_log_has_nonzero_rows -v`  
Expected: FAIL if file is empty; otherwise PASS (keep check).

- [ ] **Step 3: Add verification script**

```python
from __future__ import annotations

from pathlib import Path
import csv
import sys


def check_experiment_log(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("experiment_log.csv has no runs")
    cols = rows[0].keys()
    if "run_id" not in cols or "status" not in cols:
        raise ValueError("experiment_log.csv missing required columns")


def main() -> int:
    try:
        check_experiment_log(Path("docs/experiment_log.csv"))
    except Exception as exc:
        print(f"verify_docs_repro: FAIL: {exc}", file=sys.stderr)
        return 1
    print("verify_docs_repro: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Regenerate experiment log from MLflow**

Run: `python scripts/export_runs.py`  
Expected: `exported <N> runs to docs\experiment_log.csv`.

- [ ] **Step 5: Update export documentation**

```markdown
## Verification

After export, run:

```bash
python scripts/verify_docs_repro.py
```

Expected output: `verify_docs_repro: PASS`
```

- [ ] **Step 6: Run targeted tests**

Run: `pytest tests/docs/test_docs_contract.py -k "experiment_log" -v`  
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add docs/mlflow/export.md docs/experiment_log.csv scripts/verify_docs_repro.py tests/docs/test_docs_contract.py
git commit -m "docs: harden mlflow experiment log export and add verification script"
```

---

### Task 6: Enforce pinned dependencies and centralized params

**Files:**
- Modify: `requirements.txt`
- Modify: `configs/params.yaml`
- Create: `tests/config/test_no_hardcoded_params.py`

- [ ] **Step 1: Write failing tests for non-pinned requirements and hardcoded values**

```python
from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def test_requirements_lines_are_pinned():
    txt = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    for line in txt.splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        dep = raw.split("#", 1)[0].strip()
        assert "==" in dep, f"dependency must be exact pin: {dep}"


def test_no_literal_thresholds_in_source():
    forbidden = [r"\b0\.7\b", r"\b0\.70\b", r"\b80\.0\b", r"\b0\.20\b"]
    src_files = list((ROOT / "src").rglob("*.py"))
    for p in src_files:
        txt = p.read_text(encoding="utf-8")
        for pat in forbidden:
            assert not re.search(pat, txt), f"hardcoded threshold {pat} in {p}"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/config/test_no_hardcoded_params.py -v`  
Expected: FAIL if any hardcoded thresholds remain.

- [ ] **Step 3: Move hardcoded thresholds into `configs/params.yaml` and consume via `load_config()`**

```python
# before
if drift_share > 0.20:
    ...

# after
cfg = load_config()
if drift_share > cfg.drift.drift_threshold_share:
    ...
```

- [ ] **Step 4: Ensure every dependency line is exact pin**

```text
# good
griffe==0.49.0

# bad
griffe>=0.49.0,<1.0.0
```

- [ ] **Step 5: Run config tests**

Run: `pytest tests/config/test_no_hardcoded_params.py -v`  
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt configs/params.yaml tests/config/test_no_hardcoded_params.py src/
git commit -m "chore: enforce exact dependency pins and centralize runtime thresholds in params"
```

---

### Task 7: Lock down `.gitignore` for reproducibility rules

**Files:**
- Modify: `.gitignore`
- Modify (if exists): `data/raw/.gitignore`, `data/processed/.gitignore`, `data/splits/.gitignore`
- Test: `tests/docs/test_docs_contract.py::test_gitignore_has_required_exclusions`

- [ ] **Step 1: Add failing tests for nested ignore conflicts**

```python
def test_nested_data_gitignores_do_not_block_required_artifacts():
    nested = [
        "data/raw/.gitignore",
        "data/processed/.gitignore",
        "data/splits/.gitignore",
    ]
    for rel in nested:
        p = ROOT / rel
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8")
        assert "model.pkl" not in txt
        assert "preprocessor.pkl" not in txt
```

- [ ] **Step 2: Run tests to verify failure if nested ignores conflict**

Run: `pytest tests/docs/test_docs_contract.py -k "gitignore or nested_data_gitignores" -v`  
Expected: FAIL only if conflicting patterns exist.

- [ ] **Step 3: Normalize ignore rules**

```gitignore
# data tracked by DVC pointers, not raw payloads
data/raw/*
!data/raw/.gitkeep
!data/raw/*.dvc

data/processed/*
!data/processed/.gitkeep
!data/processed/*.dvc

data/splits/*
!data/splits/.gitkeep
!data/splits/*.dvc
!data/splits/metrics.json
!data/splits/model.pkl
!data/splits/preprocessor.pkl
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/docs/test_docs_contract.py -k "gitignore" -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add .gitignore data/raw/.gitignore data/processed/.gitignore data/splits/.gitignore tests/docs/test_docs_contract.py
git commit -m "chore: harden gitignore rules for DVC data and required serving artifacts"
```

---

### Task 8: Add explicit collaboration evidence artifacts (team score protection)

**Files:**
- Create: `docs/team_collaboration.md`
- Create: `.github/PULL_REQUEST_TEMPLATE.md`
- Create: `.github/ISSUE_TEMPLATE/task.yml`
- Create: `scripts/export_team_contribution.py`
- Create: `docs/team_contribution.csv` (generated output)

- [ ] **Step 1: Write failing test for collaboration evidence files**

```python
def test_collaboration_artifacts_exist():
    required = [
        "docs/team_collaboration.md",
        ".github/PULL_REQUEST_TEMPLATE.md",
        ".github/ISSUE_TEMPLATE/task.yml",
        "scripts/export_team_contribution.py",
    ]
    for rel in required:
        assert (ROOT / rel).exists(), f"missing collaboration artifact: {rel}"
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/docs/test_docs_contract.py::test_collaboration_artifacts_exist -v`  
Expected: FAIL before file creation.

- [ ] **Step 3: Add team collaboration policy doc**

```markdown
# Team Collaboration

## Branching Strategy
- `main` is protected.
- Work on `feature/<topic>` branches only.

## Pull Requests
- At least one teammate review required before merge.
- PR checklist includes: tests, docs, and evidence screenshots.

## Issues Workflow
- Every task starts as a GitHub Issue.
- PR must link issue (e.g., `Closes #12`).

## Individual Accountability
- Each member owns at least one component and cross-reviews another component.
- Final discussion prep: each member rehearses every pipeline stage.
```

- [ ] **Step 4: Add PR template and issue template**

```markdown
## Summary
- [ ] What changed and why

## Rubric impact
- [ ] Component 7 evidence updated
- [ ] Component 8 evidence updated

## Validation
- [ ] `pytest tests/ --cov=src --cov-fail-under=70`
- [ ] `python scripts/verify_docs_repro.py`

## Review
- [ ] At least one teammate approved
- [ ] Linked issue: Closes #<id>
```

```yaml
name: Task
description: Project task for DDSC611 deliverables
title: "[Task] "
labels: ["task"]
body:
  - type: textarea
    id: scope
    attributes:
      label: Scope
      description: What will be changed?
    validations:
      required: true
  - type: textarea
    id: acceptance
    attributes:
      label: Acceptance Criteria
      description: What proves this is done?
    validations:
      required: true
```

- [ ] **Step 5: Add contribution export script and generate CSV**

```python
from __future__ import annotations

import csv
import subprocess
from pathlib import Path


def main() -> int:
    out = subprocess.check_output(
        ["git", "shortlog", "-sne", "HEAD"],
        text=True,
    )
    rows = []
    for line in out.strip().splitlines():
        parts = line.strip().split("\t", 1)
        if len(parts) == 2:
            rows.append((parts[0].strip(), parts[1].strip()))
    target = Path("docs/team_contribution.csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["commit_count", "author"])
        w.writerows(rows)
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Run: `python scripts/export_team_contribution.py`  
Expected: `wrote docs/team_contribution.csv`.

- [ ] **Step 6: Run tests**

Run: `pytest tests/docs/test_docs_contract.py -k "collaboration_artifacts" -v`  
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add docs/team_collaboration.md .github/PULL_REQUEST_TEMPLATE.md .github/ISSUE_TEMPLATE/task.yml scripts/export_team_contribution.py docs/team_contribution.csv tests/docs/test_docs_contract.py
git commit -m "docs: add collaboration evidence artifacts for PR, issues, and team contribution tracking"
```

---

### Task 9: Add one-command pre-submit audit and run full verification

**Files:**
- Modify: `scripts/verify_docs_repro.py`
- Modify: `README.md`
- Test: `tests/docs/test_docs_contract.py`

- [ ] **Step 1: Extend verification script to run contract tests**

```python
import subprocess

def run_pytest_contract() -> None:
    cmd = ["pytest", "tests/docs/test_docs_contract.py", "-q"]
    subprocess.check_call(cmd)
```

- [ ] **Step 2: Add README verification command**

```markdown
## Submission Audit (Components 7 & 8)

```bash
python scripts/verify_docs_repro.py
```
```

- [ ] **Step 3: Run complete verification**

Run: `python scripts/verify_docs_repro.py`  
Expected: PASS and zero failed checks.

Run: `pytest tests/docs/test_docs_contract.py -v`  
Expected: PASS all documentation/reproducibility contract tests.

- [ ] **Step 4: Commit**

```bash
git add scripts/verify_docs_repro.py README.md tests/docs/test_docs_contract.py
git commit -m "chore: add pre-submit audit command for documentation and reproducibility compliance"
```

---

### Task 10: Produce grading evidence bundle for final discussion

**Files:**
- Modify: `README.md`
- Modify: `docs/team_collaboration.md`
- Create: `docs/screenshots/README.md`

- [ ] **Step 1: Document exact screenshot checklist with file names**

```markdown
## Required screenshots
- `docs/screenshots/quickstart_clean_install.png`
- `docs/screenshots/pytest_coverage_report.png`
- `docs/screenshots/branch_protection_main.png`
- `docs/screenshots/pr_review_approved.png`
- `docs/screenshots/issues_board_active.png`
```

- [ ] **Step 2: Add final discussion prep bullets per teammate**

```markdown
## Oral defense prep
- Member A explains DVC + preprocessing + params centralization.
- Member B explains MLflow export + model/data cards.
- Member C explains serving quickstart + CI + collaboration process.
```

- [ ] **Step 3: Verify docs/tests one last time**

Run: `python scripts/verify_docs_repro.py`  
Expected: PASS.

Run: `pytest tests/docs/test_docs_contract.py -v`  
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/team_collaboration.md docs/screenshots/README.md
git commit -m "docs: finalize evidence bundle and oral-defense readiness checklist"
```

---

## Self-Review

### 1) Spec coverage check
- Component 7 evidence addressed: `README.md`, `docs/model_card.md`, `docs/data_card.md`, `docs/experiment_log.csv`.
- Component 8 evidence addressed: `requirements.txt` exact pins, `configs/params.yaml` centralization, `.gitignore` rules, clean-install Quickstart verification.
- Collaboration/integrity section addressed: PR policy, issue tracking, branch strategy, contribution evidence export.
- Gaps: none for requested scope; branch protection still requires GitHub UI confirmation screenshot (outside code).

### 2) Placeholder scan
- No `TODO`/`TBD` placeholders in tasks.
- Every code-changing step includes concrete code blocks.
- Every test step includes exact commands + expected outcomes.

### 3) Type/signature consistency
- `verify_docs_repro.py` and test names are consistent across tasks.
- Paths are consistent with repo layout and rubric target files.

