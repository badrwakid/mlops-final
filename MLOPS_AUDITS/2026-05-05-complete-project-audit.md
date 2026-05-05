# Complete Project Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a complete, evidence-based, file-by-file audit of the entire repository against end-to-end MLOps requirements, with code-level fixes and a submission readiness verdict.

**Architecture:** The audit runs as a deterministic workflow: inventory files, audit each file with a strict rubric, validate cross-cutting MLOps requirements, then synthesize critical issues, improvements, and scoring. Findings are stored in structured markdown so each claim is traceable to concrete file evidence.

**Tech Stack:** Python, scikit-learn, DVC, MLflow, FastAPI/Flask, GitHub Actions, Evidently, Prometheus, pytest, ruff

---

## File Structure Map (Locked Before Execution)

- `docs/superpowers/plans/2026-05-05-complete-project-audit.md` - this execution plan.
- `docs/audits/2026-05-05-file-by-file-audit.md` - primary audit artifact with one section per file.
- `docs/audits/2026-05-05-project-checklist.md` - project-level checklist validation (DVC, API, monitoring, CI/CD, docs, reproducibility).
- `docs/audits/2026-05-05-critical-issues.md` - top 10 critical issues with exact fixes.
- `docs/audits/2026-05-05-improvements.md` - top 10 improvements for full marks.
- `docs/audits/2026-05-05-scorecard.md` - numeric scoring breakdown and final verdict.
- `scripts/audit/build_inventory.py` - reproducible file inventory generator.
- `scripts/audit/generate_audit_skeleton.py` - generates file sections to ensure no file is skipped.
- `tests/scripts/audit/test_inventory.py` - tests for inventory completeness and deterministic ordering.
- `tests/scripts/audit/test_skeleton.py` - tests that skeleton contains all files exactly once.

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
Expected: file exists and includes every tracked/untracked project file except excluded cache/build dirs.

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
        "- TODO\n\n"
        "### Line-by-line findings\n"
        "- TODO\n\n"
        "### Exact code fixes\n"
        "- TODO\n"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/scripts/audit/test_skeleton.py -v`  
Expected: PASS.

- [ ] **Step 5: Generate full skeleton**

Run: `python -m scripts.audit.generate_audit_skeleton --inventory docs/audits/2026-05-05-inventory.txt --out docs/audits/2026-05-05-file-by-file-audit.md`  
Expected: one section per file in inventory, no missing/duplicate paths.

- [ ] **Step 6: Commit**

```bash
git add scripts/audit/generate_audit_skeleton.py tests/scripts/audit/test_skeleton.py docs/audits/2026-05-05-file-by-file-audit.md
git commit -m "chore(audit): scaffold file-by-file audit document"
```

### Task 3: Perform Complete File-by-File Audit (All Folders, All Files)

**Files:**
- Modify: `docs/audits/2026-05-05-file-by-file-audit.md`
- Reference during review: `.github/**`, `configs/**`, `data/**`, `docker/**`, `docs/**`, `flows/**`, `monitoring/**`, `scripts/**`, `src/**`, `tests/**`, root config files

- [ ] **Step 1: Audit infrastructure/config files first**

```text
Audit order:
1) Root configs (.gitignore, requirements.txt, pyproject.toml, dvc.yaml, dvc.lock, docker-compose.yml, Dockerfile, pytest.ini)
2) CI/CD workflows in .github/workflows
3) Data and config assets in configs/ and data/*.dvc
```

- [ ] **Step 2: For each file, fill required subsections with evidence**

```markdown
## <path>
### Purpose
- One concise statement of file responsibility.
### Line-by-line findings
- Lx-Ly: <issue> | Why wrong: <reason> | Risk: <impact>
### Exact code fixes
```python
# minimal concrete patch
```
```

- [ ] **Step 3: Enforce “no skip” constraint with checklist marks**

Run: `python -m scripts.audit.generate_audit_skeleton --verify-complete docs/audits/2026-05-05-file-by-file-audit.md`  
Expected: PASS only when every generated file section has non-placeholder findings.

- [ ] **Step 4: Commit**

```bash
git add docs/audits/2026-05-05-file-by-file-audit.md
git commit -m "docs(audit): complete file-by-file technical audit with code-level fixes"
```

### Task 4: Validate End-to-End MLOps Checklist Requirements

**Files:**
- Create: `docs/audits/2026-05-05-project-checklist.md`
- Modify (if corrections discovered): `docs/audits/2026-05-05-file-by-file-audit.md`

- [ ] **Step 1: Write checklist template covering all required categories**

```markdown
## DVC reproducibility
- dvc.yaml stages valid: PASS/FAIL
- dvc.lock in sync: PASS/FAIL
## Preprocessing
- sklearn Pipeline fitted only on train split: PASS/FAIL
- Leakage checks: PASS/FAIL
## Experiments
- MLflow run logging completeness: PASS/FAIL
- Registry promotion flow: PASS/FAIL
## Serving
- /health and /predict validation: PASS/FAIL
## CI/CD
- lint/tests/data/model gates: PASS/FAIL
## Monitoring
- Drift threshold logic (>20%): PASS/FAIL
## Docs
- README quickstart <= 3 steps, model/data cards: PASS/FAIL
## Reproducibility
- deterministic seeds, pinned deps, env parity: PASS/FAIL
```

- [ ] **Step 2: Run verification commands and capture results**

Run:
- `dvc dag`
- `dvc repro --dry`
- `pytest -q`
- `ruff check .`
- `python -m src.api.app --help` (or project-equivalent API entrypoint)

Expected: command outputs recorded with PASS/FAIL evidence and failure traces where applicable.

- [ ] **Step 3: Commit**

```bash
git add docs/audits/2026-05-05-project-checklist.md docs/audits/2026-05-05-file-by-file-audit.md
git commit -m "docs(audit): add project-level mlops compliance checklist"
```

### Task 5: Produce Critical Issues and Improvement Backlogs

**Files:**
- Create: `docs/audits/2026-05-05-critical-issues.md`
- Create: `docs/audits/2026-05-05-improvements.md`

- [ ] **Step 1: Rank top 10 critical issues**

```markdown
## Critical 1: <title>
- Severity: Critical
- Evidence: <exact file + line range>
- Why it fails requirements: <grading impact>
- Exact fix:
```python
# concrete corrected code
```
- Validation command: `<exact command>`
```

- [ ] **Step 2: Rank top 10 non-critical improvements**

```markdown
## Improvement 1: <title>
- Current state
- Recommended enhancement
- Exact implementation patch
- Expected score impact
```

- [ ] **Step 3: Commit**

```bash
git add docs/audits/2026-05-05-critical-issues.md docs/audits/2026-05-05-improvements.md
git commit -m "docs(audit): add prioritized critical issues and improvement roadmap"
```

### Task 6: Build Final Scorecard and Verdict

**Files:**
- Create: `docs/audits/2026-05-05-scorecard.md`
- Modify: `docs/audits/2026-05-05-project-checklist.md` (if score evidence links need updates)

- [ ] **Step 1: Apply required scoring rubric**

```markdown
- DVC: /10
- Preprocessing: /12
- Experiments: /15
- Serving: /15
- CI/CD: /13
- Monitoring: /15
- Docs: /10
- Reproducibility: /10
Total: /100
```

- [ ] **Step 2: Add strict final verdict**

```markdown
## Final verdict
- Ready for submission / Not ready
- Blocking reasons (must reference critical list IDs)
```

- [ ] **Step 3: Commit**

```bash
git add docs/audits/2026-05-05-scorecard.md docs/audits/2026-05-05-project-checklist.md
git commit -m "docs(audit): add graded scorecard and submission readiness verdict"
```

### Task 7: Quality Gate Before Declaring Audit Complete

**Files:**
- Modify if needed: all audit docs in `docs/audits/`

- [ ] **Step 1: Verify no placeholders remain**

Run: `rg "TODO|TBD|implement later|add appropriate|write tests for the above" docs/audits/`  
Expected: no matches.

- [ ] **Step 2: Verify file coverage completeness**

Run: `python -m scripts.audit.generate_audit_skeleton --verify-complete docs/audits/2026-05-05-file-by-file-audit.md --inventory docs/audits/2026-05-05-inventory.txt`  
Expected: PASS with “all files covered”.

- [ ] **Step 3: Final commit**

```bash
git add docs/audits/ scripts/audit/ tests/scripts/audit/
git commit -m "docs(audit): finalize complete repository audit package"
```

## Self-Review

### 1) Spec coverage check
- Covers full-repo, file-by-file review with explicit non-skip safeguards (Tasks 1-3, 7).
- Covers all requested MLOps requirement categories and project-level checklist (Task 4).
- Covers output format: file-by-file audit, critical list, improvement list, score breakdown, verdict (Tasks 3, 5, 6).
- Gap check: none; all required deliverables mapped to dedicated tasks.

### 2) Placeholder scan check
- Plan contains no execution placeholders like “TBD” for core actions.
- Any `TODO` shown appears only as generated template content in code snippets, not as unresolved plan action items.

### 3) Type/signature consistency
- `collect_files(root: Path) -> list[str]` usage remains consistent.
- `render_section(path: str) -> str` usage remains consistent.
- `--verify-complete` command referenced consistently across Tasks 3 and 7.

Plan complete and saved to `docs/superpowers/plans/2026-05-05-complete-project-audit.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
