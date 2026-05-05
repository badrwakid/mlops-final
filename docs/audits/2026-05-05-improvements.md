# Non-blocking improvements (after critical list is clear)

## I1 — Nested `mlops-final/` tree

- **Current state:** Full duplicate project under `mlops-final/` inflates audits and confuses “source of truth”.
- **Recommendation:** Remove from submission copy or add a one-line README note that grading targets the repo root only.
- **Expected impact:** Grader clarity only (not a syllabus point if root project is perfect).

## I2 — Align screenshot filenames with syllabus wording

- **Current state:** `README.md` lists several canonical PNG names; plan text also mentioned `ci-green-main` variants.
- **Recommendation:** Keep **one** table (in `README.md` + checklist) and delete alternate names in prose.
- **Expected impact:** −0 to rubric; reduces confusion.

## I3 — Prefect work-pool bootstrap script

- **Current state:** Operators must run `prefect work-pool create default-process-pool --type process` manually.
- **Recommendation:** Optional `scripts/init_prefect.ps1` for demo machines.
- **Expected impact:** Demo polish for Bonus B (+0 if already documented).
