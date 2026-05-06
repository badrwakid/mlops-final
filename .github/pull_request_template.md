## Rubric impact

- [ ] Reproducibility
- [ ] Documentation
- [ ] Testing
- [ ] Model/serving behavior
- [ ] Monitoring/operations
- [ ] No rubric impact

## Validation checklist

- [ ] Local tests pass for touched scope
- [ ] Docs and examples updated if behavior changed
- [ ] No secrets or large artifacts added
- [ ] Linked issue is included
## Title (Conventional Commits)

Use the same style as commit messages and PRs on `main`:

`type(scope): short description`

- **Types:** `feat` | `fix` | `chore` | `test` | `docs`
- **Examples:** `feat(serving): add batch predict endpoint` · `chore(ci): cache pip in workflow` · `fix(dvc): correct train stage deps` · `docs(evidence): update technical report screenshots`

## Branch

- [ ] This PR is from a **short-lived** branch named `task/<id>-<short-name>` (not from `main` and not long-lived).
- [ ] Suggested slugs (examples): `component-2-3-train`, `component-4-5-serve-ci`, `component-6-monitoring`, `bonus-docker-prefect`, `component-7-docs`.

## Summary

<!-- What changed and why (1–3 sentences). -->

## How to test

<!-- e.g. commands, Docker, or “N/A (docs-only)”. -->

## Docs / evidence (if applicable)

<!-- e.g. `docs/technical_report.md`, `docs/screenshots/`, or “N/A”. -->

## Related issues

<!-- Link issues so they close automatically when this merges: -->

Closes #

<!-- Other refs: Related # -->

## Pre-merge checklist (team workflow)

- [ ] **At least 1** teammate has **approved** this PR (required).
- [ ] Merge strategy will be **squash merge** via GitHub (no direct pushes to `main`).
- [ ] **Branch is up to date** with `main` if branch protection requires it.
- [ ] **CI:** All required checks are **green** on the latest commit (once branch protection is configured):
  - `lint`
  - `test`
  - `data-validation`
  - `model-validation`
- [ ] **Repo hygiene:** No raw data under `data/` (use DVC), no `.env`, local venvs (e.g. `.venv/`, `.venv_strict/`), or experiment dirs such as `mlruns/` / `mlruns_local_strict/` committed unintentionally.
- [ ] **Config:** Deliberate updates to `configs/params.yaml`, `dvc.yaml` / `dvc.lock` only when intended.
