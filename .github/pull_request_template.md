## Title (Conventional Commits)

Use the same style as commit messages and PRs on `main`:

`type(scope): short description`

- **Types:** `feat` | `fix` | `chore` | `test` | `docs`
- **Examples:** `feat(serving): add batch predict endpoint` · `chore(ci): cache pip in workflow` · `fix(dvc): correct train stage deps`

## Branch

- [ ] This PR is from a **short-lived** branch named `feature/<component-slug>` (not from `main` and not long-lived).
- [ ] Suggested slugs (examples): `component-2-3-train`, `component-4-5-serve-ci`, `component-6-monitoring`, `bonus-docker-prefect`, `component-7-docs`.

## Summary

<!-- What changed and why (1–3 sentences). -->

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
- [ ] **Repo hygiene:** No raw data under `data/` (use DVC), no `.env`, venv, or `mlruns/` committed unintentionally.
- [ ] **Config:** Deliberate updates to `configs/params.yaml`, `dvc.yaml` / `dvc.lock` only when intended.
