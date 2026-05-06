# Team collaboration

## Branch strategy

- Use short-lived feature branches named `task/<id>-<short-name>`.
- Keep `main` always releasable; direct pushes to `main` are not allowed.
- Rebase or merge `main` into the feature branch before opening a pull request.

## PR review rule

- Every pull request requires at least one reviewer approval before merge.
- The author cannot self-approve or merge without review.
- Required checks must pass before merge.

## Issue workflow

1. Open an issue for each task with clear scope and acceptance criteria.
2. Assign the issue owner and link the working branch/PR.
3. Move issue state through: `Open -> In progress -> In review -> Done`.
4. Close the issue only after the PR is merged.

## Individual accountability

- Each contributor owns at least one issue at a time and reports status updates.
- Commits must be attributable to the contributor's git identity.
- Team contribution evidence is exported to `docs/team_contribution.csv`.

## Final discussion prep

- `badrwakid` explains DVC reproducibility flow, preprocessing pipeline, and centralized parameters.
- `Batran5` explains MLflow export evidence, model/data cards, and reproducibility checks.
- Both members can cover serving quickstart, CI/branch protection evidence, and collaboration process.
