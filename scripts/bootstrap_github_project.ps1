#Requires -Version 5.1
<#
.SYNOPSIS
  Creates milestones and tracking issues for the DDSC611 MLOps final project on GitHub.

.DESCRIPTION
  Requires GitHub CLI v2 (`gh`) and a logged-in account:
    gh auth login

  Run from the repository root (or any path under the repo):
    pwsh -File scripts/bootstrap_github_project.ps1

  The script is idempotent for milestones (skips if a milestone title already exists).
  Re-running may create duplicate issues; use the GitHub UI to close extras if needed.

.NOTES
  This does not create GitHub "Projects" (v2) columns; add issues to a project board in the UI if required.
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-RepoSlug {
  $url = (git -C (Get-Location) remote get-url origin 2>$null)
  if (-not $url) { throw "Not a git repo with origin, or 'git' not in PATH." }
  if ($url -match "github\.com[:/](?<owner>[^/]+)/(?<repo>[^/.]+)") {
    return @{ Owner = $Matches["owner"]; Repo = $Matches["repo"].TrimEnd(".git") }
  }
  throw "Could not parse owner/repo from remote: $url"
}

function Test-GhAuthed {
  $prev = $ErrorActionPreference
  $ErrorActionPreference = "SilentlyContinue"
  try {
    & gh auth status 1>$null 2>$null
    return ($LASTEXITCODE -eq 0)
  } finally {
    $ErrorActionPreference = $prev
  }
}

function Get-Milestones {
  param([string]$Owner, [string]$Repo)
  $raw = & gh api "repos/$Owner/$Repo/milestones?state=all&per_page=100" 2>$null
  if ($LASTEXITCODE -ne 0) { throw "gh api milestones failed for repos/$Owner/$Repo/milestones" }
  return ($raw | ConvertFrom-Json)
}

function New-Milestone {
  param(
    [string]$Owner,
    [string]$Repo,
    [string]$Title,
    [string]$Description
  )
  $existing = (Get-Milestones -Owner $Owner -Repo $Repo) | Where-Object { $_.title -eq $Title }
  if ($existing) {
    Write-Host "Milestone exists: $Title (number $($existing.number))" -ForegroundColor DarkGray
    return $existing
  }
  $payload = @{ title = $Title; description = $Description } | ConvertTo-Json -Compress
  $json = $payload | & gh api -X POST "repos/$Owner/$Repo/milestones" --input - 2>&1
  if ($LASTEXITCODE -ne 0) { throw "Failed to create milestone: $Title - $json" }
  $m = $json | ConvertFrom-Json
  Write-Host "Created milestone: $Title (#$($m.number))" -ForegroundColor Green
  return $m
}

function New-Issue {
  param(
    [string]$Owner,
    [string]$Repo,
    [string]$Title,
    [string]$Body,
    [string]$MilestoneTitle
  )
  $args = @(
    "issue", "create",
    "--repo", "$Owner/$Repo",
    "--title", $Title,
    "--body", $Body
  )
  if ($MilestoneTitle) {
    $args += @("--milestone", $MilestoneTitle)
  }
  & gh @args
  if ($LASTEXITCODE -ne 0) { throw "Failed to create issue: $Title" }
  Write-Host "Created issue: $Title" -ForegroundColor Green
}

if (-not (Test-GhAuthed)) {
  Write-Host "GitHub CLI is not authenticated. Run: gh auth login" -ForegroundColor Red
  exit 1
}

$slug = Get-RepoSlug
$Owner = $slug.Owner
$Repo = $slug.Repo
Write-Host "Repository: $Owner/$Repo" -ForegroundColor Cyan

# --- Milestones (order matches delivery sequence) ---
$m3 = New-Milestone -Owner $Owner -Repo $Repo -Title "Phase 3: Preprocessing & MLflow" -Description @"
Component 2+3: DVC train stage, sklearn preprocessing pipeline artifacts, Optuna HPO, MLflow experiments and model registry workflow.
"@
$m45 = New-Milestone -Owner $Owner -Repo $Repo -Title 'Phase 4-5: Serving and CI' -Description @'
Component 4+5: FastAPI service (/health, /predict, /predict/batch), Prometheus metrics, integration tests, GitHub Actions (lint, test, data-validation, model-validation).
'@
$m6 = New-Milestone -Owner $Owner -Repo $Repo -Title "Phase 6: Monitoring" -Description @"
Evidently drift reports, Prometheus/Grafana or equivalent observability, scheduled or triggered checks.
"@
$mb = New-Milestone -Owner $Owner -Repo $Repo -Title 'Bonus: Docker and Prefect' -Description @'
Bonus A: Docker + Compose. Bonus B: Prefect orchestration for pipeline stages.
'@
$m7 = New-Milestone -Owner $Owner -Repo $Repo -Title "Phase 7: Documentation" -Description @"
Model card, data card, README updates for reproducibility and governance.
"@
$meta = New-Milestone -Owner $Owner -Repo $Repo -Title "Course: Submission & repo hygiene" -Description @"
Final checklist: branch protection, required CI checks, clean data/.gitignore, no secrets committed.
"@

# --- Issues ---
$milestonePhase3 = "Phase 3: Preprocessing & MLflow"
New-Issue -Owner $Owner -Repo $Repo -MilestoneTitle $milestonePhase3 -Title "PR: Merge training + MLflow (feature/component-2-3-train)" -Body @"
## Goal
Land preprocessing + training + MLflow work on `main` via squash merge.

## Scope
- `feature/component-2-3-train` → `main`
- Title suggestion: **feat: Component 2+3 — preprocessing pipeline and MLflow tracking**

## Acceptance
- [ ] Unit tests pass locally; DVC pipeline includes train stage; `configs/params.yaml` drives training.
- [ ] At least 1 review; CI expectations documented if workflow lands in a follow-up PR.

Closes when PR is merged.
"@

New-Issue -Owner $Owner -Repo $Repo -MilestoneTitle $milestonePhase3 -Title "Verify MLflow: at least 3 runs and Production promotion path" -Body @"
## Goal
Confirm experiment tracking and registry promotion match the course rubric.

## Checklist
- [ ] Three or more distinct runs visible in the MLflow UI (or export).
- [ ] Best model registered; Production stage set via API or documented UI steps.
- [ ] `data/splits/metrics.json` matches gate used in training (if committed).

Notes: adjust commands/paths to match your tracking URI and experiment name in `configs/params.yaml`.
"@

$milestonePhase45 = 'Phase 4-5: Serving and CI'
New-Issue -Owner $Owner -Repo $Repo -MilestoneTitle $milestonePhase45 -Title "PR: Merge FastAPI + CI (feature/component-4-5-serve-ci)" -Body @"
## Goal
Land serving and CI on `main`.

## Scope
- Prefer rebasing this branch onto latest `main` after the Phase 3 PR merges, then open PR to `main`.
- Title suggestion: **feat: Component 4+5 — FastAPI serving and GitHub Actions CI**

## Acceptance
- [ ] Integration tests green; `scripts/validate_model.py` and `validation.rmse_threshold` documented.
- [ ] Workflow job names exactly: `lint`, `test`, `data-validation`, `model-validation`.

Closes when PR is merged.
"@

New-Issue -Owner $Owner -Repo $Repo -MilestoneTitle $milestonePhase45 -Title "Configure branch protection for main" -Body @"
## Goal
Enforce review + CI before merge.

## Settings (GitHub → Settings → Branches)
- Branch pattern: `main`
- Require pull request (1 approval)
- Require status checks: **lint**, **test**, **data-validation**, **model-validation**
- Require branches up to date before merging
- Do not allow bypass

Run after the CI workflow exists on `main` and at least one PR has produced green checks (so the checks appear in the picker).

Refs: ``docs`` / course spec for DDSC611.
"@

New-Issue -Owner $Owner -Repo $Repo -MilestoneTitle $milestonePhase45 -Title "Optional: hosted MLflow for CI model-validation" -Body @"
## Context
CI sets `SKIP_MLFLOW_REGISTRY=1` so `model-validation` does not require a live registry.

## Optional hardening
- [ ] Point `MLFLOW_TRACKING_URI` at a reachable server in CI secrets.
- [ ] Remove or narrow `SKIP_MLFLOW_REGISTRY` so `scripts/validate_model.py` loads `Production` in CI.

Only if the rubric requires registry load in CI.
"@

$milestonePhase6 = "Phase 6: Monitoring"
New-Issue -Owner $Owner -Repo $Repo -MilestoneTitle $milestonePhase6 -Title "Implement monitoring & drift (feature/component-6-monitoring)" -Body @"
## Goal
Evidently reports (e.g. data drift), Prometheus metrics from serving or batch jobs, documented runbook.

## Branch
`feature/component-6-monitoring` (already created remotely as placeholder — rebase from `main` before work).

## Acceptance
- [ ] Drift or quality report generated from reference vs current data path in config.
- [ ] Metrics exposed or scraped consistent with course requirements.
"@

$milestoneBonus = 'Bonus: Docker and Prefect'
New-Issue -Owner $Owner -Repo $Repo -MilestoneTitle $milestoneBonus -Title 'Bonus A: Docker and Compose' -Body @"
## Goal
Containerize training and/or API; `docker compose` brings up stack per project spec.

## Branch
`feature/bonus-docker-prefect`
"@

New-Issue -Owner $Owner -Repo $Repo -MilestoneTitle $milestoneBonus -Title 'Bonus B: Prefect orchestration' -Body @"
## Goal
Prefect flows for prepare → train → validate (or equivalent DAG in `flows/`).

## Branch
`feature/bonus-docker-prefect`
"@

$milestoneDocs = "Phase 7: Documentation"
New-Issue -Owner $Owner -Repo $Repo -MilestoneTitle $milestoneDocs -Title "Docs: model card, data card, README (feature/component-7-docs)" -Body @"
## Deliverables
- `docs/model_card.md` — intended use, metrics, limitations, lineage.
- `docs/data_card.md` — sources, splits, known biases.
- Root `README` — how to run DVC, train, serve, and CI locally.

## Branch
`feature/component-7-docs`
"@

$milestoneMeta = "Course: Submission & repo hygiene"
New-Issue -Owner $Owner -Repo $Repo -MilestoneTitle $milestoneMeta -Title "Final submission checklist" -Body @"
## Before submitting
- [ ] No raw data under `data/` committed (DVC only); `.gitignore` respected.
- [ ] No `.env`, `mlruns/`, or venv committed.
- [ ] All required PRs merged; `main` passes CI.
- [ ] Issue numbers referenced in PR bodies (`Closes #…`) where applicable.

"@

Write-Host "`nDone. Open Issues and Milestones on GitHub to verify and assign teammates." -ForegroundColor Cyan
