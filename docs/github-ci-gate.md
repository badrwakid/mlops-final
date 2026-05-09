# GitHub + CI as the shared gate

Large datasets and trained artifacts (**`data/raw/hour.csv`**, most parquet splits, **`preprocessor.pkl`**, **`model.pkl`**) are **not stored in Git**. Small files **tracked in Git** on purpose:

- **`data/splits/metrics.json`** — `cache: false` in `dvc.yaml` so gates can read thresholds from Git.
- **`data/splits/reference.parquet`** — small drift baseline so **`docker compose build api`** works on a clone without **`dvc pull`** (still must match **`dvc.lock`**).

### DVC remote vs free bootstrap

Default **human** remote in **`.dvc/config`** is DagsHub (basic auth).

- **Prefer shared storage:** after **`dvc repro`**, run **`dvc push`** successfully once. Then any clone with **`DAGSHUB_TOKEN`** can **`dvc pull`**.
- **GitHub Actions optional secrets:** **`DVC_REMOTE_URL`** must match your DVC HTTP remote (e.g. `https://dagshub.com/badrwakid/mlops-final.dvc`), plus **`DAGSHUB_TOKEN`** for `dvc pull` Basic auth (`user badrwakid` is wired in workflows). Actions rewrite `localremote` for the runner.

### Free CI with no remote secrets (no credit card)

If **`DVC_REMOTE_URL`** / **`DAGSHUB_TOKEN`** are **not** set, workflows match a local **`python scripts/bootstrap_dvc_workspace.py`** style flow:

- **Raw data:** Actions runs **`python scripts/fetch_uci_hour_csv.py`**, which downloads the official UCI *Bike Sharing* zip and extracts **`hour.csv`** (MD5-checked against **`hour.csv.dvc`**).
- **Pipeline artifacts:** **`dvc repro`** on the runner (with **`MLFLOW_TRACKING_URI=file:./mlruns`**) restores **`production.parquet`**, **`model.pkl`**, etc. from the lockfile.

Optional secrets remain useful so CI can **`dvc pull`** instead of retraining on every run.

If **`DVC_REMOTE_URL`** uses **`s3://...`**, add **`AWS_*`** secrets instead of **`DAGSHUB_TOKEN`** (workflows rely on boto3-backed DVC remote for S3).

Team workflow for this repo: **branch → run `scripts/run_full_ci_local.ps1` locally → open Pull Request → merge** (no required teammate approval on GitHub; use Issues if your instructor still wants planning evidence).

## Repository

Public repo: **https://github.com/badrwakid/mlops-final**  
Default branch: **`main`**.

### Commit authorship (no Cursor co-author lines)

Commits should reflect **only** your Git **`user.name`** / **`user.email`** (e.g. **badrwakid**). The tracked hook **`.githooks/prepare-commit-msg`** removes **`Co-authored-by: Cursor`** if an editor injects it.

After cloning, enable hooks once from the repo root:

```bash
git config core.hooksPath .githooks
```

In Cursor, disable any option that **auto-adds co-authors** to commits if that line still appears.

## GitHub Actions

For public repositories, Actions are typically **enabled by default**. To confirm:

**Settings → Actions → General → Actions permissions** — allow Actions.

Workflows live under `.github/workflows/` (notably **`CI`** and **`Monitoring Drift Report`**). The **`CI`** workflow runs **`compose-validate`** (builds **`mlflow`**, **`api`**, **`dashboard`** images), **`model-validation`** (Docker Compose **`mlflow`** + **`scripts/seed_mlflow_production.py`** + **`scripts/validate_model.py`**), and **`monitoring-validation`** (operational Evidently drift only; **`generate_synthetic_demo_report`** must stay **`false`**). The scheduled **`Monitoring Drift Report`** workflow repeats operational drift with the same artifact bootstrap rules.

## Repository secrets (Actions)

**Settings → Secrets and variables → Actions → New repository secret**

| Secret | When you need it |
|--------|------------------|
| **`DVC_REMOTE_URL`** | Optional when using **free CI bootstrap** (`fetch_uci_hour_csv.py` + `dvc repro`). Set to your DVC storage URL (**DagsHub** `https://dagshub.com/.../*.dvc` or **`s3://...`**). Must match whatever **`dvc push`** publishes to. |
| **`DAGSHUB_TOKEN`** | With **`DVC_REMOTE_URL`** pointing at **DagsHub**. Workflows configure Basic auth **`user`** + this token as **`password`** (same as **`dvc remote modify --local localremote password`** locally). Never commit. |
| **`AWS_ACCESS_KEY_ID`** | Only if **`DVC_REMOTE_URL`** is **S3**. Pair with **`AWS_SECRET_ACCESS_KEY`** and **`AWS_DEFAULT_REGION`**. |
| **`AWS_SECRET_ACCESS_KEY`** | Pair with **`AWS_ACCESS_KEY_ID`**. |
| **`AWS_DEFAULT_REGION`** | e.g. **`eu-west-1`** — must match the bucket region. |

CLI examples:

```bash
# DagsHub + pull in CI (optional; omit both to use free bootstrap)
gh secret set DVC_REMOTE_URL -b"https://dagshub.com/badrwakid/mlops-final.dvc"
gh secret set DAGSHUB_TOKEN -b"<token>"

# S3 alternative
gh secret set DVC_REMOTE_URL -b"s3://YOUR_BUCKET/YOUR_PREFIX"
gh secret set AWS_ACCESS_KEY_ID -b"AKIA..."
gh secret set AWS_SECRET_ACCESS_KEY -b"..."
gh secret set AWS_DEFAULT_REGION -b"eu-west-1"
```

After secrets exist locally, **`dvc push`** publishes to the **same logical URL**. Repeat **`dvc push`** whenever **`dvc repro`** produces new tracked hashes so **`dvc pull`** works on other machines.

## Branch protection on `main`

**`main`** may still have basic protection (e.g. no force-push); it does **not** require a teammate’s approving review before merge.

Optional: after **`CI`** is green on PRs, add **required status checks** under **Settings → Rules / Branch protection** for **`CI`** jobs (**lint**, **test**, **data-validation**, **model-validation**, **compose-validate**, **monitoring-validation**) so merges cannot bypass the pipeline.

## Collaboration checklist

- **Branches**: `feature/<short-description>` → **Pull Request** → **`main`** → **Merge** when checks pass.
- **Local gate**: run **`powershell -ExecutionPolicy Bypass -File scripts/run_full_ci_local.ps1`** before pushing when possible.
- **Issues** (if your course asks for evidence): track tasks and bugs in **Issues**.
- **Commits**: each member should still contribute meaningful commits if grading expects it.

## Verify from your laptop

```bash
git fetch origin
git checkout main
git pull origin main
gh secret list -R badrwakid/mlops-final
gh api repos/badrwakid/mlops-final/branches/main/protection --jq ".required_pull_request_reviews"
```

If reviews are off, the last line prints **`null`**.
