# GitHub + CI as the shared gate

Large datasets and trained artifacts (**`data/raw/hour.csv`**, splits, **`model.pkl`**) are **not stored in Git** (except small JSON metrics / optional pinned pickles if your branch commits them). For **shared team storage**, use a **DVC remote** (e.g. S3): GitHub Actions runs **`dvc pull`** after rewriting `localremote` to **`DVC_REMOTE_URL`** — set that URI plus AWS credentials as repository **Secrets** (see table below). Locally **`dvc push`** after **`dvc repro`** publishes artifacts to that remote.

### Free CI with no cloud account (no credit card)

If you **do not** set **`DVC_REMOTE_URL`** (and no S3 keys), workflows still pass **without paid storage**:

- **Raw data:** Actions runs **`python scripts/fetch_uci_hour_csv.py`**, which downloads the official UCI *Bike Sharing* zip and extracts **`hour.csv`** (MD5-checked against **`hour.csv.dvc`**).
- **Splits / models needed only on some jobs:** When parquet splits are missing, Actions runs **`dvc repro`** on the runner after the fetch above. That rebuilds **`reference.parquet`**, **`production.parquet`**, **`model.pkl`**, etc. deterministically from the lockfile—no AWS/GCP/Azure signup required.

Optional secrets remain useful so CI can **`dvc pull`** instead of retraining on every run (faster, matches a shared remote).

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
| **`DVC_REMOTE_URL`** | Optional if you rely on **free CI bootstrap** (UCI fetch + `dvc repro`). Set when you want **`dvc pull`** from shared storage instead (e.g. **`s3://your-bucket/path/to/dvc-store`**). Same logical storage your team uses after `dvc push`. |
| **`AWS_ACCESS_KEY_ID`** | With **`AWS_SECRET_ACCESS_KEY`** and **`AWS_DEFAULT_REGION`**, only if the remote is **S3** and you set **`DVC_REMOTE_URL`**. Use an IAM user limited to read (CI) and write (developers pushing). Never commit keys. |
| **`AWS_SECRET_ACCESS_KEY`** | Pair with the above. |
| **`AWS_DEFAULT_REGION`** | e.g. **`eu-west-1`** — must match the bucket region. |

CLI (non-interactive file input):

```bash
gh secret set DVC_REMOTE_URL -b"s3://YOUR_BUCKET/YOUR_PREFIX"
gh secret set AWS_ACCESS_KEY_ID -b"AKIA..."
gh secret set AWS_SECRET_ACCESS_KEY -b"..."
gh secret set AWS_DEFAULT_REGION -b"eu-west-1"
```

After secrets exist, align your **local** remote once (same URL as `DVC_REMOTE_URL`):

```bash
pip install "dvc[s3]==3.51.2"
dvc remote modify localremote url s3://YOUR_BUCKET/YOUR_PREFIX
dvc push
```

Repeat **`dvc push`** whenever **`dvc repro`** (or targeted stages) produces updated tracked outputs so CI and teammates can **`dvc pull`** them.

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
