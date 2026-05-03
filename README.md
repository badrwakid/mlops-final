# MLOps Final Project — Bike Sharing Demand

End-to-end MLOps pipeline for predicting hourly bike rental counts (UCI Bike Sharing dataset).

## Quickstart

Use a virtual environment (recommended). From the repository root:

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**MLflow tracking server (Docker)** — required for `configs/params.yaml` `tracking_uri: http://localhost:5000`:

```bash
docker compose up mlflow
```

UI: [http://localhost:5000](http://localhost:5000). The service uses SQLite at `./mlflow.db` and artifacts under `./mlartifacts/` (see `docker-compose.yml`). With the server running, training logs to the same experiment; use `MLFLOW_TRACKING_URI` to override if needed.

**Optional — Ridge baseline** (same experiment, different algorithm; no model registry promotion):

```bash
python -m src.training.train_baseline
```

Run after `dvc repro` has produced `data/splits/preprocessor.pkl` and splits.

**Then reproduce data artifacts and run the API**

```bash
dvc repro
dvc push
docker compose up --build
```

`dvc repro` runs the full DVC pipeline locally. **`dvc push` copies outputs to the configured local DVC remote** so your cache matches the project layout (run it after `dvc repro` whenever you want to persist artifacts under the remote). If someone else has already populated the remote directory, you can run **`dvc pull`** before `dvc repro` to avoid rebuilding large artifacts.

**DVC storage:** The default remote `localremote` targets **`dvc-storage/` at the repository root** (see `.dvc/config`; path is gitignored). On a fresh clone the folder may be missing—create it if needed (`mkdir dvc-storage` on Unix, `mkdir dvc-storage` in PowerShell), then run `dvc repro` and `dvc push`. To use another directory:  
`dvc remote modify localremote url <absolute-or-relative-path>`.

Then open `curl http://localhost:8000/health`.

## Documentation

- `docs/technical_report.md` — pipeline evidence (includes DVC DAG figure)
- `docs/model_card.md` and `docs/data_card.md` — model and data details (when present)
