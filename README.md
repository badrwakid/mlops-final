# MLOps Final Project — Bike Sharing Demand

End-to-end MLOps pipeline for predicting hourly bike rental counts (UCI Bike Sharing dataset).

## Quickstart (3 commands)

```bash
pip install -r requirements.txt
dvc pull && dvc repro
docker compose up --build
```

Then `curl http://localhost:8000/health`.

See `docs/model_card.md` and `docs/data_card.md` for details.