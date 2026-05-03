# Technical report — MLOps final project (Bike sharing demand)

This note captures evidence requested for the pipeline and reproducibility sections of the course technical report.

## DVC pipeline

The Data Version Control (DVC) graph below matches `dvc.yaml`: raw data is tracked via `hour.csv.dvc`, then stages **prepare → preprocess → featurize → train**. Training (**train**) depends on both **preprocess** and **featurize** outputs.

![DVC pipeline DAG](screenshots/dvc_dag.png)

To regenerate the figure after dependency installs (`matplotlib`, `networkx`):

```bash
python scripts/render_dvc_dag_png.py
```

If Graphviz is installed, an equivalent diagram can be produced with:

```bash
dvc dag --dot | dot -Tpng -o docs/screenshots/dvc_dag.png
```

## Related documents

- `docs/model_card.md` — model summary (when present)
- `docs/data_card.md` — data summary (when present)
