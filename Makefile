PYTHON ?= python

.PHONY: repro train baseline monitor drift-report verify dvc-push install-hooks

repro:
	$(PYTHON) -m dvc repro

train:
	$(PYTHON) -m src.training.train

baseline:
	$(PYTHON) scripts/generate_baseline_report.py

monitor:
	$(PYTHON) -m monitoring.run_monitoring

drift-report:
	$(PYTHON) -c "from src.drift_report import run_and_log; import pandas as pd; df = pd.read_csv('artifacts/prediction_log.csv').tail(500); print(run_and_log('artifacts/reference.parquet', df))"

verify:
	$(PYTHON) scripts/validate_model.py

dvc-push:
	$(PYTHON) -m dvc commit -f
	$(PYTHON) -m dvc push

install-hooks:
	cp scripts/pre-push .git/hooks/pre-push
	chmod +x .git/hooks/pre-push
	@echo "Git pre-push hook installed — dvc push runs automatically before every git push."
