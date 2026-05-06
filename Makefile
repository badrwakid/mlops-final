PYTHON ?= python

.PHONY: repro train baseline monitor verify

repro:
	$(PYTHON) -m dvc repro

train:
	$(PYTHON) -m src.training.train

baseline:
	$(PYTHON) scripts/generate_baseline_report.py

monitor:
	$(PYTHON) -m monitoring.run_monitoring

verify:
	$(PYTHON) scripts/validate_model.py
