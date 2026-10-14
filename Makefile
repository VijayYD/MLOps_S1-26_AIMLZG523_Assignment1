.PHONY: check-env venv lint test data train mlflow-ui

check-env:
	bash scripts/check_env.sh

venv:
	bash scripts/setup_venv.sh

lint:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

test:
	.venv/bin/pytest

data:
	.venv/bin/python -m heart_mlops.data.download
	.venv/bin/python -m heart_mlops.data.clean

train:
	bash scripts/train.sh

mlflow-ui:
	bash scripts/mlflow_ui.sh
