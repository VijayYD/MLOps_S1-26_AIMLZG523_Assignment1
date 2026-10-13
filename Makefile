.PHONY: check-env venv lint test

check-env:
	bash scripts/check_env.sh

venv:
	bash scripts/setup_venv.sh

lint:
	.venv/bin/ruff check .
	.venv/bin/ruff format --check .

test:
	.venv/bin/pytest
