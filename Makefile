.PHONY: all check-env venv lint test data train mlflow-ui \
        build-image docker-smoke compose-up compose-down \
        k8s-deploy k8s-verify k8s-teardown load-test lab-parity report

# End-to-end on a machine that already has .venv (scripts/setup_venv.sh) --
# lint, data, test, train, in that order, matching the CI job sequence.
all: lint data test train

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

build-image:
	bash scripts/build_image.sh

docker-smoke:
	bash scripts/run_container_smoke.sh

compose-up:
	docker compose -f docker/docker-compose.yml up -d --build

compose-down:
	docker compose -f docker/docker-compose.yml down

k8s-deploy:
	bash scripts/k8s_deploy.sh

k8s-verify:
	bash scripts/k8s_verify.sh

k8s-teardown:
	bash scripts/k8s_teardown.sh

load-test:
	bash scripts/load_test.sh

lab-parity:
	bash scripts/lab_parity_check.sh

report:
	.venv/bin/python reports/build_report.py
