#!/usr/bin/env bash
# Serves the MLflow UI against this project's local tracking store.
# Port 5001, not the MLflow default 5000, because 5000 is used by AirPlay
# Receiver on macOS (see README.md).
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -x .venv/bin/python ]; then
  echo "ERROR: .venv not found. Run scripts/setup_venv.sh first." >&2
  exit 1
fi

.venv/bin/mlflow ui --backend-store-uri "sqlite:///mlflow.db" --port 5001
