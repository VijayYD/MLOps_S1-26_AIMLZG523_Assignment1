#!/usr/bin/env bash
# Runs the full training session: tunes all three models, selects one by
# the pre-registered rule (docs/DECISIONS.md D3.1), logs everything to
# MLflow (nested runs, metrics, artifacts, model), and registers the
# selected model with the "champion" alias.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -x .venv/bin/python ]; then
  echo "ERROR: .venv not found. Run scripts/setup_venv.sh first." >&2
  exit 1
fi

if [ ! -f data/processed/heart_combined_clean.csv ]; then
  echo "data/processed/heart_combined_clean.csv not found -- running download + clean first." >&2
  .venv/bin/python -m heart_mlops.data.download
  .venv/bin/python -m heart_mlops.data.clean
fi

.venv/bin/python -m heart_mlops.models.train
