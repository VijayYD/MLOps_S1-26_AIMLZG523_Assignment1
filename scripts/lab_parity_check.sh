#!/usr/bin/env bash
# Proves requirements.txt/requirements-dev.txt's exact pins actually
# install and run correctly on Python 3.12.9 (the lab's version) -- not
# just "the version range looks compatible on paper" (two real pin bugs
# were only ever caught by actually running things: docs/DECISIONS.md
# D1.1 and D4.1).
#
# Archives the committed tree (git archive -- tracked files only, so
# .venv/data/raw/mlruns/etc. never leak in), mounts it READ-ONLY into a
# fresh python:3.12.9-slim container, and inside the container copies it
# again to a writable location before installing anything -- the
# read-only mount itself is never written to.
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE="python:3.12.9-slim"
TMP_DIR="$(mktemp -d)"
cleanup() { rm -rf "${TMP_DIR}"; }
trap cleanup EXIT

echo "Archiving the committed tree (tracked files only) to ${TMP_DIR}/repo..."
mkdir -p "${TMP_DIR}/repo"
git archive HEAD | tar -x -C "${TMP_DIR}/repo"

echo "Running inside a fresh ${IMAGE} container..."
docker run --rm \
  --platform linux/amd64 \
  -v "${TMP_DIR}/repo:/repo:ro" \
  "${IMAGE}" \
  bash -c '
    set -euo pipefail
    cp -r /repo /work
    cd /work
    python -m venv .venv
    .venv/bin/pip install --upgrade pip --quiet
    .venv/bin/pip install -r requirements-dev.txt --quiet
    .venv/bin/pip install -e . --quiet

    echo "=== ruff ==="
    .venv/bin/ruff check .
    .venv/bin/ruff format --check .

    echo "=== pytest ==="
    .venv/bin/pytest

    echo "=== short training run (proves the real pipeline fits on Python 3.12.9) ==="
    .venv/bin/python scripts/_lab_parity_train_check.py
  '

echo "Lab-parity check passed: requirements install and run correctly on ${IMAGE}."
