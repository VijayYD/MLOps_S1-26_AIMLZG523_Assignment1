#!/usr/bin/env bash
# Builds the image, runs it, waits for health, sends every sample request
# in test/sample_requests/ and checks the response matches what the
# filename promises (valid_*.json -> 200, invalid_*.json -> 422), and
# saves the full console output to screenshots/logs/.
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE_NAME="${IMAGE_NAME:-heart-mlops-api}"
IMAGE_TAG="${IMAGE_TAG:-smoke}"
CONTAINER_NAME="heart-api-smoke"
PORT="${PORT:-8000}"

mkdir -p screenshots/logs
LOG_FILE="screenshots/logs/run_container_smoke_$(date +%Y%m%d_%H%M%S).log"

cleanup() {
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

{
  FAILED=0
  echo "=== Building image ==="
  IMAGE_NAME="${IMAGE_NAME}" IMAGE_TAG="${IMAGE_TAG}" bash scripts/build_image.sh

  echo "=== Starting container ==="
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  docker run -d --name "${CONTAINER_NAME}" -p "${PORT}:8000" "${IMAGE_NAME}:${IMAGE_TAG}"

  echo "=== Waiting for /health ==="
  HEALTHY=0
  for i in $(seq 1 30); do
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      echo "Healthy after ${i}s"
      HEALTHY=1
      break
    fi
    sleep 1
  done
  if [ "$HEALTHY" -ne 1 ]; then
    echo "ERROR: container did not become healthy in time" >&2
    docker logs "${CONTAINER_NAME}" || true
    exit 1
  fi

  echo "=== /ready ==="
  curl -fsS "http://127.0.0.1:${PORT}/ready"; echo

  echo "=== /model-info ==="
  curl -fsS "http://127.0.0.1:${PORT}/model-info"; echo

  echo "=== Sample requests ==="
  for f in test/sample_requests/*.json; do
    name="$(basename "$f")"
    expected=200
    case "$name" in
      invalid_*) expected=422 ;;
    esac

    echo "--- $name (expect HTTP ${expected}) ---"
    status="$(curl -sS -o /tmp/smoke_response.json -w "%{http_code}" -X POST "http://127.0.0.1:${PORT}/predict" \
      -H "Content-Type: application/json" -d @"$f")"
    cat /tmp/smoke_response.json; echo
    echo "HTTP ${status}"

    if [ "$status" != "$expected" ]; then
      echo "ERROR: $name expected HTTP $expected, got $status" >&2
      FAILED=1
    fi
  done

  echo "=== /metrics (first 20 lines) ==="
  curl -fsS "http://127.0.0.1:${PORT}/metrics" | head -20

  echo "=== Container logs ==="
  docker logs "${CONTAINER_NAME}"
  exit "${FAILED}"
} 2>&1 | tee "${LOG_FILE}" || true
# `|| true` above: with `set -e` + pipefail, a non-zero exit from the { }
# block would otherwise abort the script right here, before the
# PIPESTATUS[0] check below ever runs.

# The { ... } group runs in a subshell because it's on the left of a pipe,
# so FAILED set inside it never reaches this shell -- PIPESTATUS[0] (the
# group's own exit code, which we set explicitly above) is what survives.
SMOKE_EXIT="${PIPESTATUS[0]}"
echo "Saved log to ${LOG_FILE}"

if [ "$SMOKE_EXIT" -ne 0 ]; then
  echo "Smoke test FAILED -- see ${LOG_FILE}" >&2
  exit 1
fi
echo "Smoke test passed."
