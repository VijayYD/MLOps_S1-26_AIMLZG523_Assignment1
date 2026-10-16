#!/usr/bin/env bash
# Small traffic generator so the Grafana dashboard shows real data.
# Usage: scripts/load_test.sh [base_url]
set -euo pipefail

cd "$(dirname "$0")/.."

BASE_URL="${1:-http://127.0.0.1:8000}"
REQUESTS="${REQUESTS:-200}"
CONCURRENCY="${CONCURRENCY:-5}"

if ! curl -fsS -o /dev/null "${BASE_URL}/health"; then
  echo "::error::Cannot reach ${BASE_URL}/health -- nothing is listening there." >&2
  echo "If this is a Kubernetes deployment, port-forward the API first, e.g.:" >&2
  echo "  kubectl -n heart-mlops port-forward svc/heart-api-lb 8000:8000 &" >&2
  echo "Then re-run this script (or pass the right URL as an argument)." >&2
  exit 1
fi

echo "Sending ${REQUESTS} requests to ${BASE_URL}/predict (concurrency ${CONCURRENCY})..."

send_one() {
  local base_url="$1"
  local f
  for f in test/sample_requests/valid_high_risk.json test/sample_requests/valid_low_risk.json; do
    curl -sS -o /dev/null -w "%{http_code}\n" -X POST "${base_url}/predict" \
      -H "Content-Type: application/json" -d @"$f"
  done
}
export -f send_one

seq 1 "${REQUESTS}" | xargs -P "${CONCURRENCY}" -I{} bash -c 'send_one "$1"' _ "${BASE_URL}" | sort | uniq -c

echo "Done."
