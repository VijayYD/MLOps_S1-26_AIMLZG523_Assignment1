#!/usr/bin/env bash
# Small traffic generator so the Grafana dashboard shows real data.
# Usage: scripts/load_test.sh [base_url]
set -euo pipefail

cd "$(dirname "$0")/.."

BASE_URL="${1:-http://127.0.0.1:8000}"
REQUESTS="${REQUESTS:-200}"
CONCURRENCY="${CONCURRENCY:-5}"

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
