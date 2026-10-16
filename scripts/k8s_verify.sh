#!/usr/bin/env bash
# Prints cluster state, then calls /health and /predict through BOTH the
# Ingress and the LoadBalancer Service.
#
# Ingress: macOS + Docker driver: minikube ip is not routable from the
# host at all, so this uses `kubectl port-forward` to an arbitrary
# unprivileged local port instead of `minikube tunnel` (which also needs
# sudo for the Ingress's privileged ports 80/443 -- an interactive
# password prompt an unattended script can't supply, and the lab
# explicitly forbids sudo anyway). Linux: minikube's `ingress` addon is
# directly reachable at `minikube ip` there, no tunnel/port-forward
# needed. See docs/DECISIONS.md D7.7.
#
# LoadBalancer (heart-api-lb): a plain `type: LoadBalancer` Service is
# NOT reachable at `minikube ip:8000` on any OS unless `minikube tunnel`
# is actually running to assign it a real external IP (status.loadBalancer
# .ingress[0].ip) -- that's a property of how LoadBalancer Services work,
# not something specific to macOS. So on every OS, this checks for an
# assigned external IP first and only falls back to `kubectl
# port-forward` if one isn't there (tunnel not running) -- never assumes
# the node IP works directly. See docs/DECISIONS.md D7.11.
#
# Reaches the Ingress with a Host header rather than editing /etc/hosts.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=_k8s_lib.sh
source scripts/_k8s_lib.sh

mkdir -p screenshots/logs
LOG_FILE="screenshots/logs/k8s_verify_$(date +%Y%m%d_%H%M%S).log"

INGRESS_PF_PID=""
LB_PF_PID=""
cleanup() {
  [ -n "${INGRESS_PF_PID}" ] && kill "${INGRESS_PF_PID}" >/dev/null 2>&1 || true
  [ -n "${LB_PF_PID}" ] && kill "${LB_PF_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

{
  echo "=== kubectl get pods,svc,ingress,hpa -n heart-mlops ==="
  ${KUBECTL} -n heart-mlops get pods,svc,ingress,hpa -o wide

  OS="$(uname -s)"
  if [ "$OS" = "Darwin" ]; then
    echo
    echo "macOS detected: port-forwarding the Ingress controller to a local"
    echo "port instead of 'minikube tunnel' (which would need an"
    echo "interactive sudo password for Ingress's privileged ports -- not"
    echo "available from an unattended script, and the lab environment"
    echo "disallows sudo entirely)."
    ${KUBECTL} port-forward -n ingress-nginx svc/ingress-nginx-controller 18080:80 \
      >/tmp/heart_mlops_pf_ingress.log 2>&1 &
    INGRESS_PF_PID=$!
    sleep 3
    INGRESS_URL="http://127.0.0.1:18080"
  else
    INGRESS_URL="http://$(minikube ip)"
  fi

  # A `type: LoadBalancer` Service only gets a real external IP once
  # `minikube tunnel` is actually running (in another terminal) -- on
  # every OS, not just macOS. If it hasn't been assigned one, fall back
  # to port-forward rather than guessing the node IP works directly.
  LB_IP="$(${KUBECTL} -n heart-mlops get svc heart-api-lb -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)"
  if [ -n "$LB_IP" ]; then
    LB_URL="http://${LB_IP}:8000"
  else
    echo
    echo "heart-api-lb has no external IP yet ('minikube tunnel' not"
    echo "running) -- port-forwarding it to a local port instead."
    ${KUBECTL} port-forward -n heart-mlops svc/heart-api-lb 18089:8000 \
      >/tmp/heart_mlops_pf_lb.log 2>&1 &
    LB_PF_PID=$!
    sleep 3
    LB_URL="http://127.0.0.1:18089"
  fi

  echo
  echo "=== /health via Ingress (Host: heart-api.local, ${INGRESS_URL}) ==="
  curl -fsS -H "Host: heart-api.local" "${INGRESS_URL}/health"; echo

  echo "=== /predict via Ingress ==="
  curl -fsS -H "Host: heart-api.local" -X POST "${INGRESS_URL}/predict" \
    -H "Content-Type: application/json" -d @test/sample_requests/valid_high_risk.json; echo

  echo
  echo "=== /health via LoadBalancer (${LB_URL}) ==="
  curl -fsS "${LB_URL}/health"; echo

  echo "=== /predict via LoadBalancer ==="
  curl -fsS -X POST "${LB_URL}/predict" \
    -H "Content-Type: application/json" -d @test/sample_requests/valid_high_risk.json; echo
} 2>&1 | tee "${LOG_FILE}"

echo "Saved to ${LOG_FILE}"
