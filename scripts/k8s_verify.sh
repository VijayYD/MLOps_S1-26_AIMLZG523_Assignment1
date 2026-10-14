#!/usr/bin/env bash
# Prints cluster state, then calls /health and /predict through BOTH the
# Ingress and the LoadBalancer Service.
#
# macOS + Docker driver: minikube ip is not routable from the host, and
# `minikube tunnel` needs sudo for the Ingress's privileged ports (80/443)
# -- which needs an interactive password prompt, and the lab explicitly
# forbids sudo anyway (see the project brief). So this uses
# `kubectl port-forward` to arbitrary unprivileged local ports for both
# Ingress and LoadBalancer on macOS instead -- same verification, no sudo,
# works unattended. See docs/DECISIONS.md D7.x for the full reasoning.
# Linux: minikube ip is directly reachable, so no tunnel/port-forward is
# needed there at all.
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
    echo "macOS detected: port-forwarding the Ingress controller and the"
    echo "LoadBalancer Service to local ports instead of 'minikube tunnel'"
    echo "(which would need an interactive sudo password for Ingress's"
    echo "privileged ports -- not available from an unattended script, and"
    echo "the lab environment disallows sudo entirely)."
    ${KUBECTL} port-forward -n ingress-nginx svc/ingress-nginx-controller 18080:80 \
      >/tmp/heart_mlops_pf_ingress.log 2>&1 &
    INGRESS_PF_PID=$!
    ${KUBECTL} port-forward -n heart-mlops svc/heart-api-lb 18089:8000 \
      >/tmp/heart_mlops_pf_lb.log 2>&1 &
    LB_PF_PID=$!
    sleep 3
    INGRESS_URL="http://127.0.0.1:18080"
    LB_URL="http://127.0.0.1:18089"
  else
    INGRESS_IP="$(minikube ip)"
    LB_IP="$(${KUBECTL} -n heart-mlops get svc heart-api-lb -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)"
    [ -z "$LB_IP" ] && LB_IP="$INGRESS_IP"
    INGRESS_URL="http://${INGRESS_IP}"
    LB_URL="http://${LB_IP}:8000"
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
