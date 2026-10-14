#!/usr/bin/env bash
# Removes the application + monitoring stack from the cluster. Leaves
# minikube itself running -- `minikube stop` or `minikube delete` separately
# if you want the cluster gone too.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=_k8s_lib.sh
source scripts/_k8s_lib.sh

${KUBECTL} delete -k k8s/overlays/minikube --ignore-not-found
${KUBECTL} delete secret grafana-admin -n heart-mlops --ignore-not-found

echo "Torn down. minikube itself is still running (minikube stop / minikube delete to remove the cluster)."
