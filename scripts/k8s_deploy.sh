#!/usr/bin/env bash
# Starts minikube if needed, enables required addons, builds the image
# inside minikube's own Docker daemon (no registry push needed), applies
# the Kustomize overlay, and waits for every Deployment to roll out.
set -euo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=_k8s_lib.sh
source scripts/_k8s_lib.sh

MINIKUBE_CPUS="${MINIKUBE_CPUS:-2}"
MINIKUBE_MEMORY="${MINIKUBE_MEMORY:-4096}"

if ! minikube status >/dev/null 2>&1; then
  echo "Starting minikube (driver=docker, cpus=${MINIKUBE_CPUS}, memory=${MINIKUBE_MEMORY})..."
  minikube start --driver=docker --cpus="${MINIKUBE_CPUS}" --memory="${MINIKUBE_MEMORY}"
else
  echo "minikube already running."
fi

echo "Enabling addons (ingress, metrics-server)..."
minikube addons enable ingress
minikube addons enable metrics-server

echo "Building the image inside minikube's Docker daemon..."
eval "$(minikube docker-env)"
IMAGE_NAME=heart-mlops-api IMAGE_TAG=latest bash scripts/build_image.sh
eval "$(minikube docker-env -u)"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
GRAFANA_ADMIN_PASSWORD="${GRAFANA_ADMIN_PASSWORD:-changeme}"

echo "Ensuring namespace and Grafana admin secret exist..."
${KUBECTL} create namespace heart-mlops --dry-run=client -o yaml | ${KUBECTL} apply -f -
${KUBECTL} create secret generic grafana-admin \
  --from-literal=admin-password="${GRAFANA_ADMIN_PASSWORD}" \
  -n heart-mlops --dry-run=client -o yaml | ${KUBECTL} apply -f -

echo "Applying k8s/overlays/minikube..."
${KUBECTL} apply -k k8s/overlays/minikube

echo "Waiting for rollout..."
${KUBECTL} -n heart-mlops rollout status deployment/heart-api --timeout=180s
${KUBECTL} -n heart-mlops rollout status deployment/prometheus --timeout=120s
${KUBECTL} -n heart-mlops rollout status deployment/grafana --timeout=120s

echo "Deployed. Run scripts/k8s_verify.sh to check connectivity through Ingress and LoadBalancer."
