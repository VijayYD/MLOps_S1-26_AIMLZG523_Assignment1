#!/usr/bin/env bash
# Builds the API Docker image with OCI labels for version and git SHA.
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE_NAME="${IMAGE_NAME:-heart-mlops-api}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
GIT_SHA="$(git rev-parse HEAD)"

docker build \
  --platform linux/amd64 \
  -f docker/Dockerfile \
  -t "${IMAGE_NAME}:${IMAGE_TAG}" \
  --build-arg GIT_SHA="${GIT_SHA}" \
  --build-arg VERSION="${IMAGE_TAG}" \
  .

echo "Built ${IMAGE_NAME}:${IMAGE_TAG} (git sha ${GIT_SHA})"
