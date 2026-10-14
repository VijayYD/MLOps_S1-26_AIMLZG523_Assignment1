# Sourced by k8s_deploy.sh / k8s_verify.sh / k8s_teardown.sh.
# Sets KUBECTL to "kubectl" if present on PATH, else falls back to
# "minikube kubectl --" (the lab's kubectl may be absent).
if command -v kubectl >/dev/null 2>&1; then
  KUBECTL="kubectl"
else
  KUBECTL="minikube kubectl --"
fi
