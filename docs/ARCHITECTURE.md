# Architecture

**Status:** written in Phase 9, once every component below actually exists and
has been run. Will cover, with a diagram generated as code
(`reports/figures/architecture.png`):

data acquisition → cleaning → feature pipeline → model training/tuning →
MLflow tracking + model registry → packaged model artefact → CI/CD
(lint/test/train/docker jobs) → Docker image → Minikube (Ingress / LoadBalancer
→ API pods) → Prometheus → Grafana.
