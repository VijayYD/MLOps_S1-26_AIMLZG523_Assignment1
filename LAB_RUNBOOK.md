# Lab Runbook (BITS Prayogshala)

**Status:** written in Phase 8 (lab-parity verification), as copy-paste steps
verified against a fresh clone and a `python:3.12.9-slim` container on the Mac
first. Will cover: clone, `scripts/check_env.sh`, venv setup, data download,
training, tests, Docker build + smoke test, Minikube deploy, verification,
monitoring port-forwards, teardown, and troubleshooting (minikube resources,
ingress addon, `minikube kubectl` fallback, port conflicts) — no `sudo`
anywhere.
