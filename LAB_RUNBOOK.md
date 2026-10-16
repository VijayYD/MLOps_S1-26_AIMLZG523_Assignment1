# Lab Runbook (BITS Prayogshala)

Copy-paste steps for a fresh machine. Every command here is identical to
what's used on the Mac — no lab-only branches, no `sudo` anywhere. Run
from the repo root unless stated otherwise.

## 1. Clone and preflight

```bash
git clone https://github.com/VijayYD/MLOps_S1-26_AIMLZG523_Assignment1.git
cd MLOps_S1-26_AIMLZG523_Assignment1
bash scripts/check_env.sh
```

Review the output: it reports python3/pip/git/curl (hard requirements —
the script exits non-zero if any are missing), docker, minikube, kubectl
(or the `minikube kubectl --` fallback), make, CPU/RAM, and whether ports
8000/5001/9090/3000 are free. Resolve any `[FAIL]` before continuing;
`[WARN]` lines are informational (e.g. a version that doesn't exactly
match the pin, or Docker not running yet).

## 2. Python environment

```bash
bash scripts/setup_venv.sh
source .venv/bin/activate
```

This resolves the Python interpreter itself (prefers a `3.12.x` already
on `$PATH`; falls back to a pyenv-managed `3.12.x` install if present) —
see `docs/DECISIONS.md` D0.3 for why it doesn't just trust `python3` on
`$PATH` blindly. If neither is available, install Python 3.12.9 first
(e.g. via your distro's package manager or pyenv) and re-run.

## 3. Data, training, and tests

```bash
make data    # downloads the 4 UCI site files (checksum-verified) and cleans
             # them -- or use the committed data/processed/heart_combined_clean.csv
             # directly if UCI is unreachable from the lab network
make test    # pytest -- JUnit XML + coverage XML/HTML, fails under 90% coverage
make train   # tunes all 3 models, selects one by the pre-registered rule
             # (docs/DECISIONS.md D3.1), logs to MLflow, packages the model.
             # Takes several minutes (full hyperparameter search).
make mlflow-ui   # serves http://127.0.0.1:5001
```

If `make data` fails because UCI is unreachable from the lab network, the
already-committed `data/processed/heart_combined_clean.csv` is used
automatically by anything that reads it (`make train` doesn't re-download
if the cleaned CSV is already there) — no manual fallback step needed.

## 4. Docker

```bash
make build-image     # builds heart-mlops-api:latest
make docker-smoke     # builds + runs + waits for health + checks every
                      # test/sample_requests/*.json against its expected
                      # status, saves the log to screenshots/logs/
```

No `sudo` is needed for Docker in the lab (Docker Engine runs without it
per the lab's own setup).

CI (`.github/workflows/ci.yml`) also pushes this same image to GitHub
Container Registry on every push to `main`, tagged by git SHA and
`latest` — nothing to do here for that, it happens automatically on
GitHub's own runners once the smoke test passes.

## 5. Kubernetes (Minikube)

```bash
make k8s-deploy
```

This starts minikube (`--cpus=2 --memory=4096`; lower these if the lab
machine is more constrained — pass `MINIKUBE_CPUS=1 MINIKUBE_MEMORY=2048
make k8s-deploy` to override), enables the `ingress` and `metrics-server`
addons, builds the image inside minikube's own Docker daemon (no registry
push needed), applies the Kustomize overlay, and waits for every
Deployment to roll out. A Grafana admin password is read from `.env`
(copy `.env.example` to `.env` first if you haven't) — never hard-coded.

If `kubectl` isn't on `PATH` in the lab, every script here already falls
back to `minikube kubectl --` automatically (see `scripts/_k8s_lib.sh`).

```bash
make k8s-verify
```

Prints `kubectl get pods,svc,ingress,hpa -n heart-mlops`, then calls
`/health` and `/predict` through **both** the Ingress (host
`heart-api.local`, reached with a `Host:` header — never editing
`/etc/hosts`) and the LoadBalancer Service. **On Linux (the lab),
`minikube ip` is directly reachable**, so this needs no tunnel and no
`sudo` at all. (On macOS, the same script uses `kubectl port-forward`
instead of `minikube tunnel`, specifically to avoid the interactive
`sudo` password `minikube tunnel` would need for the Ingress's privileged
ports — see `docs/DECISIONS.md` D7.7. Either way: no `sudo`.)

```bash
make load-test    # small concurrent traffic generator, so Grafana has real data
```

### Monitoring

```bash
${KUBECTL:-kubectl} -n heart-mlops port-forward svc/prometheus 9090:9090 &
${KUBECTL:-kubectl} -n heart-mlops port-forward svc/grafana 3000:3000 &
```

Then open `http://127.0.0.1:9090/targets` (Prometheus) and
`http://127.0.0.1:3000` (Grafana — the "Heart Disease API" dashboard is
already provisioned; admin password is whatever you set in `.env`).

### Teardown

```bash
make k8s-teardown     # removes the app + monitoring stack; leaves minikube running
minikube stop         # or: minikube delete, to remove the cluster entirely
```

## 6. Lab-parity verification (optional, already run on the Mac before this runbook was written)

```bash
make lab-parity
```

Re-proves the exact dependency pins work inside a fresh
`python:3.12.9-slim` container — useful if the lab's own Python 3.12.9
setup differs subtly from what `check_env.sh` reports.

## Troubleshooting

- **`make: command not found`**: `make` is optional everywhere in this
  runbook -- every target has a plain bash equivalent, no `sudo` needed
  either way:

  | `make <target>` | Equivalent |
  |---|---|
  | `check-env` | `bash scripts/check_env.sh` |
  | `venv` | `bash scripts/setup_venv.sh` |
  | `lint` | `.venv/bin/ruff check . && .venv/bin/ruff format --check .` |
  | `data` | `.venv/bin/python -m heart_mlops.data.download && .venv/bin/python -m heart_mlops.data.clean` |
  | `test` | `.venv/bin/pytest` |
  | `train` | `bash scripts/train.sh` |
  | `mlflow-ui` | `bash scripts/mlflow_ui.sh` |
  | `build-image` | `bash scripts/build_image.sh` |
  | `docker-smoke` | `bash scripts/run_container_smoke.sh` |
  | `compose-up` / `compose-down` | `docker compose -f docker/docker-compose.yml up -d --build` / `... down` |
  | `k8s-deploy` | `bash scripts/k8s_deploy.sh` |
  | `k8s-verify` | `bash scripts/k8s_verify.sh` |
  | `k8s-teardown` | `bash scripts/k8s_teardown.sh` |
  | `load-test` | `bash scripts/load_test.sh` |
  | `lab-parity` | `bash scripts/lab_parity_check.sh` |
  | `report` | `.venv/bin/python reports/build_report.py` |

- **minikube won't start / not enough resources**: lower the resource
  request, e.g. `MINIKUBE_CPUS=1 MINIKUBE_MEMORY=2048 make k8s-deploy`. If
  a previous `minikube` profile is in a bad state, `minikube delete` and
  re-run `make k8s-deploy`.
- **`ingress` addon stuck "Verifying ingress addon..."**: usually just
  slow on a loaded machine. Check `kubectl get pods -n ingress-nginx`;
  re-run `minikube addons enable ingress` if a pod is stuck pending.
- **`kubectl: command not found`**: every script here (`k8s_deploy.sh`,
  `k8s_verify.sh`, `k8s_teardown.sh`) auto-falls-back to
  `minikube kubectl --` via `scripts/_k8s_lib.sh` — nothing to do.
- **Port already in use** (8000/5001/9090/3000), e.g. `docker compose`
  failing with `address already in use`: `scripts/check_env.sh` reports
  which ports are busy before you start; check `docker ps -a` for a
  leftover container from an earlier run that was never torn down first
  (`docker rm -f <name>`, or `docker compose -f docker/docker-compose.yml
  down` if it was a previous compose stack). If the port is genuinely
  held by something else, override it instead of trying to free it:
  `PORT=8001 bash scripts/run_container_smoke.sh` for the standalone
  smoke test, or `API_PORT=8001 PROMETHEUS_PORT=9091
  GRAFANA_PORT=3001 make compose-up` for the docker-compose stack.
- **Corporate proxy / restricted network blocks UCI or Docker Hub**: set
  the usual `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY` environment variables
  before running any command here — nothing in this repo hard-codes a
  direct connection. If UCI is unreachable even with a proxy configured,
  `make train` still works from the already-committed
  `data/processed/heart_combined_clean.csv` (see step 3).
- **No `sudo` anywhere**: if any command in this runbook ever prompts for
  a password, something has gone wrong — every step here is designed to
  avoid it entirely. Stop and re-check which command triggered it before
  entering a password.
