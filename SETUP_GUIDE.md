# Setup and Usage Guide

A complete, self-contained walkthrough for setting up, running, testing, and
deploying this project on any Linux or macOS machine. Every command is
copy-pasteable and runs from the repository root unless stated otherwise.
Nothing in this guide requires administrator/root privileges.

If a command's expected output isn't shown, the command either prints its
own progress clearly or exits with status `0` on success — check with
`echo $?` immediately after if you want to confirm.

---

## 1. What this project is

An end-to-end machine learning pipeline that predicts the presence of
heart disease from patient clinical attributes, covering:

- Data acquisition and cleaning (the combined four-site UCI Heart Disease
  dataset)
- Exploratory data analysis
- Model training, tuning, and selection (Logistic Regression, Random
  Forest, HistGradientBoostingClassifier)
- Experiment tracking and model registry (MLflow)
- A packaged, reproducible model artifact
- A FastAPI serving layer with input validation, structured logging, and
  Prometheus metrics
- Automated tests and a CI pipeline (GitHub Actions)
- A Docker image and Kubernetes manifests (Kustomize)
- Monitoring (Prometheus + Grafana, provisioned as code)

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for a diagram and
[`docs/DECISIONS.md`](docs/DECISIONS.md) for the reasoning behind every
non-obvious choice.

## 2. Prerequisites

Install these before starting:

| Tool | Why | Notes |
|---|---|---|
| `git` | clone the repository | any recent version |
| Python **3.12.x or 3.13.x** | run the project | pinned dependencies are verified against 3.12.9 and 3.13.5 specifically |
| `pip` | install Python packages | ships with Python |
| `curl` | health checks, API testing | any recent version |
| `make` | convenience wrapper around the scripts below | optional — every `make <target>` has an equivalent `bash scripts/<name>.sh` |
| `docker` (with the Compose plugin) | build/run the container, run Kubernetes locally | Docker daemon must be running |
| `minikube` **v1.35.0** | run a local Kubernetes cluster | exact version matters — mismatched versions can behave differently |
| `kubectl` | talk to the Kubernetes cluster | optional — if missing, every script here automatically falls back to `minikube kubectl --` |

Check what you already have and what's missing:

```bash
bash scripts/check_env.sh
```

This prints the version of every tool above, your CPU/RAM, and whether
ports `8000` (API), `5001` (MLflow UI), `9090` (Prometheus), and `3000`
(Grafana) are currently free. It exits with a non-zero status only if a
hard requirement (`python3`, `pip`, `git`, `curl`) is missing — everything
else is a warning you can act on as needed.

## 3. Clone the repository

```bash
git clone https://github.com/VijayYD/MLOps_S1-26_AIMLZG523_Assignment1.git
cd MLOps_S1-26_AIMLZG523_Assignment1
```

## 4. Set up the Python environment

```bash
bash scripts/setup_venv.sh
source .venv/bin/activate
```

This creates a `.venv` virtual environment using a Python 3.12.x or 3.13.x
interpreter (it actively looks for one — checking `python3` on your `PATH`
first, then any pyenv-managed installs — rather than blindly trusting
whatever `python3` happens to resolve to), installs every pinned
dependency from `requirements-dev.txt` (which includes everything in
`requirements.txt` plus testing/linting/notebook tools), installs this
project itself in editable mode, and prints the final `pip list` so you
can see exactly what was installed.

If no Python 3.12.x/3.13.x interpreter is found, the script tells you so
and exits — install one (via your package manager, or a tool like
[pyenv](https://github.com/pyenv/pyenv)) and re-run it.

**From this point on, keep the virtual environment activated** (you'll
see `(.venv)` in your shell prompt). If you open a new terminal, re-run
`source .venv/bin/activate` first.

## 5. Download and clean the data

```bash
make data
```

or, without `make`:

```bash
python -m heart_mlops.data.download
python -m heart_mlops.data.clean
```

This downloads the four source files from the UCI Machine Learning
Repository (with SHA-256 checksum verification and an automatic fallback
if the primary source is briefly unreachable), then cleans and combines
them into `data/processed/heart_combined_clean.csv`. A cleaned copy of
this file is already committed to the repository, so **this step is
optional** — everything downstream works from the committed file if you
skip it.

Expect output ending with a small table summarising rows, missing data,
and class balance per site.

## 6. Explore the data

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

Then open `notebooks/01_eda.ipynb` in Jupyter or VS Code to see the
missingness heatmaps, class balance charts, and correlation analysis —
or just browse the already-executed notebook as committed, and the
corresponding figures in `reports/figures/01_*.png` through `06_*.png`.

## 7. Train the models and track experiments

```bash
make train
```

or:

```bash
bash scripts/train.sh
```

This tunes three candidate models with cross-validated hyperparameter
search, applies a pre-registered model-selection rule, logs every run's
parameters/metrics/artifacts to a local MLflow tracking store
(`mlflow.db` + `mlartifacts/`, both created automatically), registers the
selected model, and writes the packaged model artifact to
`models/heart_model.joblib` + `models/model_metadata.json`. **This takes
several minutes** (a full hyperparameter search across three models).

A committed copy of the packaged model already exists in the repository,
so this step is also optional if you just want to run the API or the
container — only needed if you want to see training happen or regenerate
the artifact yourself.

View the tracked experiments:

```bash
make mlflow-ui
```

Then open `http://127.0.0.1:5001` in a browser.

## 8. Run the test suite

```bash
make test
```

or:

```bash
pytest
```

Runs the full suite (data, preprocessing, model/training, and API tests —
none of which require real network access or a trained model to exist
first) and prints a coverage report. The build is configured to fail if
coverage drops below 90%.

## 9. Lint the code

```bash
make lint
```

or:

```bash
ruff check .
ruff format --check .
```

## 10. Run the API locally

```bash
uvicorn heart_mlops.api.main:app --reload --port 8000
```

Then, in another terminal:

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @test/sample_requests/valid_high_risk.json
```

Open `http://127.0.0.1:8000/docs` in a browser for interactive API
documentation with example payloads.

## 11. Build and test the Docker image

```bash
make docker-smoke
```

or:

```bash
bash scripts/run_container_smoke.sh
```

Builds the image, starts a container, waits for it to report healthy,
sends every sample request in `test/sample_requests/` and checks each
response against its expected status code, and saves the full console
output to `screenshots/logs/`.

To just build the image without running the smoke test:

```bash
make build-image
```

To run the API alongside Prometheus and Grafana locally, without
Kubernetes:

```bash
make compose-up      # starts all three
make compose-down     # stops them
```

The CI pipeline also publishes this same image to GitHub Container
Registry automatically on every push to the main branch, once the smoke
test passes — nothing to run locally for that.

## 12. Deploy to a local Kubernetes cluster

```bash
make k8s-deploy
```

or:

```bash
bash scripts/k8s_deploy.sh
```

This starts a local Minikube cluster if one isn't already running,
enables the `ingress` and `metrics-server` addons, builds the Docker
image directly inside the cluster's own container runtime (no image
registry needed), applies the Kubernetes manifests, and waits for every
deployment to finish rolling out.

**Before running this for the first time**, copy the example environment
file and set a Grafana admin password:

```bash
cp .env.example .env
# edit .env and set GRAFANA_ADMIN_PASSWORD to something of your choosing
```

To adjust the cluster's resource allocation:

```bash
MINIKUBE_CPUS=2 MINIKUBE_MEMORY=4096 make k8s-deploy
```

### Verify the deployment

```bash
make k8s-verify
```

Prints the status of every pod, service, ingress, and autoscaler, then
calls `/health` and `/predict` through both an Ingress and a
LoadBalancer Service to confirm the API is actually reachable end to end
(not just that the pods are "Running"). If the cluster's internal IP
isn't directly reachable from your machine (common with some
container-based Kubernetes setups), this script automatically uses port
forwarding instead — no manual configuration needed either way.

### Generate traffic and view monitoring

```bash
make load-test
```

Sends a short burst of real requests so the monitoring dashboards have
something to show. Then, in separate terminals:

```bash
${KUBECTL:-kubectl} -n heart-mlops port-forward svc/prometheus 9090:9090
${KUBECTL:-kubectl} -n heart-mlops port-forward svc/grafana 3000:3000
```

Open `http://127.0.0.1:9090/targets` (Prometheus should show the API
pods as "up") and `http://127.0.0.1:3000` (Grafana — log in with
`admin` and whatever you set `GRAFANA_ADMIN_PASSWORD` to; the "Heart
Disease API" dashboard is already there, no setup needed).

### Tear down

```bash
make k8s-teardown
```

Removes the application and monitoring stack. The Minikube cluster
itself keeps running — stop it with `minikube stop`, or remove it
entirely with `minikube delete`.

## 13. Verify dependency pins in a clean environment

```bash
make lab-parity
```

or:

```bash
bash scripts/lab_parity_check.sh
```

Takes a snapshot of exactly what's committed to version control, runs it
inside a brand-new `python:3.12.9-slim` container with no cached
packages, and re-runs linting, the full test suite, and a short training
pass there. This is the strongest check that the pinned dependency
versions actually work — not just that they look compatible on paper —
on a machine with nothing pre-installed.

## 14. Generate the project report

```bash
make report
```

or:

```bash
python reports/build_architecture_diagram.py
python reports/build_report.py
```

Regenerates the architecture diagram and the full written report
(`reports/MLOps_Assignment1_Report.docx`) from the current model
metadata and figures. Any screenshot saved at the exact path listed in
`screenshots/SCREENSHOT_CHECKLIST.md` is embedded automatically; anything
missing shows as a clearly labelled placeholder instead.

## Troubleshooting

- **A port is already in use** (`8000`, `5001`, `9090`, `3000`, `8080`):
  find and stop whatever else is using it, or override the port for the
  one command that needs it — for example `PORT=8001 bash
  scripts/run_container_smoke.sh`.
- **Minikube won't start, or runs out of resources**: lower the request,
  e.g. `MINIKUBE_CPUS=1 MINIKUBE_MEMORY=2048 make k8s-deploy`. If a
  previous cluster is in a broken state, `minikube delete` and try again.
- **The `ingress` addon seems stuck**: check `kubectl get pods -n
  ingress-nginx` — if a pod is stuck pending, re-run `minikube addons
  enable ingress`.
- **`kubectl: command not found`**: every script here falls back to
  `minikube kubectl --` automatically; nothing to do.
- **A restricted network blocks the UCI dataset or Docker Hub**: set the
  usual `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY` environment variables
  before running any command — nothing here hard-codes a direct
  connection. If the UCI dataset is unreachable even with a proxy set,
  training still works from the already-committed
  `data/processed/heart_combined_clean.csv` (step 5 is optional for
  exactly this reason).
- **Something asks for a password/elevated privileges**: nothing in this
  guide should. Stop and check which command triggered it before
  entering credentials.

## Where to go next

- [`README.md`](README.md) — project overview and a more detailed
  breakdown of each component
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — architecture diagram
  and walkthrough
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — every design decision and
  its rationale
- [`models/MODEL_CARD.md`](models/MODEL_CARD.md) — intended use,
  limitations, and performance by data source
- [`reports/MLOps_Assignment1_Report.docx`](reports/MLOps_Assignment1_Report.docx) —
  the full written report
