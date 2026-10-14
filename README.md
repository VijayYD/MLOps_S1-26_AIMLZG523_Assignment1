# Heart Disease Risk Prediction — End-to-End MLOps

An end-to-end MLOps pipeline for predicting the presence of heart disease from
patient health attributes: data acquisition and EDA, feature engineering and
model development, experiment tracking, packaging, CI/CD, containerisation,
Kubernetes deployment, and monitoring.

Dataset: the combined four-site UCI Heart Disease dataset (Cleveland,
Hungarian, Switzerland, VA Long Beach; ~920 rows), which brings real
site-dependent missingness and distribution shift rather than the smaller,
cleaner 303-row Cleveland-only subset usually seen. See
[`data/README.md`](data/README.md) for provenance and the column dictionary
(populated in Phase 2).

**Status:** data acquisition/EDA, feature engineering/modelling, and
experiment tracking are done. Serving, deployment and monitoring sections
below will be filled in as the corresponding phases land.

## Repository map

```
src/heart_mlops/   Python package: data, features, models, api
notebooks/         EDA / modelling / inference-demo notebooks
scripts/           check_env, setup_venv, and (later) download_data, train,
                   build_image, k8s_deploy, etc. -- bash equivalents of make targets
test/              pytest unit + API tests
docker/            Dockerfile + docker-compose.yml
k8s/               Kustomize manifests (base / monitoring / overlays/minikube)
monitoring/        Prometheus + Grafana config (provisioned, not click-ops)
models/            Trained model artefact + metadata + model card
reports/           Generated figures + the DOCX report
docs/              Design decisions, architecture, video script
```

## Prerequisites

- Python **3.13.x** on macOS (dev) / **3.12.x** on Linux (BITS Prayogshala lab)
- git, curl, make
- Docker (for Phases 7+)
- minikube **v1.35.0** and kubectl (for Phases 7+; a missing `kubectl` falls
  back to `minikube kubectl --` everywhere)

Run the preflight check first, on either OS:

```bash
bash scripts/check_env.sh
```

It reports versions of every tool above, CPU/RAM, and whether ports
8000 (API) / 5001 (MLflow UI) / 9090 (Prometheus) / 3000 (Grafana) are free.
It exits non-zero only if a hard requirement (python3, pip, git, curl) is
missing; everything else is reported as a warning so you can decide how to
proceed. Port 5000 is deliberately avoided for MLflow because it's used by
AirPlay Receiver on macOS.

## Quick start

```bash
bash scripts/check_env.sh      # or: make check-env
bash scripts/setup_venv.sh     # or: make venv
source .venv/bin/activate
```

`scripts/setup_venv.sh` picks the Python version pinned for your OS (3.13.x on
macOS, 3.12.x on Linux) rather than trusting whatever `python3` happens to
resolve to on `$PATH` — see [`docs/DECISIONS.md`](docs/DECISIONS.md#d03--scriptssetup_venvsh-does-not-blindly-trust-python3-on-path)
for why. It then installs `requirements-dev.txt` (a superset of
`requirements.txt`) and prints `pip list`.

## Data, training, and experiment tracking

```bash
make data        # downloads the 4 UCI site files (checksum-verified) and cleans them into
                 # data/processed/heart_combined_clean.csv
make train       # tunes all 3 models, selects one by the pre-registered rule in
                 # docs/DECISIONS.md D3.1, logs everything to MLflow, registers the
                 # champion -- or: bash scripts/train.sh (also runs `make data` first
                 # if the cleaned CSV isn't present yet)
make mlflow-ui   # serves the MLflow UI at http://127.0.0.1:5001 -- or: bash scripts/mlflow_ui.sh
```

Each training run creates one MLflow parent run ("training-session") with
one nested child run per model (hyperparameters, CV + hold-out metrics,
`cv_results.csv`, classification report, confusion matrix); the selected
model's child run additionally gets a feature-importance plot and the
logged model itself (with signature + input example). The parent run also
gets the model comparison table, ROC/PR/calibration plots, and the model
card. The selected model is registered under
`heart_disease_classifier` and aliased `champion` -- not MLflow's
deprecated Staging/Production *stages* (see
[`docs/DECISIONS.md`](docs/DECISIONS.md) D4.2) -- so serving code should
always resolve it as `models:/heart_disease_classifier@champion`, never by
a hardcoded version number. Both the backend store (`mlflow.db`) and
artifacts (`mlartifacts/`) are local and gitignored by default;
`MLFLOW_TRACKING_URI` overrides this without code changes.

## Model packaging

`make train` also writes `models/heart_model.joblib` (the full
preprocessing + classifier pipeline) and `models/model_metadata.json`
(model name, registry version, threshold, feature schema, hold-out
metrics, training data checksum, sklearn version, git SHA, timestamp).
These are the *same* fitted pipeline just registered in MLflow, saved a
second time in a serving-friendly format — the API (Phase 7) loads this
pair directly rather than depending on an MLflow server being reachable at
serving time (`docs/DECISIONS.md` D5.1/D5.2). Both files are small enough
to commit, so `docker build` works on a fresh clone without retraining.
See [`models/MODEL_CARD.md`](models/MODEL_CARD.md) for intended use,
limitations, and per-site performance.

`test/test_reproducibility.py` checks that re-fitting the packaged model's
exact hyperparameters on the same data twice gives matching predictions
(`pytest test/test_reproducibility.py`) — a fast determinism check, not a
re-run of the full tuning search (`docs/DECISIONS.md` D5.3).

## Design decisions

Every non-obvious choice in this project (dataset, missing-value handling,
Python/minikube version pinning, threshold selection, etc.) is logged with its
rationale and alternatives considered in [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Roadmap

| Phase | Status |
|---|---|
| 0 — Preflight | Done |
| 1 — Repository scaffold | Done |
| 2 — Data acquisition & EDA | Done |
| 3 — Feature engineering & modelling | Done |
| 4 — Experiment tracking (MLflow) | Done |
| 5 — Packaging & reproducibility | Done |
| 6 — CI/CD & automated testing | Not started |
| 7 — Containerisation, Kubernetes, monitoring | Not started |
| 8 — Lab-parity verification | Not started |
| 9 — Documentation & reporting | Not started |
