# Architecture

![Architecture diagram](../reports/figures/architecture.png)

Generated as code by [`reports/build_architecture_diagram.py`](../reports/build_architecture_diagram.py)
(`python reports/build_architecture_diagram.py` to regenerate).

## Flow

1. **Data & EDA** (Phase 2) — `src/heart_mlops/data/download.py` pulls the
   four UCI site files (checksum-verified, with a fallback), `clean.py`
   produces `data/processed/heart_combined_clean.csv`, and
   `notebooks/01_eda.ipynb` documents the missingness/site-shift findings
   that drove the feature decisions below.
2. **Features & modelling** (Phase 3) — `src/heart_mlops/features/preprocess.py`
   builds one `ColumnTransformer`-based `Pipeline` per model;
   `src/heart_mlops/models/train.py` tunes Logistic Regression, Random
   Forest, and HistGradientBoostingClassifier with `RandomizedSearchCV`,
   and selects a winner by the rule pre-registered in `docs/DECISIONS.md`
   D3.1 — before any hold-out result is looked at.
3. **Tracking & packaging** (Phase 4-5) — the same training run logs
   every model's params/metrics/artifacts to **MLflow** (one parent run,
   one nested child run per model) and registers the selected model as
   `heart_disease_classifier`, aliased `champion`. It also packages the
   *exact same* fitted pipeline as `models/heart_model.joblib` +
   `model_metadata.json` — the thing the API actually loads, with no
   MLflow server needed at serving time (D5.1/D5.2).
4. **CI/CD & container** (Phase 6-7) — `.github/workflows/ci.yml` runs
   lint, tests (Python 3.12.9 and 3.13.5), a training run, and a Docker
   build + container smoke test on every push/PR.
   `docker/Dockerfile` packages the FastAPI app (`src/heart_mlops/api/`)
   with the committed model artifact into a non-root,
   `python:3.12.9-slim`-based image. On a real push to `main`, once the
   smoke test passes, the same image is also pushed to GitHub Container
   Registry (`ghcr.io/<owner>/heart-mlops-api`, tagged by git SHA and
   `latest`) using the workflow's own `GITHUB_TOKEN` (D7.9).
5. **Kubernetes serving** (Phase 7) — `k8s/` (Kustomize: `base/` +
   `monitoring/` + `overlays/minikube/`) deploys 2 API pod replicas
   (HPA 2-5 on CPU) behind both an Ingress (`heart-api.local`) and a
   LoadBalancer Service.
6. **Monitoring** (Phase 7) — Prometheus scrapes the API pods directly
   (Kubernetes pod discovery, not a static target) for custom metrics
   (request count/latency, predictions by class, risk-probability
   histogram, inference errors); Grafana's datasource and the
   "Heart Disease API" dashboard are both provisioned as code, no manual
   clicking.
