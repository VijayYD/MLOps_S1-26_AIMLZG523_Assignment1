# Video Walkthrough Script

Target length: 6-7 minutes. Record on the Mac with the built-in screen
recorder (**Cmd+Shift+5** → "Record Selected Portion" or "Record Entire
Screen"). Record audio narration live while screen-recording, or record
narration separately and layer it in afterward — either works.

Have a terminal (large font, dark theme recommended for readability) and
a browser tab ready before you start recording. Run `make k8s-deploy`
once beforehand so the cluster is already warm — don't record a 2-minute
`minikube start` wait.

---

## Scene 1 — Introduction (30s)

**Show:** Title slide or just the repo's GitHub page / README rendered.

**Say:** "This is an end-to-end MLOps pipeline for predicting heart
disease risk from patient clinical attributes. It uses the combined
four-site UCI Heart Disease dataset — not just the common 303-row
Cleveland subset — which brings real site-dependent missingness and
distribution shift into the problem. The pipeline covers data
acquisition, modelling, experiment tracking, packaging, CI/CD,
containerisation, Kubernetes deployment, and monitoring."

**Commands:** none yet.

---

## Scene 2 — Data and EDA (45s)

**Show:** `notebooks/01_eda.ipynb` scrolled to the missingness heatmap and
the per-site class balance chart.

**Say:** "The four sites differ enormously: Hungarian, Switzerland and VA
are missing `ca`, `thal` and `slope` in 60 to 99 percent of rows, and
disease prevalence ranges from 36 percent in Hungary to over 93 percent
in Switzerland. That's the evidence behind every missing-value and
feature decision later in the pipeline — logged in `docs/DECISIONS.md`."

**Commands:**
```bash
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb
```
(or just scroll the already-executed notebook if it's already open).

---

## Scene 3 — Modelling and experiment tracking (90s)

**Show:** `notebooks/02_modelling.ipynb`'s model comparison table and the
leave-one-site-out bar chart, then switch to the MLflow UI.

**Say:** "Three models were tuned — Logistic Regression, Random Forest,
and HistGradientBoostingClassifier, which handles missing values
natively. The selection rule was written down and committed *before* any
hold-out result was ever looked at: highest cross-validated ROC-AUC wins.
Logistic Regression was selected. Beyond standard cross-validation, a
leave-one-site-out evaluation trains on three sites and tests entirely on
the fourth — recall holds up for three of four sites, but Cleveland is an
exception, and that's reported honestly rather than hidden."

"Every training run logs to MLflow: one parent run, one nested child run
per model, with the selected model registered and aliased 'champion' —
not a hard-coded version number."

**Commands:**
```bash
make mlflow-ui
```
Then open http://127.0.0.1:5001 in the browser and click into the
`heart-disease-4site` experiment.

---

## Scene 4 — Packaging, testing, and CI (60s)

**Show:** Terminal running `make test`, then the GitHub Actions tab with
a green run.

**Say:** "The trained pipeline is packaged twice — once in MLflow, once
as a joblib artifact the API actually loads, with no MLflow server
needed at serving time. 68 tests cover data cleaning, preprocessing, the
model and registry code, and the API itself, at 93 percent coverage.
GitHub Actions runs lint, tests on both Python 3.12 and 3.13, a training
job, and a Docker build-and-smoke-test job on every push."

**Commands:**
```bash
make test
```

---

## Scene 5 — API and Docker (60s)

**Show:** Terminal running the container smoke test, then the Swagger UI
at `/docs`.

**Say:** "The FastAPI app validates every request against clinically
plausible ranges — an out-of-range value returns a 422, not a silent bad
prediction — and never logs raw patient feature values. The Docker image
runs as a non-root user with a health check, and this smoke test checks
every sample request against its expected response before calling it
done."

**Commands:**
```bash
make docker-smoke
```
Then open http://127.0.0.1:8000/docs (or port-forward the in-cluster
Service) and show the `/predict` schema with its examples.

---

## Scene 6 — Kubernetes deployment and monitoring (90s)

**Show:** Terminal running `kubectl get pods,svc,ingress,hpa -n
heart-mlops`, then a `/predict` call through the Ingress, then the
Grafana dashboard.

**Say:** "In Kubernetes, the API runs as two replicas behind both an
Ingress and a LoadBalancer Service, with a CPU-based autoscaler. A
request through the Ingress, using a Host header rather than editing
`/etc/hosts`, returns the same prediction. Prometheus discovers the pods
directly through the Kubernetes API, and Grafana's dashboard — built
entirely from a provisioned JSON file, no manual clicking — shows live
request and prediction traffic after a short load test."

**Commands:**
```bash
make k8s-verify
make load-test
```
Then port-forward Grafana and open http://127.0.0.1:3000.

---

## Scene 7 — Closing (20s)

**Show:** The architecture diagram (`reports/figures/architecture.png`
or `docs/ARCHITECTURE.md`).

**Say:** "That's the full pipeline, end to end — from raw UCI data
through to a monitored Kubernetes deployment. Every design decision
along the way, and why it was made, is logged in `docs/DECISIONS.md`."
