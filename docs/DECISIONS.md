# Design Decision Log

Each entry: the decision, the alternatives considered, and why. Ordered by phase.

## Phase 0 — Environment

### D0.1 — minikube pinned to the lab's exact version, not Homebrew's latest
Homebrew's current `minikube` formula installs v1.39.0; the Prayogshala lab has
v1.35.0 pre-installed and cannot be upgraded on demand. Installed the v1.35.0
binary directly from the GitHub releases page instead of `brew install minikube`,
so the Minikube/Kustomize/Docker-driver behaviour is identical on both machines.
**Alternative rejected:** Homebrew latest (1.39.0) + "document the drift" — rejected
because Minikube's addon and `kubectl` bundling behaviour has changed enough
across minor versions that "document and hope" is a weaker guarantee than just
matching the binary.

### D0.2 — Python 3.13.5 installed via pyenv, not Homebrew
The Mac's system `python3` (`/usr/bin/python3`) is 3.9.6 (Xcode Command Line
Tools' bundled Python), and pyenv only had 3.11.10 installed. Used
`pyenv install 3.13.5` rather than `brew install python@3.13`, since pyenv was
already present and keeps this project's interpreter isolated from the system
one and from other projects' pins.

### D0.3 — `scripts/setup_venv.sh` does not blindly trust `python3` on PATH
On this Mac, `/usr/bin` precedes `~/.pyenv/shims` in `$PATH`, and `pyenv init`
is not wired into the shell rc files, so a plain `python3` call resolves to the
system 3.9.6 interpreter even after installing 3.13.5 via pyenv — silently
creating a venv on the wrong Python version. `scripts/setup_venv.sh` therefore
resolves the interpreter itself: it uses PATH `python3` only if its major.minor
already matches the OS-specific pin (3.13 macOS / 3.12 Linux), otherwise it
looks for a matching pyenv-managed install, and only fails if neither exists.
An explicit `HEART_MLOPS_PYTHON` env var can override this.
**Why this matters for reproducibility:** the whole point of pinning exact
dependency versions is undermined if the interpreter itself is wrong; a
silent 3.9-vs-3.13 mismatch would be far harder to debug later than a clear
error now. This also protects the lab run: if the lab's `python3` on PATH
ever turns out not to be 3.12.9, the script will say so instead of silently
using whatever is there.

### D0.4 — Recall target for the clinical decision threshold: 0.95
Missed heart disease cases (false negatives) are costlier than false alarms
in a screening context, so the operating threshold (chosen later, in
Phase 3, from out-of-fold predicted probabilities) is tuned to achieve at
least 95% recall on the positive (disease) class.

## Phase 1 — Repository scaffold

### D1.1 — `mlflow==2.18.0` instead of `2.17.2`, with `pyarrow==18.1.0` pinned explicitly
`mlflow==2.17.2` constrains `pyarrow<18`, but PyPI's `pyarrow` only ships
`cp313` (Python 3.13) wheels starting at `18.0.0` — installing `mlflow==2.17.2`
on Python 3.13 forces pip to source-build an old `pyarrow`, which itself fails
(`ModuleNotFoundError: No module named 'pkg_resources'` building against a
`setuptools` that no longer vendors it). `mlflow==2.18.0` relaxed the
constraint to `pyarrow<19`, which includes `18.1.0` (has 3.13 wheels). Verified
by downloading mlflow's wheel METADATA directly from PyPI for 2.17.2 through
2.22.0 rather than guessing. Pinned `pyarrow==18.1.0` explicitly in
`requirements.txt` (rather than leaving it to mlflow's transitive range) so the
exact version is identical on Mac (3.13.5) and in the lab-parity container
(3.12.9), where a wheel is also available.

### D1.2 — `.python-version` is not committed to the repo
pyenv's per-directory `.python-version` file only names one version, but this
project intentionally targets two (3.13.x dev / 3.12.x lab). Committing a
single pinned version would make pyenv (if present in the lab) try to
auto-switch to an unavailable 3.13.x and fail confusingly. `scripts/check_env.sh`
and `scripts/setup_venv.sh` detect the right version per-OS instead, so no
committed dotfile is needed. It is listed in `.gitignore` so a developer's
local `pyenv local` doesn't get committed by accident.

### D1.3 — `requirements-dev.txt` is a superset of `requirements.txt` (`-r requirements.txt`)
`scripts/setup_venv.sh` always installs `requirements-dev.txt` on the dev
machine (pytest, ruff, jupyter, python-docx, etc., plus everything in
`requirements.txt`). The Docker image (`docker/Dockerfile`, Phase 7) installs
only `requirements.txt`, so dev/test/reporting tooling never ships in the
serving image.

### D1.4 — `scripts/setup_venv.sh` also installs the project itself (`pip install -e .`)
`src/heart_mlops` must be importable as `heart_mlops` (for
`python -m heart_mlops.data.download`, notebooks, and `pytest`'s
`pythonpath = ["src"]` config). An editable install is simpler and more
standard than manually exporting `PYTHONPATH` in every script.

## Phase 2 — Data acquisition and EDA

### D2.1 — UCI source URLs and fallback verified live, not assumed
Before writing `src/heart_mlops/data/download.py`, fetched all four primary
URLs and the dataset-id-45 fallback zip directly with curl: all returned
HTTP 200, the four files inside the fallback zip are byte-identical (`diff`)
to the four primary-URL downloads, and the combined row count is exactly 920
(303 + 294 + 123 + 200), confirming the brief's "~920 rows" figure. Full
detail and checksums in `data/README.md`.

### D2.2 — Implausible-zero columns: `chol` and `trestbps`, confirmed by direct inspection
Queried the raw data before deciding: `chol == 0` in **all 123 Switzerland
rows** and **49 of 200 VA rows**; `trestbps == 0` in 1 VA row. Both are
physiologically impossible in a living patient, so both are treated as
missing in `clean.py`. No other numeric column had suspicious zeros
(`thalach == 0`: none).

### D2.3 — Exact-duplicate rows dropped (2 pairs, 918 rows retained)
Found by `df.duplicated()` across all 14 raw columns + `site`: one duplicate
pair in Hungarian, one in VA, each pair identical in every field including
`oldpeak`'s decimal value -- consistent with the same record entered twice,
not coincidence. Dropped the second occurrence of each pair.

### D2.4 — Cleaned CSV keeps every NaN; no imputation at cleaning time
`clean.py` only type-coerces, flags implausible zeros, de-duplicates and
binarises the target -- it does not impute or drop any column. Imputation
happens inside the sklearn pipeline (Phase 3), fit on training folds only,
so validation/test rows (and, in leave-one-site-out, held-out sites) never
leak into imputed values.

### D2.5 — Confirmed: keep `ca`, `thal` and `slope` with missing-indicators; exclude `site` entirely
Evidence (non-missing rate by site): `ca` is 98.7% present in Cleveland but
only 1-4% present in the other three sites -- a missing-indicator on `ca`
mostly encodes "is this a Cleveland row" rather than real clinical signal,
i.e. a site-leakage risk. Decided to **keep `ca`** anyway (with a
missing-indicator flag) rather than drop it,
specifically so the leave-one-site-out evaluation (Phase 3) can show
empirically whether it actually hurts cross-site generalisation, instead of
assuming it will. `thal` (17-90% missing outside Cleveland) and `slope`
(35-65% missing outside Cleveland) are confirmed kept with missing-indicator
flags for the same reason, with their importance scores and the
leave-one-site-out results watched for leakage signs. `site` itself is
confirmed **excluded** as a model input entirely -- a model deployed on an
unseen hospital has no value to give it anyway, and including it risks
learning site-specific shortcuts rather than clinical relationships; it is
used only for leave-one-site-out grouping and per-site reporting.
**Status: decided; to be implemented in Phase 3's
`features/preprocess.py`. If `ca`'s feature importance or permutation
importance in Phase 3 turns out to be dominated by site rather than signal,
that will be reported explicitly rather than silently acted on.**

## Phase 3 — Feature engineering and model development

### D3.1 — Model-selection rule, fixed before any hold-out result is seen
Per the brief's requirement to write the rule down before looking at
hold-out results: **the model with the highest mean cross-validated
ROC-AUC wins** (each model's own best `RandomizedSearchCV` hyperparameters,
5-fold `StratifiedKFold`). Ties within 0.005 ROC-AUC are broken by
preferring the simpler model, in order logreg > random_forest > hist_gb.
The hold-out set is touched only *after* this rule has already picked a
winner, and plays no role in choosing between models -- it only reports how
the already-selected model performs on unseen data. Implemented verbatim
in `src/heart_mlops/models/train.py::select_best_model`, written before
`python -m heart_mlops.models.train` was ever run.

Tuning optimises ROC-AUC specifically (not F1 or accuracy) because it is
threshold-independent; the actual decision threshold is chosen separately
afterwards from out-of-fold probabilities to hit the 0.95 recall target
(D0.4), so optimising a threshold-dependent metric during tuning would
conflate two different decisions.

### D3.2 — Hold-out split stratified on (site, target) jointly, not target alone
Switzerland has only 8 negative-class (`target=0`) rows in the entire
dataset. Stratifying the train/hold-out split on `target` alone could, by
chance, place a disproportionate share of those 8 rows into either split.
Stratifying on the combined `site + "_" + target` key instead guarantees
every site's class balance is preserved in both the training and hold-out
sets. CV folds *within* training use `StratifiedKFold` on `target` alone
(not the joint key) -- deliberately simpler, since 5-fold CV on strata as
small as Switzerland's negatives (fewer than 8 rows once held out) risks a
fold with too few members of a stratum. The joint-key risk is for the single
hold-out split (acceptable, one split); the per-fold risk under 5-fold CV
is a much tighter constraint, which is what the separate leave-one-site-out
evaluation exists to probe properly instead.

### D3.3 — HistGradientBoostingClassifier receives no imputation at all
Unlike logreg/random_forest (median/constant-imputed via a
`ColumnTransformer`, see `src/heart_mlops/features/preprocess.py`),
`hist_gb`'s pipeline has no preprocessing step -- the raw DataFrame (NaNs
included) is passed straight to `HistGradientBoostingClassifier`, with
`categorical_features=CATEGORICAL_FEATURES` declared so it treats those
columns' integer codes as categories (and missingness as its own category)
rather than assuming a linear numeric relationship. This is the entire
reason it was chosen as the third model ("handles NaN natively" in the
brief) -- running it through the same imputer as the other two models would
defeat the point of including it as a genuinely different approach to the
missingness problem, not just a third algorithm.

### D3.4 — `joblib.parallel_backend("threading")` around tuning, not the process-based default
The first run of `python -m heart_mlops.models.train` completed correctly
(exit code 0, correct results) but printed ~16 harmless
`ChildProcessError: [Errno 10] No child processes` tracebacks from CPython's
`multiprocessing.resource_tracker` during interpreter shutdown -- a known
Python 3.13 + macOS + joblib/loky compatibility quirk, not a real failure.
Since the brief explicitly wants "clear logs" and this is also the script
CI's `train` job runs on every push (Phase 6), a wall of tracebacks at the
end of a successful run is exactly the kind of noise that could be
mis-read as a failure by a reviewer skimming Actions output. Switched to
joblib's `threading` backend for the tuning/search/permutation-importance
calls: it eliminates the tracebacks entirely (verified: identical
`best_score_`/`best_params_` results, confirming no behavioural change,
only how parallelism is implemented) at the cost of slower Random Forest
search specifically (~95s instead of ~16s for 30 candidates x 5 folds,
since nested joblib contexts downgrade `RandomForestClassifier(n_jobs=-1)`'s
own parallelism to threads too). Accepted this runtime cost -- total
training is still a few minutes -- in exchange for clean logs.

## Phase 4 — Experiment tracking

### D4.1 — `sqlalchemy==2.0.36` pinned explicitly (mlflow 2.18.0 incompatible with SQLAlchemy 2.1.x)
`requirements.txt` didn't pin `sqlalchemy` directly; pip resolved the newest
compatible-looking version, 2.1.4, which `mlflow==2.18.0`'s SQLite backend
store cannot actually use -- `python -m heart_mlops.models.train` failed
with `ImportError: cannot import name 'FallbackAsyncAdaptedQueuePool' from
'sqlalchemy.pool'` as soon as it tried to open the tracking store (the
tuning/selection logic itself ran correctly first; only the MLflow logging
step failed). `FallbackAsyncAdaptedQueuePool` doesn't exist in SQLAlchemy
2.1 -- mlflow 2.18.0 predates that release. Pinned `sqlalchemy==2.0.36`
(the newest 2.0.x release) explicitly; verified it resolves the import and
successfully runs mlflow's SQLite schema migrations end-to-end. Same
lesson as D1.1 (mlflow's transitive pins drift out of sync with its actual
compatibility range) -- checked by actually running the training script,
not by reading version ranges.

### D4.2 — Model Registry: `champion` alias, not the deprecated "Production" stage
MLflow's older model-stage system (`None`/`Staging`/`Production`/`Archived`)
is deprecated in favour of **aliases** -- arbitrary, mutable labels
(`client.set_registered_model_alias`) that can point at any model version.
Used a single alias, `champion`, pointing at whichever version the
selection rule most recently chose. Verified this is genuinely mutable by
running `python -m heart_mlops.models.train` twice: the second run created
version 2 and moved `champion` from v1 to v2 automatically (confirmed via
`MlflowClient.get_model_version_by_alias`), with v1 left in the registry,
unaliased, for audit/rollback rather than being overwritten or deleted.
Serving code (Phase 5+) should resolve the model by alias
(`models:/heart_disease_classifier@champion`), never by a hardcoded
version number, so promoting a new champion never requires a code change.

### D4.3 — Model card drafted in Phase 4, not held until Phase 5
The brief lists "the model card" among the artefacts this phase's MLflow
run should log, but `models/MODEL_CARD.md` as a deliverable file is
formally a Phase 5 task. Resolved the ordering conflict by drafting a real
(not placeholder) model card now -- using the actual Phase 3 metrics,
per-site leave-one-site-out results, and limitations -- logged as an
MLflow artifact on the parent run. Phase 5 finalises/extends it (adds the
packaged joblib artifact's version info, sklearn version, timestamp) once
that artefact exists, rather than writing it from scratch twice.

## Phase 5 — Model packaging and reproducibility

### D5.1 — The API loads the joblib + metadata artifact, not the MLflow model/registry, at serving time
`models/heart_model.joblib` + `models/model_metadata.json` are the only
things the Docker image (Phase 7) needs to serve predictions. The
alternative -- loading `models:/heart_disease_classifier@champion` from
the MLflow Model Registry at container startup -- would make the serving
container depend on an MLflow tracking server (or a shared SQLite file +
`mlartifacts/` directory) being reachable, which is an extra moving part
and an extra failure mode with zero benefit at serving time: the model is
already fully trained and selected by the time it needs to serve traffic.
Training (`python -m heart_mlops.models.train`) still logs everything to
MLflow (Phase 4) for tracking/comparison/audit, and *also* writes the
joblib + metadata pair as a direct byproduct of the same run -- so there
is exactly one training path, with MLflow as the system of record for
experiment history and the joblib artifact as the self-contained thing
that actually ships.

### D5.2 — The packaged joblib artifact is the exact same fitted pipeline registered in MLflow, not refit on the full dataset
Considered refitting the selected model on the full 918-row dataset
(train + hold-out combined) before packaging, to use all available data
in the deployed model -- standard practice in some MLOps setups. Decided
against it: doing so would make the MLflow-registered model and the
joblib artifact two *different* trained models (one on 734 rows, one on
918), contradicting the brief's framing of these as the same pipeline in
two serialisation forms. It would also make `model_metadata.json`'s
hold-out metrics describe a model that is not the one actually being
shipped -- the 184 hold-out rows would no longer be held out from the
artefact being evaluated. `models/heart_model.joblib` is therefore
`search_results[selected_name].best_estimator_` (fit on the 734-row
training split) -- byte-for-byte the same object passed to
`mlflow.sklearn.log_model`, saved a second time in a different format.

### D5.3 — Reproducibility test retrains with fixed hyperparameters, not a full re-tune
Re-running the entire `RandomizedSearchCV` search twice (3 models x 30
candidates x 5 folds, each) to prove reproducibility would cost ~12
minutes per test run -- unworkable for a test that runs on every CI push
(Phase 6). Since `RandomizedSearchCV` and the pipelines themselves are
already seeded deterministically, the property actually worth testing is
narrower: *given fixed hyperparameters and a fixed seed, does fitting the
same pipeline twice on the same data produce the same predictions?*
`test/test_reproducibility.py` reads the already-selected model's name
and tuned hyperparameters straight out of `models/model_metadata.json`,
fits two fresh instances, and asserts their predicted probabilities match
to floating-point tolerance -- a fast, meaningful check of the actual
shipped model's determinism, not a re-run of the (already-tested-by-having-
run-once) tuning search.

## Phase 6 — CI/CD and automated testing

### D6.1 — The API is built now, not held until Phase 7
The brief's Phase 6 test list explicitly requires FastAPI `TestClient`
tests (`/health`, `/ready`, `/predict`, validation errors, `/metrics`),
but `src/heart_mlops/api/` is formally a Phase 7 ("Containerisation...")
deliverable -- the same kind of ordering conflict as the model card (D4.3).
Resolved the same way: built a real, functional API now
(`main.py`/`schemas.py`/`metrics.py`/`logging_config.py`) so Phase 6 can
test it properly, with Phase 7 adding the Docker/Kubernetes/Prometheus
operational wrapping around what already works, not building the API
itself from scratch at that point.

### D6.2 — All 13 request fields are required; none are nullable
The trained pipeline can handle missing values internally (that's the
entire point of the imputation work in Phase 3), but a client-facing API
that accepts partial clinical records adds real complexity -- which
fields may be omitted together, how `Literal` categorical fields interact
with `None`, what "missing at inference time" should even mean for a
single synchronous prediction request -- for a feature this project does
not otherwise need. All 13 `PatientFeatures` fields are required, with
evidence-based bounds taken from the actual observed min/max in
`data/processed/heart_combined_clean.csv` (not guessed) plus headroom for
clinically plausible values outside this particular 918-row sample --
e.g. `oldpeak`'s lower bound is -3.0, not a naive >=0, because the real
data's minimum is -2.6.

### D6.3 — Found and fixed a real dtype bug between training and serving
Manually smoke-testing `/predict` with a realistic JSON payload (ints for
`sex`/`cp`/etc., as any real client would send) raised a 500 error:
`ValueError: The dtype of the filling value (float64) cannot be cast to
... dtype('int64')`. Root cause: at training time, `ColumnTransformer`
extracts the categorical columns into one array; pandas upcasts a mix of
int64 (`sex`, `cp` -- never missing) and float64 (the rest -- have NaN) to
a common float64 dtype, so the fitted `SimpleImputer` expects float64.
Pydantic's `Literal[int]` fields give plain Python `int`s, which pandas
stores as int64 when *every* categorical value in a request happens to be
an int (the normal case) -- a dtype mismatch sklearn correctly refuses.
Fixed by explicitly casting the assembled request row to `float64` before
calling `predict_proba` in `main.py`. Caught by actually calling the
endpoint with realistic input, not just unit-testing each piece in
isolation -- worth remembering for Phase 7's container smoke test too.

### D6.4 — No raw patient features in logs
`api/logging_config.py::log_request` logs request id, path, status code,
latency, predicted class and model version -- never the request body.
Patient-level clinical data is sensitive, retained in log files far
longer than the request itself persists, and nothing about serving or
debugging predictions requires it to be there. Verified with a test that
asserts distinctive feature values from a real request never appear in
captured log output.

### D6.5 — Coverage target: 90%
The brief asked for a proposed-and-confirmed target; proposed 80%
initially, then raised it to 90% instead. Met with unit tests across
data/preprocessing/model/API, plus one fast
end-to-end integration test (`test_main_end_to_end_with_tiny_data_and_grids`)
that runs the real training + MLflow-logging + packaging code path against
a tiny synthetic fixture and tiny hyperparameter grids/fold counts --
seconds instead of the ~6 minutes the full search takes on the full
dataset -- rather than mocking that path into something less trustworthy.
Actual result: 93% (`pytest`'s `--cov-fail-under=90` enforces the floor in
CI going forward).

### D6.6 — The CI `docker` job is deferred to Phase 7
The brief's Phase 6 CI description includes a `docker` job (build image,
run container, curl `/predict`). `docker/Dockerfile` does not exist yet --
it is a Phase 7 deliverable. Rather than add a job guaranteed to fail
until Phase 7 lands (which would make `main` red for a reason that has
nothing to do with a real regression), `.github/workflows/ci.yml` ships
now with `lint`, `test` (matrix: Python 3.12.9 and 3.13.5, mirroring the
Mac and lab exactly), and `train` jobs only. Phase 7 extends this same
file with the `docker` job once there is something real for it to build
and test.

## Phase 7 — Containerisation, Kubernetes, monitoring

### D7.1 — POST /predict/batch implemented
The brief gated this behind confirming scope first. Shares the single
`/predict` endpoint's row-building and response-shaping logic
(`_features_to_row`, `_risk_to_response` helpers in `main.py`) rather than
duplicating it, and runs one vectorised `predict_proba` call across the
whole batch instead of looping model calls one row at a time. Capped at
100 records per request (`BatchPredictRequest.records`, `max_length=100`)
so a single request can't become an unbounded inference job.

### D7.2 — HPA on CPU implemented
The brief gated this behind confirming scope first. `k8s/base/hpa.yaml`
targets the `heart-api` Deployment, 2-5 replicas, 70% average CPU
utilisation, requiring the `metrics-server` addon (enabled by
`scripts/k8s_deploy.sh`).

### D7.3 — ConfigMap's threshold/log-level values are wired into the app, not decorative
The brief asks for a ConfigMap carrying "threshold, log level" -- rather
than add env vars the app ignores, `main.py` actually reads `LOG_LEVEL`
(passed to `configure_logging`) and an optional `PREDICTION_THRESHOLD`
that overrides the packaged model's threshold at startup when set (the
packaged `model_metadata.json` value remains the default). Verified both
with a test that sets `PREDICTION_THRESHOLD` and checks `/model-info`
reflects it. This lets ops adjust the operating threshold via
`kubectl edit configmap` + a rollout restart, without retraining or
rebuilding the image.

### D7.4 — Kustomize's security sandbox blocks sharing Grafana config files across `k8s/` and `monitoring/`
Tried to have `k8s/monitoring/kustomization.yaml` read the *same*
provisioning files docker-compose mounts directly
(`monitoring/grafana/provisioning/...`, `monitoring/grafana/dashboards/heart_api.json`)
via a `configMapGenerator`'s `files:` entries with a `../../monitoring/...`
relative path -- one source of truth for both. Kustomize's `kustomize
build`/`kubectl apply -k` refused: its security sandbox restricts file
references to inside or below the kustomization's own root directory,
and a relative path that escapes it (even one that stays inside the
repo) is rejected outright, with no in-file way to disable this (only an
external `--load-restrictor` CLI flag, which `kubectl apply -k` doesn't
expose and which scripts would have to remember to pass every time).
Resolved by inlining the same (small, stable) content directly as plain
`ConfigMap` manifests under `k8s/monitoring/` instead -- accepting minor
duplication between `monitoring/grafana/provisioning/*` (used by
docker-compose) and `k8s/monitoring/grafana-*-configmap.yaml` (used by
`kubectl apply -k`) in exchange for `kubectl apply -k` actually working
without extra flags. Each k8s file's header comment points back at its
compose counterpart so a future edit isn't made in only one place by
accident.

### D7.5 — Prometheus scrape config differs between docker-compose and Kubernetes (not shared)
Unlike the Grafana provisioning files, `monitoring/prometheus.yml`
(docker-compose: a static target, `api:8000`) and
`k8s/monitoring/prometheus-configmap.yaml` (Kubernetes:
`kubernetes_sd_configs` pod discovery, since pod IPs are dynamic and a
static target can't work) are genuinely different mechanisms, not just
differently-formatted versions of the same config -- there is nothing to
deduplicate here, unlike D7.4's Grafana files.

### D7.6 — Verified end-to-end before moving on, at every layer
Built and ran the actual Docker image (not just reviewed the Dockerfile):
confirmed it runs as the non-root `appuser`, the `HEALTHCHECK` reports
healthy, and OCI labels carry the real git SHA. Ran `scripts/run_container_smoke.sh`
for real against all four `test/sample_requests/*.json` files, each
checked against its expected status code (200 for `valid_*`, 422 for
`invalid_*`) -- caught a real bash bug while writing it: variable
assignments inside a `{ ... } | tee` pipeline run in a subshell and never
reach the parent shell, so the smoke test's pass/fail signal was silently
lost; fixed by having the subshell `exit` with the right code and reading
it back via `${PIPESTATUS[0]}` (with an `|| true` guard, since `set -e` +
`pipefail` would otherwise abort the script before that line ever ran).
Also ran the full docker-compose stack for real and confirmed, via its
HTTP APIs (not just "containers are up"): Prometheus's target for the API
reports `health: "up"`, a `predictions_total` query returns real data
after a real `/predict` call and one scrape interval, and Grafana's
`/api/search` shows the "Heart Disease API" dashboard auto-provisioned
with zero manual clicking.

### D7.7 — macOS Ingress verification uses `kubectl port-forward`, not `minikube tunnel`
Tried `minikube tunnel` first, as the brief suggests. Its log said
plainly: `The service/ingress heart-api-ingress requires privileged
ports to be exposed: [80 443]. sudo permission will be asked for it` --
and from a non-interactive script (no controlling TTY), `sudo` fails
immediately rather than prompting, so the Ingress tunnel silently never
came up (confirmed: nothing was listening on port 80 after `minikube
tunnel` reported "successfully started"; what briefly appeared to answer
on port 80 during debugging was this Mac's own unrelated local Apache
instance, not minikube). The LoadBalancer side of `minikube tunnel`
worked fine without sudo (its port, 8000, isn't privileged) -- only the
Ingress's fixed 80/443 is the problem, and that's a property of Ingress
itself, not something choosing a different app port avoids. Also, the
project's lab explicitly disallows `sudo` entirely, so a script that only
works via an interactive sudo prompt isn't portable to begin with.
`scripts/k8s_verify.sh` now uses `kubectl port-forward` to arbitrary
unprivileged local ports for both the Ingress controller and the
LoadBalancer Service on macOS -- verified end-to-end: `/health` and
`/predict` both return correctly through each path. Linux is unaffected
(`minikube ip` is directly reachable there, no tunnel or port-forward
needed for either path).

### D7.8 — CI's `docker` job added now that `docker/Dockerfile` exists (resolves D6.6)
D6.6 deferred this job to avoid a guaranteed-red CI step while the
Dockerfile didn't exist yet. It exists now; `.github/workflows/ci.yml`'s
`docker` job runs `scripts/run_container_smoke.sh` directly (no
reimplemented build/run/curl logic -- the same script a developer runs
locally), gated on `test` rather than `train`: the image is built from
whatever `models/heart_model.joblib` is already committed in the repo
(the thing a fresh `git clone` actually gets), not from `train`'s
freshly-generated-but-never-committed artifact, so a slow or flaky UCI
download in `train` can never block the container smoke test. GHCR push
was deferred at this point, pending confirming scope first, same as
D7.1/D7.2.

### D7.9 — CI pushes the smoke-tested image to GHCR, main-branch pushes only
Added two steps to the end of the `docker` job: log in to `ghcr.io` with
`docker/login-action`, then build and push `ghcr.io/<owner>/heart-mlops-api`
tagged both `:<git-sha>` and `:latest`. Gated on
`github.event_name == 'push' && github.ref == 'refs/heads/main'` so pull
request runs never attempt a push (a fork's PR run has no write access to
packages, and a throwaway demo branch like D9.3's `ci-fail-demo` should
never publish an image anyway). Placed *after* the smoke-test step rather
than as a parallel build, so only an image that has already passed
`scripts/run_container_smoke.sh` is ever published -- a failed smoke test
fails the job before the push steps run. Uses the workflow's own
`GITHUB_TOKEN` (scoped via the job-level `permissions: packages: write`)
rather than a separately managed PAT, since `GITHUB_TOKEN` already has
everything GHCR push needs and rotates automatically with each run.
Re-tags and re-pushes the already-built image rather than calling
`docker/build-push-action`, so the exact same `scripts/build_image.sh`
path (and its OCI labels, D5.1-adjacent) is used for both the local smoke
test and the published image -- one build script, not two slightly
different build paths that could drift apart.
**Note:** a package pushed to GHCR from a personal repository is private
by default; visit the package's own settings on GitHub
(`github.com/<owner>?tab=packages`) to make it public if it needs to be
pullable without authentication.

### D7.10 — docker-compose host ports made overridable (`API_PORT`/`PROMETHEUS_PORT`/`GRAFANA_PORT`)
Hit this directly running `make compose-up` on a shared machine:
`Error response from daemon: ... failed to bind host port for
0.0.0.0:8000:...: address already in use`. `scripts/run_container_smoke.sh`
already supported a `PORT` override for the standalone smoke-test
container, but `docker/docker-compose.yml` hard-coded `8000:8000`,
`9090:9090` and `3000:3000` on the host side with no way to override
any of them -- the one command in this project without a port escape
hatch, on exactly the kind of shared machine where a port collision is
expected (another student's leftover container, or a previous run of
this project's own stack never torn down). Changed all three services'
host-side port mappings to `"${API_PORT:-8000}:8000"` (and the
equivalent for Prometheus/Grafana) -- the container-internal ports are
unchanged, so nothing else in the stack (the Grafana datasource pointing
at `prometheus:9090` internally, the Dockerfile's internal `API_PORT`
env var) needed to change. Verified with `docker compose -f
docker/docker-compose.yml config` that the override substitutes
correctly and the unset-case default (`8000`) is preserved exactly as
before.

### D7.11 — Fixed a real bug: `k8s_verify.sh` assumed the LoadBalancer Service was reachable at `minikube ip:8000` on Linux
Hit this directly running `k8s_verify.sh` in the lab: `/health` and
`/predict` through the Ingress both succeeded at `minikube ip`, but
`/health` through the LoadBalancer failed with `curl: (7) Failed to
connect ... Connection refused` on the same IP, port 8000.
Root cause: `k8s/base/service-lb.yaml`'s own header comment claimed the
Service was "usable ... directly via `minikube ip` (Linux)", and
`k8s_verify.sh`'s Linux branch was written to match that claim -- but
it's wrong. A plain `type: LoadBalancer` Service only gets a real
external IP (`status.loadBalancer.ingress[0].ip`) once `minikube tunnel`
is actually running; that's a property of how LoadBalancer Services
work in Kubernetes, not something minikube's Linux/Docker-driver
combination special-cases the way it does for the `ingress` addon
(which *is* genuinely reachable at `minikube ip` directly on Linux,
unrelated to this bug). This specific code path was never actually
exercised during D8.2's dry run, because that ran on macOS, where the
script always port-forwards *both* Ingress and LoadBalancer regardless
-- so the untested Linux-only assumption sat in the repo, documented as
if verified, until the real lab run hit it.
Fixed `k8s_verify.sh` to treat the LoadBalancer check the same way on
every OS: check for an assigned external IP first, and automatically
fall back to `kubectl port-forward` if there isn't one (`minikube
tunnel` not running), rather than ever assuming the node IP works
directly. No `sudo` needed on either path, since the LoadBalancer's
port (8000) isn't privileged (D7.7's sudo concern was specific to
Ingress's 80/443). Corrected the misleading comment in
`k8s/base/service-lb.yaml` and the same stale claim in `README.md` and
`LAB_RUNBOOK.md` to match.

### D7.12 — `${KUBECTL:-kubectl}` in the standalone monitoring port-forward commands never actually worked without `kubectl` installed
Hit this directly in the lab: `${KUBECTL:-kubectl} -n heart-mlops
port-forward svc/prometheus 9090:9090`, copy-pasted straight out of
`SETUP_GUIDE.md`/`LAB_RUNBOOK.md`, failed with `kubectl: command not
found`. The `${KUBECTL:-kubectl}` fallback syntax only resolves to
something real when the `KUBECTL` variable has actually been set --
which happens inside `k8s_deploy.sh`/`k8s_verify.sh`/`k8s_teardown.sh`
because they `source scripts/_k8s_lib.sh`, but never happens for these
two commands, which were written as bare one-off terminal commands, not
run through any script. The placeholder syntax looked like a working
fallback but was dead code outside the three scripts that already
sourced the lib. Fixed both docs to `source scripts/_k8s_lib.sh` first
(which sets `KUBECTL` to `minikube kubectl --` when `kubectl` isn't on
PATH) before using `${KUBECTL}`, and added this to both troubleshooting
sections' existing "`kubectl: command not found`" entry.

### D7.13 — Documented that changing `GRAFANA_ADMIN_PASSWORD` in `.env` requires both a redeploy and a pod restart
Hit this directly in the lab: changed `GRAFANA_ADMIN_PASSWORD` in `.env`
and Grafana still rejected it with "invalid username or password".
`.env` is only ever read by `scripts/k8s_deploy.sh`, which writes it
into the `grafana-admin` Secret -- editing the file has no effect on an
already-running cluster by itself. Separately, even after the Secret is
updated, Kubernetes does not restart a pod just because a Secret it
references changed, so the already-running Grafana container keeps
using whatever `GF_SECURITY_ADMIN_PASSWORD` value it started with.
Confirmed `k8s/monitoring/grafana-deployment.yaml` has no
`PersistentVolume` for Grafana's own data (only the read-only
provisioning ConfigMaps are mounted), so a pod restart fully
reinitializes Grafana rather than keeping a stale admin user from a
previous boot -- meaning `scripts/k8s_deploy.sh` (re-applies the
Secret) followed by `kubectl -n heart-mlops rollout restart
deployment/grafana` is sufficient, no manual Grafana admin-reset
command needed. Added this exact sequence to both runbooks'
troubleshooting sections, since the previous docs only described the
first-time setup, not what changing the password later requires.

### D7.14 — `load_test.sh`'s default `BASE_URL` silently failed against the Kubernetes deployment; added a preflight check
Hit this directly in the lab: ran `make load-test` after a Kubernetes
deploy, it printed "Done." with no visible error, and Grafana stayed
completely empty. Root cause: `scripts/load_test.sh` defaults
`BASE_URL` to `http://127.0.0.1:8000`, which is correct for
docker-compose (D7.6 verified real Grafana data that way) and a local
`uvicorn` run, but nothing listens on that host port for a Kubernetes
deployment unless the API (or its LoadBalancer Service) is separately
port-forwarded there first -- a step neither runbook's documented
`k8s-deploy` -> `k8s-verify` -> `load-test` sequence included.
Every request failed to connect; the script's `curl` calls aren't run
with `-f`, so a connection failure just produces an http_code of `000`
that gets silently tallied by `sort | uniq -c` like any other status
code, and the script still exits 0 and prints "Done." -- no real
traffic ever reached the API, so Prometheus had nothing new to scrape
and Grafana's panels stayed empty. This exact gap was never caught
earlier because D8.2's dry run verified the Kubernetes Ingress/
LoadBalancer paths and D7.6 verified Grafana against docker-compose,
but no prior check ran `load_test.sh` against the Kubernetes deployment
specifically.
Fixed two ways: (1) added a preflight `curl -fsS .../health` check to
`load_test.sh` that fails loudly with exit 1 and a clear port-forward
hint instead of silently sending doomed requests -- verified both
failure (unreachable URL, real non-zero exit) and success (a real local
`uvicorn` instance, 20/20 real `200`s) paths by actually running the
script, not just reading it; (2) updated both runbooks' "Generate
traffic and view monitoring" sections to port-forward
`svc/heart-api-lb` to `8000:8000` before calling `make load-test` for
the Kubernetes case.

### D7.15 — Pinned the provisioned Grafana datasource's `uid` explicitly, since the dashboard JSON hard-codes it
Hit this directly in the lab: real traffic confirmed reaching the API
(live `200`s from `curl`), Prometheus confirmed scraping both `heart-api`
pods (`2/2 up`) and confirmed to actually hold the metric
(`http_requests_total` returned real rows, `sum(rate(...))` nonzero when
queried directly in Prometheus's own `/graph` page) -- yet Grafana's
dashboard panels still rendered a flat line. Root cause: every panel in
`monitoring/grafana/dashboards/heart_api.json` references its datasource
as `{"type": "prometheus", "uid": "Prometheus"}`, but
`monitoring/grafana/provisioning/datasources/datasource.yaml` (and its
Kubernetes copy, `k8s/monitoring/grafana-datasources-configmap.yaml`)
never set an explicit `uid:` field -- so Grafana auto-generated a random
one (confirmed via `GET /api/datasources`: the real uid was
`PBFA97CFB590B2093`, not `"Prometheus"`). Every panel was therefore
referencing a datasource UID that didn't exist. This likely rendered
correctly only intermittently because the datasource is also marked
`isDefault: true`, and some Grafana code paths silently fall back to the
default datasource when a referenced UID can't be resolved -- which
explains why one earlier burst test happened to show a real spike while
a later, otherwise-identical test showed nothing: never a reliable
signal either way, just an implementation-detail fallback that isn't
guaranteed.
Fixed by adding `uid: Prometheus` explicitly to both provisioning files,
so the datasource is always created with the exact UID every panel
already expects -- removing any dependency on undocumented
default-datasource fallback behavior. Because Grafana has no persistent
volume here (D7.13), the fix takes effect on the next pod restart with
no manual re-pointing of panels needed: `kubectl -n heart-mlops rollout
restart deployment/grafana` re-provisions the datasource fresh with the
pinned UID.

## Phase 8 — Lab-parity verification

### D8.1 — `git archive HEAD`, not a file copy, for the lab-parity container
`scripts/lab_parity_check.sh` needs an exact copy of what the lab's
`git clone` would actually get. `git archive HEAD` gives exactly the
tracked, committed tree -- no `.venv`, no `data/raw/` downloads, no
`mlruns`/`mlartifacts`, no stray local files -- without needing a
hand-maintained exclude list that could drift out of sync with
`.gitignore`. The archive is mounted read-only into the container and
copied again to a writable path inside before anything installs, so the
read-only mount is never written to.

### D8.2 — Fresh-clone dry run found zero gaps in `LAB_RUNBOOK.md`
Per the brief: cloned the repo fresh (from the local `.git` history, since
nothing had been pushed to GitHub yet at this point -- the first push was
deliberately held back until the rest of the pipeline was verified) and
followed
`LAB_RUNBOOK.md` literally, step by step: `check_env.sh`, `setup_venv.sh`,
`make data`, `make test`, `make train` (full real run, correct model
selected, correct hold-out metrics), `make docker-smoke` (with a `PORT`
override, to prove that override path works too), and `make k8s-deploy` +
`k8s_verify.sh` against the already-running Minikube cluster (idempotent
re-apply -- every resource came back `unchanged`, rollout healthy,
`/health` and `/predict` both correct through Ingress and LoadBalancer).
Every command worked exactly as written; no runbook edits were needed as
a result. (One container-only observation, not a bug: `python:3.12.9-slim`
has no `git` binary, so MLflow's own internal GitPython-based detection
warns and continues -- this project's own `get_git_sha()` already expects
exactly this and falls back to `"unknown"`, by design, D4.1-adjacent.)

## Phase 9 — Documentation and reporting

### D9.1 — `model_comparison.csv` / `leave_one_site_out.csv` are a transcription of the notebook's own output, not a re-computation
`reports/build_report.py` needs the Phase 3 comparison and
leave-one-site-out tables as data files, not hand-typed numbers in the
report script -- but re-running the full ~6 minute hyperparameter search
and leave-one-site-out evaluation just to regenerate two small already-
computed tables (whose values are already fixed and committed in
`notebooks/02_modelling.ipynb`'s executed output) would be wasteful.
`reports/figures/model_comparison.csv` and `leave_one_site_out.csv` are a
direct, verified transcription of that notebook's own stored output
cells (read back from the `.ipynb` JSON and checked value-for-value, not
retyped from memory) -- not an independent re-derivation. If Phase 3 is
ever re-run with different results, these two files need regenerating
from the new notebook output alongside it.

### D9.2 — Rendered the DOCX to images and actually looked at it, rather than trusting python-docx's API to have done the right thing
Installed LibreOffice (worth the ~1GB download for this) specifically to
render `reports/MLOps_Assignment1_Report.docx` to a real PDF and inspect
each page as an image -- not just checking that
`python-docx` ran without raising. This caught two real layout problems
invisible from the generation code alone: a 16-column data-quality table
that would have overflowed the page width (cut down to the 8 columns
that actually matter, matching `data/README.md`'s own curated summary),
and several near-square figures (ROC/PR/calibration/confusion-matrix,
each originally sized to the same 6-inch width as every other figure)
each consuming most of a page with a large blank gap below, inflating
the report from an initial 21 pages to 18 and then, after narrowing just
those four square figures specifically, to 16 -- closer to the brief's
~10-page guidance without cutting any actual content. Removed the
one-off `pymupdf` dependency from the venv afterward since it was a
diagnostic tool for this check, not a project dependency.

### D9.3 — CI verified on GitHub itself, including the deliberate-failure case, not just locally
Pushed `main` for the first time and watched the real run: all five jobs
(`lint`, `test` x2, `train`, `docker`) passed on the first push. Then, per
the Definition of Done, proved CI
fails correctly: a throwaway branch (`ci-fail-demo`) with one assertion in
`test/test_data.py::test_target_binarisation` changed to an intentionally
wrong expected value, opened as a PR to trigger the `pull_request`
workflow trigger. Result matched the workflow's own `needs:` design
exactly: `lint` still passed (the breakage is a logic error, not a style
one), both `test` jobs failed on the broken assertion, and `train`/`docker`
correctly *skipped* rather than running, since both depend on `test`
succeeding first. Closed the PR without merging and deleted the branch
(confirmed gone on GitHub via the API, and pruned locally); `main`'s own
test suite re-run afterward, unaffected, still 68 passed at 93% coverage.

### D9.4 — Added a standalone, OS-agnostic SETUP_GUIDE.md; redrew the architecture diagram as a strict grid
Added `SETUP_GUIDE.md` as a second, self-contained walkthrough alongside
`LAB_RUNBOOK.md`, written so it never names a specific machine or lab --
every command, prerequisite, and troubleshooting note is phrased
generically (e.g. "a Python 3.12.x/3.13.x interpreter" rather than "3.13.x
on macOS / 3.12.x on Linux") so it stands on its own for someone cloning
the repository on an unknown machine, with no assumed context from this
project's own development environment. `LAB_RUNBOOK.md` is kept as-is for
its original audience; the two files serve different readers rather than
one replacing the other.

Separately, redrew `reports/build_architecture_diagram.py`: the original
layout placed boxes in a zigzag with diagonal `FancyArrowPatch` connectors,
which visually crossed through or near unrelated boxes and made the data
flow hard to follow at a glance. Replaced it with a strict grid where every
arrow is constrained to be purely horizontal (same row) or purely vertical
(same column) -- a boustrophedon (snake) row layout -- so no arrow can ever
cross another box by construction. Regenerated
`reports/MLOps_Assignment1_Report.docx` afterward so the corrected diagram
is embedded in the report, not just the standalone PNG.

### D9.5 — Added `notebooks/03_inference_demo.ipynb` to close a real deliverables gap
Re-read the actual assignment brief against the finished repo and found a
genuine gap: the brief's Deliverables section names three notebook/script
categories explicitly -- "EDA, training, inference" -- but only the first
two existed. `03_inference_demo.ipynb` loads the packaged
`heart_model.joblib` + `model_metadata.json` directly (no API server), and
mirrors `api/main.py`'s `_features_to_row`/`_risk_to_response` logic
exactly, so it is a faithful standalone demonstration of the same serving
path rather than a second, divergent implementation of it. Covers
single-record inference (using the same `test/sample_requests/*.json`
files the container smoke test sends) and batch inference against 8 real
sampled patients with ground truth, run for real via `jupyter nbconvert
--execute` (7/8 correct on this sample -- not cherry-picked to look
perfect). Generated via `scripts/_build_inference_notebook.py` using
`nbformat` rather than hand-written JSON, so the notebook can be
regenerated if ever needed.

### D9.6 — Trimmed the report from 20 pages toward the brief's ~10-page guidance, without touching the hand-embedded video
The same re-read against the brief flagged a second gap: the report had
grown to 20 pages (LibreOffice-rendered) against the brief's stated "10
pages" once all 11 real screenshots were embedded. Re-ran the actual
figure-display-width levers in `build_report.py` (4.6in down to 2.1-4.3in
depending on section), tightened `Normal` style spacing and page margins,
shortened prose across every section while keeping every decision ID and
number, and lowered `build_architecture_diagram.py`'s save DPI from 200 to
130 (the diagram displays at a few inches wide; 200 DPI there was pure
excess weight, not legibility). Also discovered and fixed a real
misunderstanding along the way: `add_picture(width=Inches(...))` only sets
a *display*-size attribute in the DOCX XML -- it embeds the source image's
full original bytes regardless of that width, so shrinking `width_in`
alone reduced page count but did nothing for file size. Fixed by actually
downsampling `reports/figures/*.png` (the notebook-generated charts) to
900px wide via Pillow -- the same fix already applied to the manual
screenshots (D-series size-reduction work) is now applied to every
embedded image, not just half of them. Got to 20 -> 12 LibreOffice-rendered
pages this way.

Went further on request: added `add_figure_grid()`, which lays a list of
images out in an n-column borderless table instead of stacking them one
per row -- same content, a fraction of the vertical space, since Word has
no other built-in way to flow images side by side. Applied it everywhere
two or more images appear together: the 6 EDA figures and the 6 modelling
figures each collapsed from a near-full-page vertical stack to a 3x2 grid
occupying roughly a third of a page, and every multi-screenshot section
(experiment tracking, CI/CD, containerisation, monitoring) got the same
treatment at 2 or 3 columns depending on count. A missing screenshot still
renders as an in-cell placeholder rather than silently collapsing the
grid's shape. Result: 20 -> 9 LibreOffice-rendered pages (expect roughly
+1 once Word's TOC field actually resolves its 13+ entries, since
LibreOffice only shows the field's placeholder text, never rendering the
real TOC content) -- landing right at the brief's ~10-page guidance.

The video embed and both hyperlinks (GitHub, Google Drive backup) were
added by hand in real Microsoft Word in a prior session -- python-docx has
no API for constructing the OLE Compound-File-Binary "Package" object Word
uses to embed an arbitrary file as a clickable icon, and hand-rolling that
binary format from scratch was assessed as too fragile to attempt safely
(no access to real Word here to verify it actually opens, only
LibreOffice). Rebuilding the report fresh from `build_report.py` to trim
it would have destroyed that hand-made embed entirely. Resolved with
`scripts/_embed_video.py`: extracts the exact `<w:object>` XML, the OLE
binary part, and the icon image out of the last known-good embedded build,
and transplants them byte-for-byte into any freshly rebuilt report in
place of a `VIDEO_OLE_MARKER` placeholder paragraph -- relocating an
already-working embed rather than constructing a new one. Verified by
actually opening the result: LibreOffice converts it to PDF without error,
and the rendered cover page shows the real video thumbnail, title, and
"(7:36) -- double-click to play" caption exactly as the original
hand-made version did. Final size: 9.3MB (video ~5.3MB of that), under the
assignment's 10MB single-file submission cap.
