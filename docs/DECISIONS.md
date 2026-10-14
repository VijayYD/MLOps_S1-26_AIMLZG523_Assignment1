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
Confirmed with the project owner. Missed heart disease cases (false negatives)
are costlier than false alarms in a screening context, so the operating
threshold (chosen later, in Phase 3, from out-of-fold predicted probabilities)
is tuned to achieve at least 95% recall on the positive (disease) class.

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
i.e. a site-leakage risk. This was flagged and the project owner chose to
**keep `ca`** anyway (with a missing-indicator flag) rather than drop it,
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
**Status: confirmed with the project owner; to be implemented in Phase 3's
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
