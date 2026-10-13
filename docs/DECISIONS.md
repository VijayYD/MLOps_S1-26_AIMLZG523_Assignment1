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

### D2.5 — Proposed: drop `ca` as a model feature; keep `thal`/`slope` with missing-indicators; exclude `site` entirely
Evidence (non-missing rate by site): `ca` is 98.7% present in Cleveland but
only 1-4% present in the other three sites -- a missing-indicator on `ca`
would mostly encode "is this a Cleveland row" rather than real clinical
signal, i.e. site leakage. `thal` (17-90% missing outside Cleveland) and
`slope` (35-65% missing outside Cleveland) have enough real coverage in 2-3
sites to be worth keeping, with missing-indicator flags, while watching their
importance scores and the leave-one-site-out results for leakage signs.
`site` itself is proposed to be excluded as a model input entirely -- a
model deployed on an unseen hospital has no value to give it anyway, and
including it risks learning site-specific shortcuts rather than clinical
relationships; it is used only for leave-one-site-out grouping and per-site
reporting. **Status: proposed at the Phase 2 checkpoint, pending
confirmation before being implemented in Phase 3's
`features/preprocess.py`.**
