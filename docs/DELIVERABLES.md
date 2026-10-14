# Deliverables

| Required deliverable | Location in repo |
|---|---|
| Code, Dockerfile(s), requirements | [`src/`](../src/), [`docker/`](../docker/), [`requirements.txt`](../requirements.txt), [`requirements-dev.txt`](../requirements-dev.txt) |
| Cleaned dataset + download script/instructions | [`data/processed/heart_combined_clean.csv`](../data/processed/heart_combined_clean.csv), [`src/heart_mlops/data/download.py`](../src/heart_mlops/data/download.py), [`data/README.md`](../data/README.md) |
| Notebooks/scripts (EDA, training, inference) | [`notebooks/`](../notebooks/), [`scripts/`](../scripts/) |
| `test/` folder with unit tests | [`test/`](../test/) |
| GitHub Actions workflow YAML | [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) |
| Deployment manifests | [`k8s/`](../k8s/) |
| Screenshot folder | [`screenshots/`](../screenshots/) |
| Final written report (~10 pages, DOCX) | [`reports/MLOps_Assignment1_Report.docx`](../reports/MLOps_Assignment1_Report.docx) |
| Deployed API access instructions (local) | [`README.md`](../README.md), [`LAB_RUNBOOK.md`](../LAB_RUNBOOK.md) |
| Short video of the overall pipeline | [`docs/VIDEO_SCRIPT.md`](VIDEO_SCRIPT.md) (recording done by the project owner) |

## Supporting documentation

| Document | Covers |
|---|---|
| [`docs/DECISIONS.md`](DECISIONS.md) | Every non-obvious design decision, its alternatives, and why — one entry per phase |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | The end-to-end architecture diagram and a written walk-through |
| [`models/MODEL_CARD.md`](../models/MODEL_CARD.md) | Intended use, not-for-clinical-use disclaimer, data, metrics, per-site performance, limitations, ethical notes |
| [`screenshots/SCREENSHOT_CHECKLIST.md`](../screenshots/SCREENSHOT_CHECKLIST.md) | Numbered list of every screenshot, the command that produces it, and which machine |
| [`docs/VIDEO_SCRIPT.md`](VIDEO_SCRIPT.md) | Scene-by-scene walkthrough script for the recorded video |
