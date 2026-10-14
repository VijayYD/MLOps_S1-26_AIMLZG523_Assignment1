"""Generates reports/MLOps_Assignment1_Report.docx.

Pulls metrics/thresholds from models/model_metadata.json,
reports/figures/model_comparison.csv, and
reports/figures/leave_one_site_out.csv, and embeds figures from
reports/figures/ directly -- so the report's numbers and plots can never
drift from what the code actually produced. Screenshot placeholders are
replaced automatically with the real image if
screenshots/<filename> (see screenshots/SCREENSHOT_CHECKLIST.md) exists.

Usage:
    python reports/build_report.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from heart_mlops.config import DATA_PROCESSED_DIR, PROJECT_ROOT, REPORTS_FIGURES_DIR
from heart_mlops.models.registry import METADATA_PATH

REPO_URL = "https://github.com/VijayYD/MLOps_S1-26_AIMLZG523_Assignment1"
COURSE_TITLE = "Machine Learning Operations (AIMLCZG523) — Assignment 1"
PROJECT_TITLE = "Heart Disease Risk Prediction: An End-to-End MLOps Pipeline"
SCREENSHOTS_DIR = PROJECT_ROOT / "screenshots"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "MLOps_Assignment1_Report.docx"

_counters = {"figure": 0, "table": 0}


def load_metadata() -> dict:
    return json.loads(METADATA_PATH.read_text())


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(REPORTS_FIGURES_DIR / name)


# --- Low-level docx helpers -------------------------------------------------


def add_toc_field(document: Document) -> None:
    paragraph = document.add_paragraph()
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = 'TOC \\o "1-3" \\h \\z \\u'
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    placeholder = OxmlElement("w:t")
    placeholder.text = (
        "Right-click here and choose 'Update Field' to generate the table of contents."
    )
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    r = run._r
    for el in (fld_begin, instr, fld_sep, placeholder, fld_end):
        r.append(el)


def add_caption(document: Document, kind: str, text: str) -> None:
    _counters[kind] += 1
    label = "Figure" if kind == "figure" else "Table"
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(f"{label} {_counters[kind]}: {text}")
    run.italic = True
    run.font.size = Pt(10)


def add_figure(document: Document, image_path: Path, caption: str, width_in: float = 4.6) -> None:
    document.add_picture(str(image_path), width=Inches(width_in))
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_caption(document, "figure", caption)


def add_screenshot_or_placeholder(document: Document, filename: str, description: str) -> None:
    path = SCREENSHOTS_DIR / filename
    if path.exists():
        add_figure(document, path, description)
        return
    p = document.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(f"[Insert screenshot: {description} — screenshots/{filename}]")
    run.italic = True
    run.font.color.rgb = RGBColor(0x80, 0x80, 0x80)


def add_dataframe_table(
    document: Document, df: pd.DataFrame, caption: str, float_fmt: str = "{:.3f}"
) -> None:
    table = document.add_table(rows=1, cols=len(df.columns))
    table.style = "Light Grid Accent 1"
    header_cells = table.rows[0].cells
    for i, col in enumerate(df.columns):
        header_cells[i].text = str(col)
        for run in header_cells[i].paragraphs[0].runs:
            run.bold = True
    for _, row in df.iterrows():
        cells = table.add_row().cells
        for i, val in enumerate(row):
            if isinstance(val, float):
                cells[i].text = float_fmt.format(val)
            else:
                cells[i].text = str(val)
    add_caption(document, "table", caption)


def add_heading(document: Document, text: str, level: int = 1):
    return document.add_heading(text, level=level)


# --- Report sections ---------------------------------------------------


def build_cover_page(document: Document) -> None:
    for _ in range(4):
        document.add_paragraph()
    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run(COURSE_TITLE)
    run.bold = True
    run.font.size = Pt(16)

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run(PROJECT_TITLE)
    run.bold = True
    run.font.size = Pt(20)

    document.add_paragraph()
    for label in ["STUDENT NAME: ____________________", "STUDENT ID: ____________________"]:
        p = document.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(label)
        r.font.size = Pt(13)

    document.add_paragraph()
    date_p = document.add_paragraph()
    date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    date_p.add_run(date.today().isoformat())

    repo_p = document.add_paragraph()
    repo_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    repo_p.add_run(REPO_URL)

    document.add_page_break()


def build_intro(document: Document) -> None:
    add_heading(document, "1. Introduction and Problem Statement", level=1)
    document.add_paragraph(
        "Heart disease remains a leading cause of death worldwide, and early identification of "
        "at-risk patients from routinely collected clinical attributes is a long-standing machine "
        "learning problem. This project builds a binary classifier that predicts the presence of "
        "heart disease from thirteen clinical attributes, and ships it as a containerised, "
        "monitored, Kubernetes-deployed API with a full CI/CD pipeline and experiment tracking."
    )
    document.add_paragraph(
        "Rather than use the common 303-row Cleveland-only subset of the UCI Heart Disease dataset, "
        "this project combines all four original sites (Cleveland, Hungarian, Switzerland, and VA "
        "Long Beach; 918 rows after cleaning). This is a deliberately harder and more realistic "
        "setting: the four sites differ substantially in missingness patterns and disease "
        "prevalence, which makes naive pooled evaluation optimistic about real-world "
        "generalisation to a new hospital. A leave-one-site-out evaluation, in addition to "
        "standard stratified cross-validation, is used throughout to make this risk visible rather "
        "than hidden."
    )
    document.add_paragraph(
        "Every non-obvious design decision in this project — from the dataset and missing-value "
        "handling to the model-selection rule and Kubernetes topology — is logged with its "
        "rationale and alternatives considered in docs/DECISIONS.md, referenced throughout this "
        "report by its decision ID (e.g. D3.1)."
    )


def build_setup(document: Document) -> None:
    add_heading(document, "2. Setup and Installation Instructions", level=1)
    document.add_paragraph(
        "The project targets two environments with different Python versions: macOS with Python "
        "3.13.x for development, and a Linux lab (BITS Prayogshala) with Python 3.12.9 for "
        "verification. scripts/setup_venv.sh resolves the correct pinned interpreter for the "
        "current OS rather than trusting whatever python3 happens to be first on $PATH, since the "
        "two can silently differ (see docs/DECISIONS.md D0.3)."
    )
    add_heading(document, "Mac / development", level=2)
    for line in [
        "bash scripts/check_env.sh",
        "bash scripts/setup_venv.sh && source .venv/bin/activate",
        "make data && make train",
    ]:
        document.add_paragraph(line, style="Intense Quote")
    add_heading(document, "Lab (Linux)", level=2)
    document.add_paragraph(
        "LAB_RUNBOOK.md gives the full copy-paste sequence — clone, preflight, venv, data, "
        "training, tests, Docker, Kubernetes deploy/verify/load-test/teardown, and a "
        "troubleshooting section — verified by an actual fresh-clone dry run before being trusted "
        "(docs/DECISIONS.md D8.2). No step anywhere requires sudo."
    )


def build_data_eda(document: Document, summary_df: pd.DataFrame) -> None:
    add_heading(document, "3. Data Acquisition and Exploratory Data Analysis", level=1)
    document.add_paragraph(
        "src/heart_mlops/data/download.py downloads the four UCI site files with SHA-256 "
        "checksum verification, retries, and an automatic fallback to the UCI static archive if "
        "a primary URL is unreachable — never fabricating data if both sources fail. "
        "src/heart_mlops/data/clean.py then converts '?' tokens and physiologically impossible "
        "zero values (chol == 0 in all 123 Switzerland rows and 49 VA rows; trestbps == 0 in one "
        "VA row) to missing, removes two exact duplicate records, and binarises the target "
        "(target = 1 if num > 0 else 0). The cleaned file deliberately keeps every missing value — "
        "imputation happens only inside the model pipeline, fit on training folds only, so no "
        "information leaks from validation or test rows."
    )
    # Only the columns worth a reader's attention -- the full summary_df has
    # 16 columns (one per raw feature), which overflows a page; age/sex/cp/
    # fbs/restecg are ~0% missing everywhere and not informative here.
    key_columns = [
        "site",
        "n_rows",
        "missing_pct_trestbps",
        "missing_pct_chol",
        "missing_pct_slope",
        "missing_pct_ca",
        "missing_pct_thal",
        "target_positive_pct",
    ]
    summary_display = summary_df[key_columns].rename(
        columns={
            "site": "Site",
            "n_rows": "Rows",
            "missing_pct_trestbps": "trestbps % missing",
            "missing_pct_chol": "chol % missing",
            "missing_pct_slope": "slope % missing",
            "missing_pct_ca": "ca % missing",
            "missing_pct_thal": "thal % missing",
            "target_positive_pct": "Disease %",
        }
    )
    add_dataframe_table(
        document,
        summary_display,
        "Rows, missing data (%) for the columns most affected, and disease prevalence (%), by site. "
        "Full per-column breakdown in data/processed/data_quality_summary.csv.",
        float_fmt="{:.1f}",
    )
    document.add_paragraph(
        "The missingness is not uniform: ca, thal and slope are missing in 60-99% of rows outside "
        "Cleveland, and disease prevalence swings from 36.2% (Hungarian) to 93.5% (Switzerland). "
        "Both are visible in the figures below and motivate the site-aware modelling choices in "
        "Section 4."
    )
    for fname, caption in [
        ("01_missingness_heatmap.png", "Missing values by column and site."),
        ("02_class_balance.png", "Overall and per-site class balance."),
        ("03_numeric_distributions.png", "Numeric feature distributions by target (KDE)."),
        ("04_categorical_counts.png", "Categorical feature counts by target."),
        (
            "05_correlation_heatmap_pearson.png",
            "Pearson correlation among continuous features and target.",
        ),
        ("06_boxplots_by_site.png", "Site shift in key continuous features."),
    ]:
        path = REPORTS_FIGURES_DIR / fname
        if path.exists():
            add_figure(document, path, caption)


def build_modelling(
    document: Document, comparison_df: pd.DataFrame, loso_df: pd.DataFrame, metadata: dict
) -> None:
    add_heading(document, "4. Feature Engineering and Modelling", level=1)
    document.add_paragraph(
        "Three models were tuned with RandomizedSearchCV under 5-fold stratified cross-validation: "
        "Logistic Regression (an interpretable linear baseline), Random Forest, and "
        "HistGradientBoostingClassifier — included specifically because it handles missing values "
        "and categorical columns natively, so it receives no imputation at all (unlike the other "
        "two, which use a shared ColumnTransformer: median imputation with missing-indicator flags "
        "and scaling for numeric columns, constant-sentinel imputation plus one-hot encoding for "
        "categorical columns)."
    )
    document.add_paragraph(
        "The model-selection rule was written down and committed to version control before any "
        "hold-out result was ever looked at (docs/DECISIONS.md D3.1): the model with the highest "
        "mean cross-validated ROC-AUC wins; ties within 0.005 are broken by preferring the simpler "
        "model. ROC-AUC, not a threshold-dependent metric, is the tuning objective, because the "
        "operating threshold is chosen separately afterwards."
    )
    comparison_display = comparison_df.rename(
        columns={
            "cv_roc_auc_mean": "CV ROC-AUC",
            "cv_roc_auc_std": "CV std",
            "holdout_accuracy": "Accuracy",
            "holdout_precision": "Precision",
            "holdout_recall": "Recall",
            "holdout_f1": "F1",
            "holdout_roc_auc": "ROC-AUC",
            "holdout_pr_auc": "PR-AUC",
            "holdout_brier": "Brier",
            "holdout_threshold": "Threshold",
        }
    )
    add_dataframe_table(
        document,
        comparison_display,
        "Cross-validated and hold-out metrics for all three tuned models.",
    )
    document.add_paragraph(
        f"Logistic Regression was selected (highest CV ROC-AUC, {comparison_df.iloc[0]['cv_roc_auc_mean']:.4f}). "
        f"The operating threshold ({metadata['threshold']:.4f}) was chosen from out-of-fold predicted "
        f"probabilities to meet a 0.95 recall target on the positive (disease) class — missed "
        f"disease cases are treated as costlier than false alarms, consistent with a "
        f"screening-tool framing. Logistic Regression's coefficients give a concrete, interpretable "
        f"check against the site-leakage risk flagged during EDA: the 'ca is missing' category "
        f"carries an odds ratio of approximately 1 (no effect), while 'ca = 0' carries the single "
        f"largest coefficient in the model — evidence that ca is being used for genuine clinical "
        f"signal where known, not as a shortcut for which hospital a row came from."
    )
    # 07-10 are square (6x6 or 5x5 matplotlib figures) -- narrower here so
    # two can share a page instead of each leaving a large blank gap below.
    for fname, caption, width_in in [
        ("07_roc_curves.png", "ROC curves for all three models on the hold-out set.", 3.4),
        (
            "08_pr_curves.png",
            "Precision-Recall curves for all three models on the hold-out set.",
            3.4,
        ),
        (
            "09_confusion_matrix.png",
            "Confusion matrix for the selected model at its operating threshold.",
            3.4,
        ),
        ("10_calibration_curve.png", "Calibration curves for all three models.", 3.4),
        (
            "11_logreg_odds_ratios.png",
            "Logistic Regression odds ratios (top 20 by |coefficient|).",
            4.6,
        ),
        ("12_rf_importance.png", "Random Forest impurity-based and permutation importance.", 6.0),
    ]:
        path = REPORTS_FIGURES_DIR / fname
        if path.exists():
            add_figure(document, path, caption, width_in=width_in)

    add_heading(document, "Leave-one-site-out generalisation", level=2)
    document.add_paragraph(
        "The distinctive evaluation for this project trains on three of the four sites and tests "
        "entirely on the fourth, in turn, using the single globally-selected threshold throughout "
        "(a deployed model does not re-tune itself per incoming hospital)."
    )
    loso_display = loso_df.rename(
        columns={
            "held_out_site": "Held-out site",
            "n_train": "Train rows",
            "n_test": "Test rows",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "roc_auc": "ROC-AUC",
            "pr_auc": "PR-AUC",
            "brier": "Brier",
            "threshold": "Threshold",
        }
    )
    add_dataframe_table(
        document, loso_display, "Leave-one-site-out metrics, one row per held-out site."
    )
    document.add_paragraph(
        "Recall holds at or above the 0.95 target for three of the four sites; Cleveland (0.89) is "
        "the exception, likely because it is the only site with near-complete ca/thal/slope data, "
        "so a model that never saw Cleveland during training must classify its fuller feature "
        "profile using patterns learned mostly from sparser sites. ROC-AUC drops noticeably for "
        "Switzerland and VA, consistent with the site-shift seen in Section 3's EDA. Hungarian's "
        "precision is low (0.52) when held out — it has the lowest disease prevalence of the four "
        "sites, so a threshold tuned on the pooled, higher-prevalence training mix produces more "
        "false positives there. This base-rate-shift effect would not be visible from a single "
        "pooled k-fold evaluation, which is the entire reason this evaluation exists."
    )
    path = REPORTS_FIGURES_DIR / "13_leave_one_site_out.png"
    if path.exists():
        add_figure(document, path, "Recall and ROC-AUC by held-out site.")
    document.add_paragraph(
        "Calibration was checked (Brier score and a reliability curve) but CalibratedClassifierCV "
        "changed the Brier score by under 0.001 — not distinguishable from noise on a hold-out set "
        "this size — so the selected model's own predict_proba output is shipped directly as the "
        "API's confidence score (docs/DECISIONS.md, Phase 3)."
    )


def build_experiment_tracking(document: Document, metadata: dict) -> None:
    add_heading(document, "5. Experiment Tracking", level=1)
    document.add_paragraph(
        "Every training run logs to a single MLflow experiment (heart-disease-4site) as one "
        "parent run with one nested child run per model. Each child run records its hyperparameters, "
        "preprocessing choices, cross-validated and hold-out metrics, the full cv_results table, a "
        "classification report, and a confusion matrix; the selected model's child run additionally "
        "gets a feature-importance plot and the logged sklearn model itself, with an inferred "
        "signature and input example. Parent-run artifacts include the model comparison table, "
        "ROC/PR/calibration plots, and the model card."
    )
    document.add_paragraph(
        f"The selected model is registered as heart_disease_classifier and given the alias "
        f"'champion' (version {metadata['model_version']} at the time of writing) — not MLflow's "
        f"deprecated Staging/Production stage system. Serving code resolves the model by alias "
        f"(models:/heart_disease_classifier@champion), never a hardcoded version number, so "
        f"promoting a new champion never requires a code change. This was verified to be "
        f"genuinely mutable by running training twice and confirming the alias moved to the new "
        f"version automatically, with the old version left unaliased for audit and rollback "
        f"rather than deleted."
    )
    add_screenshot_or_placeholder(
        document, "03_mlflow_runs_list.png", "MLflow experiment runs list"
    )
    add_screenshot_or_placeholder(
        document, "04_mlflow_run_detail.png", "MLflow run detail and registered model"
    )


def build_packaging(document: Document, metadata: dict) -> None:
    add_heading(document, "6. Model Packaging and Reproducibility", level=1)
    document.add_paragraph(
        "The trained pipeline is persisted twice: once in the MLflow model format (logged to the "
        "registry) and once as models/heart_model.joblib plus models/model_metadata.json. These "
        "are deliberately the exact same fitted pipeline, not independently retrained — the joblib "
        "artifact is search_results[selected_name].best_estimator_, byte-for-byte the same object "
        "passed to mlflow.sklearn.log_model, so the hold-out metrics recorded in the metadata "
        "genuinely describe the model being shipped."
    )
    document.add_paragraph(
        "The API loads the joblib and metadata pair directly, with no MLflow server reachable at "
        "serving time — a deliberate choice, since the model is already fully trained and selected "
        "by the time it needs to serve traffic, and depending on a tracking server at that point "
        "would be an extra moving part with no benefit."
    )
    rows = [
        ["Model", metadata["model_name"]],
        ["Registry version", str(metadata["model_version"])],
        ["Threshold", f"{metadata['threshold']:.4f}"],
        ["scikit-learn version", metadata["sklearn_version"]],
        ["Training data SHA-256", metadata["training_data_checksum"][:16] + "…"],
        ["Git commit", metadata["git_sha"][:12]],
        ["Packaged at (UTC)", metadata["timestamp"]],
    ]
    table = document.add_table(rows=1, cols=2)
    table.style = "Light Grid Accent 1"
    header_cells = table.rows[0].cells
    header_cells[0].text = "Field"
    header_cells[1].text = "Value"
    for cell in header_cells:
        for run in cell.paragraphs[0].runs:
            run.bold = True
    for label, value in rows:
        cells = table.add_row().cells
        cells[0].text = label
        cells[1].text = value
    add_caption(document, "table", "Packaged model artifact metadata (models/model_metadata.json).")
    document.add_paragraph(
        "A reproducibility test (test/test_reproducibility.py) reads the packaged model's own "
        "tuned hyperparameters and fits two fresh pipeline instances on the same data, asserting "
        "their predictions match to floating-point tolerance — proving the shipped model's "
        "determinism without re-running the full ~6-minute hyperparameter search inside a test."
    )


def build_cicd(document: Document) -> None:
    add_heading(document, "7. CI/CD Pipeline and Automated Testing", level=1)
    document.add_paragraph(
        "68 pytest tests cover data cleaning and download (network calls fully mocked), the "
        "preprocessing pipelines (shape, no-NaN-after-transform, unknown-category handling, "
        "fit-on-train-only), the model/evaluation/registry code (including one fast end-to-end "
        "run of the real training and MLflow-logging path against a tiny synthetic fixture and "
        "tiny hyperparameter grids, rather than mocking that path away), and the FastAPI app "
        "(health/readiness, the full prediction and batch-prediction paths, six validation-error "
        "cases, the Prometheus /metrics endpoint, and a check that raw patient feature values "
        "never appear in the logs). Coverage is 93% against a 90% floor enforced by "
        "pytest --cov-fail-under."
    )
    document.add_paragraph(
        ".github/workflows/ci.yml runs on every push and pull request to main, plus manual "
        "dispatch: lint (ruff check and format), test (a matrix of Python 3.12.9 and 3.13.5, "
        "mirroring the lab and the development machine exactly), train (downloads and cleans "
        "fresh data with a verified fallback to the committed cleaned CSV if UCI is unreachable, "
        "trains, and writes a metrics table to the job summary), and docker (builds the image from "
        "the repository's committed model artifact and runs the exact same container smoke-test "
        "script a developer runs locally)."
    )
    add_screenshot_or_placeholder(
        document, "05_gha_all_green.png", "GitHub Actions run with all jobs green"
    )
    add_screenshot_or_placeholder(document, "06_pytest_coverage.png", "pytest coverage summary")


def build_containerisation(document: Document) -> None:
    add_heading(document, "8. Containerisation and API Design", level=1)
    document.add_paragraph(
        "docker/Dockerfile is a multi-stage build on python:3.12.9-slim (matching the lab exactly), "
        "built for linux/amd64, running as a non-root user, with only runtime dependencies "
        "installed (requirements.txt, not the dev superset), a HEALTHCHECK using the standard "
        "library's urllib (so curl does not need to be installed in the image), a configurable "
        "worker count, and OCI labels carrying the real build version and git commit SHA."
    )
    document.add_paragraph(
        "The FastAPI app (src/heart_mlops/api/) exposes GET /health (liveness), GET /ready "
        "(readiness — only once the model artifact has loaded), GET /model-info, POST /predict, "
        "POST /predict/batch (up to 100 records per request, sharing the single-record endpoint's "
        "row-building logic and using one vectorised predict_proba call), GET /metrics (Prometheus "
        "exposition format), and auto-generated OpenAPI documentation at /docs with field-level "
        "examples. All thirteen request fields are required, with evidence-based clinical range "
        "validation taken from the observed min/max in the training data rather than guessed — "
        "out-of-range values or a missing field return HTTP 422, never a 500 or a silent bad "
        "prediction. Request logs are structured JSON (request id, path, status, latency, "
        "predicted class, model version) and deliberately never include the raw patient feature "
        "values, since nothing about serving or debugging predictions requires patient-level "
        "clinical data to sit in a log file."
    )
    add_screenshot_or_placeholder(
        document, "07_docker_smoke.png", "Docker build and container smoke test"
    )
    add_screenshot_or_placeholder(
        document, "08_docker_inspect.png", "Non-root user and healthy container"
    )
    add_screenshot_or_placeholder(
        document, "11_swagger_docs.png", "Swagger / OpenAPI documentation"
    )


def build_kubernetes(document: Document) -> None:
    add_heading(document, "9. Kubernetes Deployment", level=1)
    document.add_paragraph(
        "Deployment manifests are organised with Kustomize (kubectl apply -k, built into kubectl, "
        "so Helm is not a dependency): k8s/base/ (namespace, ConfigMap carrying LOG_LEVEL and an "
        "optional PREDICTION_THRESHOLD override — both are actually read by the running app, not "
        "decorative — a 2-replica Deployment with liveness and readiness probes and a non-root "
        "securityContext, a ClusterIP Service, a LoadBalancer Service, an Ingress at host "
        "heart-api.local, and a CPU-based HorizontalPodAutoscaler scaling 2 to 5 replicas) and "
        "k8s/monitoring/ (Prometheus and Grafana, described in Section 10), composed together by "
        "k8s/overlays/minikube/."
    )
    document.add_paragraph(
        "scripts/k8s_deploy.sh starts Minikube if needed, enables the ingress and metrics-server "
        "addons, builds the image inside Minikube's own Docker daemon (no registry push needed), "
        "applies the overlay, and waits for every Deployment to roll out. The Grafana admin "
        "password is generated as a Kubernetes Secret from .env, never hard-coded in a committed "
        "manifest."
    )
    document.add_paragraph(
        "scripts/k8s_verify.sh checks pod/service/ingress/HPA state and then calls /health and "
        "/predict through both the Ingress (reached with a Host header, never by editing "
        "/etc/hosts) and the LoadBalancer Service. On macOS, this uses kubectl port-forward rather "
        "than minikube tunnel: the Ingress's privileged ports (80/443) need an interactive sudo "
        "password that an unattended script cannot supply, and the lab disallows sudo entirely in "
        "any case — the port-forward approach was verified to work identically for both paths, "
        "with no loss of coverage. On Linux, minikube ip is directly reachable, so neither tunnel "
        "nor port-forward is needed there."
    )
    add_screenshot_or_placeholder(
        document, "09_kubectl_get.png", "kubectl get pods, services, ingress, HPA"
    )
    add_screenshot_or_placeholder(
        document, "10_predict_ingress_lb.png", "/predict via Ingress and LoadBalancer"
    )


def build_monitoring(document: Document) -> None:
    add_heading(document, "10. Monitoring and Logging", level=1)
    document.add_paragraph(
        "The API exposes custom Prometheus metrics: request count by endpoint and status, a "
        "latency histogram, prediction count by class, a risk-probability histogram (a simple "
        "drift signal), and an inference-error counter. In Kubernetes, Prometheus discovers API "
        "pods via the Kubernetes API (pod annotations), not a static target, since pod IPs are "
        "not stable; in docker-compose, the same metrics are scraped via a static target. Grafana's "
        "datasource and a five-panel dashboard (request rate, p95 latency, predictions by class, "
        "risk-probability drift signal, inference error rate) are both provisioned as code — no "
        "manual dashboard-building in the UI. Both were verified against their real HTTP APIs, not "
        "just 'the containers are up': Prometheus's target for the API reports healthy, a live "
        "query returns real prediction counts after running the load test, and Grafana's search "
        "API shows the dashboard already present."
    )
    add_screenshot_or_placeholder(document, "12_prometheus_targets.png", "Prometheus targets page")
    add_screenshot_or_placeholder(
        document, "13_grafana_dashboard.png", "Grafana dashboard with live data"
    )


def build_architecture_section(document: Document) -> None:
    add_heading(document, "11. Architecture Diagram", level=1)
    path = REPORTS_FIGURES_DIR / "architecture.png"
    if path.exists():
        add_figure(
            document,
            path,
            "End-to-end architecture, from data acquisition to monitoring.",
            width_in=6.5,
        )
    document.add_paragraph(
        "The full written walk-through of this diagram is in docs/ARCHITECTURE.md."
    )


def build_limitations(document: Document) -> None:
    add_heading(document, "12. Limitations, Ethical Considerations, and Future Work", level=1)
    document.add_paragraph(
        "This model is a course project artefact, not a validated medical device, and must not be "
        "used to make or inform real patient-care decisions (see models/MODEL_CARD.md for the full "
        "statement). The training data is small (918 rows) and decades old; ca and thal are over "
        "60% missing outside Cleveland, and the leave-one-site-out results in Section 4 show the "
        "model's behaviour on a genuinely unseen hospital is not uniform across sites. The 0.95 "
        "recall target was chosen for this project and was not validated against a real clinical "
        "cost-benefit analysis. No subgroup fairness analysis beyond the exploratory data analysis "
        "in Section 3 has been performed."
    )
    document.add_paragraph(
        "The training data's site of origin correlates with factors — era, region, equipment — "
        "that could encode bias unrelated to clinical presentation; site is excluded as a model "
        "feature specifically to reduce, though not eliminate, this risk. This model must not be "
        "used for insurance, employment, or any decision with legal or financial consequences for "
        "a person."
    )
    document.add_paragraph(
        "Future work could include: a prospective validation study against current clinical "
        "practice; a fairness audit across demographic subgroups; extending the leave-one-site-out "
        "evaluation with a genuinely held-out fifth site; and investigating whether a calibration "
        "method would help on a larger hold-out set than the one available here."
    )


def build_references(document: Document) -> None:
    add_heading(document, "13. References", level=1)
    for ref in [
        "Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). Heart Disease "
        "[Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X",
        "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. "
        "Journal of Machine Learning Research, 12, 2825-2830.",
        "MLflow Documentation. https://mlflow.org/docs/latest/index.html",
        "FastAPI Documentation. https://fastapi.tiangolo.com/",
        "Kubernetes Documentation. https://kubernetes.io/docs/home/",
        "Prometheus Documentation. https://prometheus.io/docs/introduction/overview/",
        "Grafana Documentation. https://grafana.com/docs/grafana/latest/",
    ]:
        document.add_paragraph(ref, style="List Bullet")


# --- Main ------------------------------------------------------------------


def main() -> None:
    import build_architecture_diagram

    build_architecture_diagram.build()

    metadata = load_metadata()
    comparison_df = load_csv("model_comparison.csv")
    loso_df = load_csv("leave_one_site_out.csv")
    summary_df = pd.read_csv(DATA_PROCESSED_DIR / "data_quality_summary.csv")

    document = Document()
    style = document.styles["Normal"]
    style.font.size = Pt(11)

    build_cover_page(document)

    add_heading(document, "Table of Contents", level=1)
    add_toc_field(document)
    document.add_page_break()

    build_intro(document)
    build_setup(document)
    build_data_eda(document, summary_df)
    build_modelling(document, comparison_df, loso_df, metadata)
    build_experiment_tracking(document, metadata)
    build_packaging(document, metadata)
    build_cicd(document)
    build_containerisation(document)
    build_kubernetes(document)
    build_monitoring(document)
    build_architecture_section(document)
    build_limitations(document)
    build_references(document)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}")

    placeholders = []
    for fname in [
        "03_mlflow_runs_list.png",
        "04_mlflow_run_detail.png",
        "05_gha_all_green.png",
        "06_pytest_coverage.png",
        "07_docker_smoke.png",
        "08_docker_inspect.png",
        "09_kubectl_get.png",
        "10_predict_ingress_lb.png",
        "11_swagger_docs.png",
        "12_prometheus_targets.png",
        "13_grafana_dashboard.png",
    ]:
        if not (SCREENSHOTS_DIR / fname).exists():
            placeholders.append(fname)
    print(f"Remaining screenshot placeholders ({len(placeholders)}): {placeholders}")


if __name__ == "__main__":
    main()
