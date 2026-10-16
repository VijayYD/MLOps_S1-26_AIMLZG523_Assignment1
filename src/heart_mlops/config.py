"""Project-wide paths, seeds and dataset constants, used across download,
cleaning, training and serving so they are defined in exactly one place."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CLEANED_DATA_PATH = DATA_PROCESSED_DIR / "heart_combined_clean.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

SEED = 42

# Operating threshold is tuned (Phase 3) to meet at least this recall on the
# positive (disease) class (docs/DECISIONS.md D0.4).
RECALL_TARGET = 0.95

# The 14 standard UCI Heart Disease columns, in file order.
RAW_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

SITES = ("cleveland", "hungarian", "switzerland", "va")

UCI_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease"
UCI_FALLBACK_ZIP_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"

# SHA-256 of each processed.<site>.data file, captured from the URLs above
# and cross-checked against the fallback zip's copies (byte-identical).
# See data/README.md for how/when these were captured.
RAW_FILE_SHA256 = {
    "cleveland": "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8",
    "hungarian": "d1ad108f785768cd3d7e82dc522e6f5a61eea93cccfb3a46ee8076f73fc3d796",
    "switzerland": "834a405ccf5b66ab4056bb77794adc8df0b7125186454c0a1d002d33c6c3b314",
    "va": "e7c93d8d0d2acdadfa4c5e8de768e2191e7f618b952e29623f1f0d5949ff6b8f",
}

# --- Modelling feature lists (Phase 3) -------------------------------------
# See docs/DECISIONS.md D2.5: `site` is excluded as a model feature
# (grouping-only, for leave-one-site-out and per-site reporting); `ca`,
# `thal`, `slope` are kept despite being sparse outside Cleveland, with
# missing-indicator / "missing" one-hot handling.
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
MODEL_FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CV_FOLDS = 5
HOLDOUT_TEST_SIZE = 0.2

# --- Experiment tracking (Phase 4) -----------------------------------------
# Both gitignored; local-only by default. MLFLOW_TRACKING_URI overrides the
# backend store (e.g. to point at a remote server) without touching code.
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_ARTIFACT_ROOT = PROJECT_ROOT / "mlartifacts"
MLFLOW_TRACKING_URI_DEFAULT = f"sqlite:///{MLFLOW_DB_PATH}"
MLFLOW_EXPERIMENT_NAME = "heart-disease-4site"
MLFLOW_REGISTERED_MODEL_NAME = "heart_disease_classifier"
MLFLOW_CHAMPION_ALIAS = "champion"
