"""Packaged model artifact (joblib + metadata) -- save/load -- and the
MLflow Model Registry lookup used by anything that wants the current
`champion`.

The API (Phase 7) loads the joblib + metadata artifact directly, **not**
the MLflow model / registry, so it never needs an MLflow server or backend
store reachable at serving time -- see docs/DECISIONS.md D5.1.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import sklearn

from heart_mlops.config import (
    CATEGORICAL_FEATURES,
    MLFLOW_CHAMPION_ALIAS,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI_DEFAULT,
    MODELS_DIR,
    NUMERIC_FEATURES,
    SEED,
)

MODEL_PATH = MODELS_DIR / "heart_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"


def build_metadata(
    *,
    model_name: str,
    model_version: int | str,
    threshold: float,
    hyperparameters: dict[str, Any],
    metrics: dict[str, float],
    data_checksum: str,
    git_sha: str,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "model_version": model_version,
        "threshold": threshold,
        "hyperparameters": hyperparameters,
        "feature_schema": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
        },
        "metrics": metrics,
        "training_data_checksum": data_checksum,
        "sklearn_version": sklearn.__version__,
        "git_sha": git_sha,
        "seed": SEED,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def save_packaged_model(pipeline, metadata: dict[str, Any]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str) + "\n")


def load_packaged_model(
    model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH
) -> tuple[Any, dict[str, Any]]:
    pipeline = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text())
    return pipeline, metadata


def get_champion_model_uri() -> str:
    """URI for the current MLflow Registry champion -- resolved by alias,
    never a hardcoded version number, so promoting a new champion never
    needs a code change (docs/DECISIONS.md D4.2)."""
    return f"models:/{MLFLOW_REGISTERED_MODEL_NAME}@{MLFLOW_CHAMPION_ALIAS}"


def get_champion_version() -> int:
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI_DEFAULT)
    from mlflow.tracking import MlflowClient

    version_info = MlflowClient().get_model_version_by_alias(
        MLFLOW_REGISTERED_MODEL_NAME, MLFLOW_CHAMPION_ALIAS
    )
    return int(version_info.version)
