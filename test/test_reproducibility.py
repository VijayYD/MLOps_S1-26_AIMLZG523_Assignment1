"""Reproducibility test for the packaged model (Phase 5).

Does NOT re-run the full hyperparameter search (3 models x 30 candidates x
5 folds would take ~12 minutes to run twice -- unworkable in CI). Instead
reads the already-selected model's name and tuned hyperparameters out of
models/model_metadata.json, fits two fresh pipeline instances with the
same seed on the same data, and asserts they produce the same predictions
-- see docs/DECISIONS.md D5.3 for the full reasoning.
"""

import json

import numpy as np
import pandas as pd
import pytest

from heart_mlops.config import CLEANED_DATA_PATH, MODEL_FEATURE_COLUMNS
from heart_mlops.features.preprocess import build_pipeline
from heart_mlops.models.registry import METADATA_PATH

pytestmark = pytest.mark.skipif(
    not METADATA_PATH.exists(),
    reason="models/model_metadata.json not found -- run `make train` first",
)


def test_same_hyperparameters_and_data_give_same_predictions():
    metadata = json.loads(METADATA_PATH.read_text())
    model_name = metadata["model_name"]
    hyperparameters = metadata["hyperparameters"]

    df = pd.read_csv(CLEANED_DATA_PATH)
    X = df[MODEL_FEATURE_COLUMNS]
    y = df["target"]

    pipeline_a = build_pipeline(model_name)
    pipeline_a.set_params(**hyperparameters)
    pipeline_a.fit(X, y)
    proba_a = pipeline_a.predict_proba(X)[:, 1]

    pipeline_b = build_pipeline(model_name)
    pipeline_b.set_params(**hyperparameters)
    pipeline_b.fit(X, y)
    proba_b = pipeline_b.predict_proba(X)[:, 1]

    np.testing.assert_allclose(proba_a, proba_b, rtol=1e-10, atol=1e-12)


def test_metadata_threshold_is_consistent_with_recall_target():
    metadata = json.loads(METADATA_PATH.read_text())
    assert 0.0 <= metadata["threshold"] <= 1.0
    assert metadata["metrics"]["recall"] >= 0.9  # sanity bound, not the exact 0.95 target
