"""Tests for src/heart_mlops/features/preprocess.py."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone

from heart_mlops.config import CATEGORICAL_FEATURES, MODEL_FEATURE_COLUMNS, NUMERIC_FEATURES
from heart_mlops.features.preprocess import MODEL_NAMES, build_pipeline


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_pipeline_fits_and_predicts(model_name, tiny_cleaned_df):
    X = tiny_cleaned_df[MODEL_FEATURE_COLUMNS]
    y = tiny_cleaned_df["target"]
    pipeline = build_pipeline(model_name)
    pipeline.fit(X, y)
    proba = pipeline.predict_proba(X)[:, 1]

    assert proba.shape == (len(X),)
    assert np.all((proba >= 0) & (proba <= 1))


@pytest.mark.parametrize("model_name", ["logreg", "random_forest"])
def test_column_transformer_output_has_no_nan(model_name, tiny_cleaned_df):
    """hist_gb is excluded deliberately -- it is fed raw NaNs by design
    (docs/DECISIONS.md D3.3), so "no NaN after preprocessing" doesn't apply."""
    X = tiny_cleaned_df[MODEL_FEATURE_COLUMNS]
    y = tiny_cleaned_df["target"]
    assert X[NUMERIC_FEATURES + CATEGORICAL_FEATURES].isna().any().any()  # fixture has real NaNs

    pipeline = build_pipeline(model_name)
    pipeline.fit(X, y)
    transformed = pipeline.named_steps["preprocess"].transform(X)
    assert not np.isnan(transformed).any()


def test_fit_on_train_only_does_not_see_test_rows(tiny_cleaned_df):
    """A ColumnTransformer fit on a subset must not error or silently use
    statistics from rows outside that subset -- here, checked by fitting
    on half the data and transforming the other half with an unseen
    categorical value."""
    X = tiny_cleaned_df[MODEL_FEATURE_COLUMNS].copy()
    y = tiny_cleaned_df["target"]
    split = len(X) // 2
    X_train, X_test = X.iloc[:split], X.iloc[split:].copy()

    pipeline = build_pipeline("logreg")
    pipeline.fit(X_train, y.iloc[:split])

    # Simulate a category value never seen in training.
    X_test.loc[X_test.index[0], "cp"] = 99
    proba = pipeline.predict_proba(X_test)[:, 1]
    assert np.all((proba >= 0) & (proba <= 1))


def test_unknown_category_at_predict_time_does_not_raise(tiny_cleaned_df):
    """OneHotEncoder(handle_unknown="ignore") must not raise on a category
    value absent from the training fold."""
    X = tiny_cleaned_df[MODEL_FEATURE_COLUMNS]
    y = tiny_cleaned_df["target"]
    pipeline = build_pipeline("random_forest")
    pipeline.fit(X, y)

    unseen_row = X.iloc[[0]].copy()
    unseen_row["thal"] = 999  # not in {3, 6, 7, NaN}
    proba = pipeline.predict_proba(unseen_row)[:, 1]
    assert proba.shape == (1,)


def test_clone_produces_independent_unfitted_pipeline(tiny_cleaned_df):
    X = tiny_cleaned_df[MODEL_FEATURE_COLUMNS]
    y = tiny_cleaned_df["target"]
    fitted = build_pipeline("logreg")
    fitted.fit(X, y)

    cloned = clone(fitted)
    with pytest.raises(Exception):  # noqa: B017 -- sklearn's NotFittedError, exact type not load-bearing here
        cloned.predict_proba(X)
