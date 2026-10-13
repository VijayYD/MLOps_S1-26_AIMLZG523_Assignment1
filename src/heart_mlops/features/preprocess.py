"""Builds the per-model preprocessing + classifier Pipeline.

Logistic Regression and Random Forest share a ColumnTransformer: numeric
columns are median-imputed with missing-indicator flags (scaled, for LR
only -- scaling is required for a regularised linear model's coefficients
to be comparable and for the penalty to apply fairly across features, and
is harmless-but-unnecessary for a tree-based model like Random Forest, so
it is skipped there rather than applied pointlessly), and categorical
columns are imputed with an explicit "missing" sentinel category (-1) before
one-hot encoding -- so missingness becomes its own learned category rather
than a separately bolted-on indicator column.

HistGradientBoostingClassifier is deliberately given no imputation at all:
its entire point as the project's third model is that it handles NaN
natively (both as an ordinary missing value for numeric splits and as its
own category for declared categorical columns), so running it through the
same imputer would throw away the reason it was chosen.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart_mlops.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, SEED

CATEGORICAL_MISSING_SENTINEL = -1

MODEL_NAMES = ("logreg", "random_forest", "hist_gb")


def _numeric_branch(scale: bool) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def _categorical_branch() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value=CATEGORICAL_MISSING_SENTINEL)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )


def build_column_transformer(scale_numeric: bool) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", _numeric_branch(scale=scale_numeric), NUMERIC_FEATURES),
            ("categorical", _categorical_branch(), CATEGORICAL_FEATURES),
        ]
    )


def build_estimator(model_name: str):
    if model_name == "logreg":
        # liblinear supports both l1 and l2 penalties (lbfgs, the default,
        # only supports l2), which the tuned search space (Phase 3) needs.
        return LogisticRegression(random_state=SEED, max_iter=2000, solver="liblinear")
    if model_name == "random_forest":
        return RandomForestClassifier(random_state=SEED, n_jobs=-1)
    if model_name == "hist_gb":
        return HistGradientBoostingClassifier(
            random_state=SEED, categorical_features=CATEGORICAL_FEATURES
        )
    raise ValueError(f"unknown model_name: {model_name!r}, expected one of {MODEL_NAMES}")


def build_pipeline(model_name: str) -> Pipeline:
    """Full preprocessing + classifier pipeline for one model.

    For logreg/random_forest, input is a DataFrame restricted to
    NUMERIC_FEATURES + CATEGORICAL_FEATURES, passed through the
    ColumnTransformer above. For hist_gb, the estimator consumes that same
    DataFrame directly (NaNs and all) with no preprocessing step, since
    HistGradientBoostingClassifier handles missingness and declared
    categorical columns on its own.
    """
    estimator = build_estimator(model_name)
    if model_name == "hist_gb":
        return Pipeline([("classifier", estimator)])

    scale_numeric = model_name == "logreg"
    preprocessor = build_column_transformer(scale_numeric=scale_numeric)
    return Pipeline([("preprocess", preprocessor), ("classifier", estimator)])
