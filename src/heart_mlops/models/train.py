"""Cross-validated hyperparameter tuning and model selection for the three
heart-disease models (Logistic Regression, Random Forest,
HistGradientBoostingClassifier).

Usage:
    python -m heart_mlops.models.train

Model-selection rule (fixed here, before any hold-out results are looked
at -- see docs/DECISIONS.md D3.1): the model with the highest mean
cross-validated ROC-AUC (`best_score_` of its own tuned
RandomizedSearchCV) wins. Ties within TIE_BREAK_TOLERANCE are broken by
preferring the simpler model, in order logreg > random_forest > hist_gb.
The hold-out set is touched only *after* this rule has already picked a
winner; it is never used to choose between models.

ROC-AUC is the tuning objective (not e.g. F1 or accuracy) because it is
threshold-independent -- the actual operating threshold is chosen
separately, afterwards, from out-of-fold probabilities to hit the agreed
clinical recall target (docs/DECISIONS.md D0.4), so optimising a
threshold-dependent metric during tuning would be premature and would
conflate two different decisions.
"""

from __future__ import annotations

import logging

import joblib
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

from heart_mlops.config import (
    CLEANED_DATA_PATH,
    CV_FOLDS,
    HOLDOUT_TEST_SIZE,
    MODEL_FEATURE_COLUMNS,
    RECALL_TARGET,
    SEED,
)
from heart_mlops.features.preprocess import MODEL_NAMES, build_pipeline
from heart_mlops.models.evaluate import compute_metrics, threshold_for_recall

logger = logging.getLogger(__name__)

N_SEARCH_ITER = 30
TIE_BREAK_TOLERANCE = 0.005
MODEL_SIMPLICITY_ORDER = ["logreg", "random_forest", "hist_gb"]

PARAM_DISTRIBUTIONS = {
    "logreg": {
        "classifier__C": loguniform(1e-3, 1e2),
        "classifier__penalty": ["l1", "l2"],
        "classifier__class_weight": [None, "balanced"],
    },
    "random_forest": {
        "classifier__n_estimators": randint(100, 600),
        "classifier__max_depth": [None, 5, 10, 20],
        "classifier__min_samples_leaf": randint(1, 11),
        "classifier__max_features": ["sqrt", "log2", None],
        "classifier__class_weight": [None, "balanced"],
    },
    "hist_gb": {
        "classifier__max_iter": randint(100, 400),
        "classifier__max_depth": [None, 3, 5, 10],
        "classifier__learning_rate": loguniform(1e-2, 3e-1),
        "classifier__max_leaf_nodes": [15, 31, 63],
        "classifier__l2_regularization": [0.0, 0.1, 1.0],
        "classifier__class_weight": [None, "balanced"],
    },
}


def load_dataset() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(CLEANED_DATA_PATH)
    X = df[MODEL_FEATURE_COLUMNS]
    y = df["target"]
    site = df["site"]
    return X, y, site


def make_holdout_split(X, y, site):
    """Stratifies on the combined (site, target) key, not target alone.

    Justification (docs/DECISIONS.md D3.2): Switzerland has only 8
    negative-class rows in total. Stratifying on target alone could, by
    chance, place a disproportionate share of those 8 rows into either the
    train or the hold-out split; stratifying on the joint key guarantees
    every site's class balance is preserved in both splits.
    """
    strata = site.astype(str) + "_" + y.astype(str)
    return train_test_split(
        X, y, site, test_size=HOLDOUT_TEST_SIZE, stratify=strata, random_state=SEED
    )


def tune_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> RandomizedSearchCV:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    search = RandomizedSearchCV(
        build_pipeline(model_name),
        param_distributions=PARAM_DISTRIBUTIONS[model_name],
        n_iter=N_SEARCH_ITER,
        scoring="roc_auc",
        cv=cv,
        random_state=SEED,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def tune_all_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, RandomizedSearchCV]:
    results = {}
    for model_name in MODEL_NAMES:
        logger.info("Tuning %s (%d candidates x %d folds)...", model_name, N_SEARCH_ITER, CV_FOLDS)
        results[model_name] = tune_model(model_name, X_train, y_train)
        logger.info(
            "%s: best CV ROC-AUC = %.4f, params = %s",
            model_name,
            results[model_name].best_score_,
            results[model_name].best_params_,
        )
    return results


def select_best_model(search_results: dict[str, RandomizedSearchCV]) -> tuple[str, str]:
    """Applies the model-selection rule from this module's docstring."""
    scores = {name: result.best_score_ for name, result in search_results.items()}
    best_score = max(scores.values())
    contenders = [
        name for name, score in scores.items() if best_score - score <= TIE_BREAK_TOLERANCE
    ]

    if len(contenders) == 1:
        winner = contenders[0]
        reason = f"highest CV ROC-AUC ({scores[winner]:.4f}), no tie within {TIE_BREAK_TOLERANCE}"
    else:
        winner = min(contenders, key=MODEL_SIMPLICITY_ORDER.index)
        reason = (
            f"tie within {TIE_BREAK_TOLERANCE} ROC-AUC among {contenders} "
            f"(scores={ {n: round(scores[n], 4) for n in contenders} }); "
            f"picked simplest per MODEL_SIMPLICITY_ORDER"
        )
    return winner, reason


def compute_oof_threshold(pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    """Out-of-fold predicted probabilities (same CV scheme as tuning, on the
    *selected* model's best hyperparameters) -> threshold meeting
    RECALL_TARGET. Out-of-fold, not in-sample, so the threshold is not
    optimistic about data the model has already seen."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    oof_proba = cross_val_predict(
        pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    return threshold_for_recall(y_train, oof_proba, RECALL_TARGET)


def main() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    X, y, site = load_dataset()
    X_train, X_test, y_train, y_test, site_train, site_test = make_holdout_split(X, y, site)
    logger.info("Train: %d rows, Hold-out: %d rows", len(X_train), len(X_test))

    # Threading backend, not the joblib/loky default process-based backend:
    # this dataset is small enough (734 training rows) that process-spawning
    # overhead dominates, and on macOS + Python 3.13 the default backend's
    # process teardown logs harmless-but-noisy ChildProcessError tracebacks
    # from CPython's own multiprocessing.resource_tracker after each run
    # completes successfully (exit code 0 either way). Threading sidesteps
    # both issues; it's not a workaround for a real failure, since nothing
    # was actually failing.
    with joblib.parallel_backend("threading"):
        search_results = tune_all_models(X_train, y_train)
        selected_name, reason = select_best_model(search_results)
        logger.info("Selected model: %s (%s)", selected_name, reason)

        selected_search = search_results[selected_name]
        selected_pipeline = selected_search.best_estimator_
        threshold = compute_oof_threshold(selected_pipeline, X_train, y_train)
    logger.info("Threshold for recall >= %.2f: %.4f", RECALL_TARGET, threshold)

    y_test_proba = selected_pipeline.predict_proba(X_test)[:, 1]
    holdout_metrics = compute_metrics(y_test, y_test_proba, threshold)
    logger.info("Hold-out metrics: %s", holdout_metrics)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "site_train": site_train,
        "site_test": site_test,
        "search_results": search_results,
        "selected_name": selected_name,
        "selection_reason": reason,
        "selected_pipeline": selected_pipeline,
        "threshold": threshold,
        "holdout_metrics": holdout_metrics,
    }


if __name__ == "__main__":
    main()
