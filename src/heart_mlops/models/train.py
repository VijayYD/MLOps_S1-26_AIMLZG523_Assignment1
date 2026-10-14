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

import hashlib
import logging
import os
import platform
import subprocess
import tempfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
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
    MLFLOW_ARTIFACT_ROOT,
    MLFLOW_CHAMPION_ALIAS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI_DEFAULT,
    MODEL_FEATURE_COLUMNS,
    MODELS_DIR,
    PROJECT_ROOT,
    RECALL_TARGET,
    SEED,
)
from heart_mlops.features.preprocess import MODEL_NAMES, build_pipeline
from heart_mlops.models import evaluate as ev
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


def evaluate_all_models(
    search_results: dict[str, RandomizedSearchCV],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, pd.Series], dict[str, float]]:
    """CV mean/std + hold-out metrics for *every* tuned model (not just the
    selected one), each at its own OOF-derived threshold, so the
    comparison is fair -- no model is scored at a threshold tuned for a
    different model. Shared by the training script and
    notebooks/02_modelling.ipynb so this logic exists in exactly one place.
    """
    rows = []
    holdout_probas = {}
    thresholds = {}
    for name, search in search_results.items():
        pipeline = search.best_estimator_
        model_threshold = compute_oof_threshold(pipeline, X_train, y_train)
        proba_test = pipeline.predict_proba(X_test)[:, 1]
        holdout_probas[name] = proba_test
        thresholds[name] = model_threshold
        metrics = compute_metrics(y_test, proba_test, model_threshold)
        rows.append(
            {
                "model": name,
                "cv_roc_auc_mean": search.cv_results_["mean_test_score"][search.best_index_],
                "cv_roc_auc_std": search.cv_results_["std_test_score"][search.best_index_],
                **{f"holdout_{k}": v for k, v in metrics.items()},
            }
        )
    comparison_df = pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False)
    return comparison_df, holdout_probas, thresholds


# --- MLflow experiment tracking (Phase 4) -----------------------------------


def get_git_sha() -> str:
    """Best-effort -- falls back to "unknown" rather than failing the whole
    training run over a provenance field, e.g. if run somewhere .git isn't
    present (not expected in normal use, but not worth crashing over)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_data_checksum() -> str:
    return hashlib.sha256(CLEANED_DATA_PATH.read_bytes()).hexdigest()


def configure_mlflow() -> None:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI_DEFAULT)
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            MLFLOW_EXPERIMENT_NAME, artifact_location=MLFLOW_ARTIFACT_ROOT.as_uri()
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


_PREPROCESSING_NOTES = {
    "logreg": "numeric: median-impute+indicator+scale; categorical: constant(-1)+one-hot",
    "random_forest": "numeric: median-impute+indicator (no scaling); categorical: constant(-1)+one-hot",
    "hist_gb": "no imputation -- native NaN + categorical_features handling",
}


def log_mlflow_run(run_context: dict) -> str:
    """Logs a full training session to MLflow: one parent run, one nested
    child run per model (params, CV + hold-out metrics, cv_results.csv,
    classification report), the selected model logged with a signature and
    input example, the model card and comparison plots as parent-level
    artifacts, and registers the selected model with the `champion` alias.
    Returns the parent run id.
    """
    configure_mlflow()

    X_train = run_context["X_train"]
    X_test, y_test = run_context["X_test"], run_context["y_test"]
    search_results = run_context["search_results"]
    selected_name = run_context["selected_name"]
    comparison_df = run_context["comparison_df"]
    holdout_probas = run_context["holdout_probas"]
    thresholds = run_context["thresholds"]

    shared_params = {
        "seed": SEED,
        "data_checksum": get_data_checksum(),
        "git_sha": get_git_sha(),
        "python_version": platform.python_version(),
        "cv_folds": CV_FOLDS,
        "recall_target": RECALL_TARGET,
        "holdout_test_size": HOLDOUT_TEST_SIZE,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        with mlflow.start_run(run_name="training-session") as parent_run:
            mlflow.log_params(shared_params)
            mlflow.set_tag("selected_model", selected_name)
            mlflow.set_tag("selection_reason", run_context["selection_reason"])

            comparison_csv = tmp_path / "model_comparison.csv"
            comparison_df.to_csv(comparison_csv, index=False)
            mlflow.log_artifact(str(comparison_csv))

            model_card_path = MODELS_DIR / "MODEL_CARD.md"
            if model_card_path.exists():
                mlflow.log_artifact(str(model_card_path))

            roc_curves = {name: (y_test, proba) for name, proba in holdout_probas.items()}
            for fig, fname in [
                (ev.plot_roc_curves(roc_curves), "roc_curves.png"),
                (ev.plot_pr_curves(roc_curves), "pr_curves.png"),
                (ev.plot_calibration_curve(roc_curves), "calibration_curve.png"),
            ]:
                fig_path = tmp_path / fname
                fig.savefig(fig_path, bbox_inches="tight", dpi=200)
                mlflow.log_artifact(str(fig_path))
                plt.close(fig)

            selected_child_run_id = None
            for name, search in search_results.items():
                with mlflow.start_run(run_name=name, nested=True) as child_run:
                    flat_params = {
                        k.replace("classifier__", ""): v for k, v in search.best_params_.items()
                    }
                    mlflow.log_params(flat_params)
                    mlflow.log_param("model_type", name)
                    mlflow.log_param("preprocessing", _PREPROCESSING_NOTES[name])

                    cv_results_csv = tmp_path / f"cv_results_{name}.csv"
                    pd.DataFrame(search.cv_results_).to_csv(cv_results_csv, index=False)
                    mlflow.log_artifact(str(cv_results_csv))

                    mlflow.log_metric(
                        "cv_roc_auc_mean", search.cv_results_["mean_test_score"][search.best_index_]
                    )
                    mlflow.log_metric(
                        "cv_roc_auc_std", search.cv_results_["std_test_score"][search.best_index_]
                    )

                    proba_test = holdout_probas[name]
                    model_threshold = thresholds[name]
                    metrics = compute_metrics(y_test, proba_test, model_threshold)
                    mlflow.log_metrics({f"holdout_{k}": v for k, v in metrics.items()})

                    report_path = tmp_path / f"classification_report_{name}.txt"
                    report_path.write_text(
                        ev.classification_report_text(y_test, proba_test, model_threshold)
                    )
                    mlflow.log_artifact(str(report_path))

                    cm_fig = ev.plot_confusion_matrix_at_threshold(
                        y_test, proba_test, model_threshold
                    )
                    cm_path = tmp_path / f"confusion_matrix_{name}.png"
                    cm_fig.savefig(cm_path, bbox_inches="tight", dpi=200)
                    mlflow.log_artifact(str(cm_path))
                    plt.close(cm_fig)

                    if name == selected_name:
                        selected_child_run_id = child_run.info.run_id
                        pipeline = search.best_estimator_

                        if name == "logreg":
                            imp_fig = ev.plot_logreg_coefficients(pipeline)
                        elif name == "random_forest":
                            imp_fig = ev.plot_rf_importance(pipeline, X_test, y_test)
                        else:
                            imp_fig = None
                        if imp_fig is not None:
                            imp_path = tmp_path / "feature_importance.png"
                            imp_fig.savefig(imp_path, bbox_inches="tight", dpi=200)
                            mlflow.log_artifact(str(imp_path))
                            plt.close(imp_fig)

                        signature = infer_signature(X_train, pipeline.predict_proba(X_train))
                        mlflow.sklearn.log_model(
                            pipeline,
                            artifact_path="model",
                            signature=signature,
                            input_example=X_train.head(3),
                        )

            model_uri = f"runs:/{selected_child_run_id}/model"
            registered = mlflow.register_model(model_uri, MLFLOW_REGISTERED_MODEL_NAME)
            MlflowClient().set_registered_model_alias(
                MLFLOW_REGISTERED_MODEL_NAME, MLFLOW_CHAMPION_ALIAS, registered.version
            )
            logger.info(
                "Registered %s v%s with alias '%s'",
                MLFLOW_REGISTERED_MODEL_NAME,
                registered.version,
                MLFLOW_CHAMPION_ALIAS,
            )

            return parent_run.info.run_id


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

    comparison_df, holdout_probas, thresholds = evaluate_all_models(
        search_results, X_train, y_train, X_test, y_test
    )

    run_context = {
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
        "comparison_df": comparison_df,
        "holdout_probas": holdout_probas,
        "thresholds": thresholds,
    }

    with joblib.parallel_backend("threading"):
        mlflow_run_id = log_mlflow_run(run_context)
    logger.info("MLflow parent run id: %s", mlflow_run_id)
    run_context["mlflow_run_id"] = mlflow_run_id

    return run_context


if __name__ == "__main__":
    main()
