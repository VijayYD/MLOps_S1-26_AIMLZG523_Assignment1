"""Metrics, plots, and the leave-one-site-out generalisation study.

Kept separate from train.py (tuning/selection) so both the training script
and notebooks/02_modelling.ipynb can call the same functions without
duplicating logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from heart_mlops.config import MODEL_FEATURE_COLUMNS, REPORTS_FIGURES_DIR, SEED, SITES


def savefig(fig: plt.Figure, name: str) -> None:
    REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=200)
    print(f"Saved {path}")


def threshold_for_recall(y_true, y_proba, recall_target: float) -> float:
    """Highest decision threshold whose precision-recall-curve recall is
    still >= recall_target -- i.e. the most precision-preserving threshold
    that meets the recall floor. Falls back to the most permissive
    threshold (0.0) if even that can't reach the target."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision[:-1]/recall[:-1] align 1:1 with `thresholds`; the trailing
    # (1.0, 0.0) pair has no corresponding threshold (sklearn convention).
    meets_target = recall[:-1] >= recall_target
    if not meets_target.any():
        return 0.0
    return float(thresholds[meets_target].max())


def compute_metrics(y_true, y_proba, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "brier": float(brier_score_loss(y_true, y_proba)),
        "threshold": float(threshold),
    }


def classification_report_text(y_true, y_proba, threshold: float) -> str:
    y_pred = (y_proba >= threshold).astype(int)
    return classification_report(
        y_true, y_pred, target_names=["No disease", "Disease"], zero_division=0
    )


# --- Plots -------------------------------------------------------------


def plot_roc_curves(curves: dict[str, tuple], save_as: str | None = None) -> plt.Figure:
    """curves: {label: (y_true, y_proba)}"""
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, (y_true, y_proba) in curves.items():
        RocCurveDisplay.from_predictions(y_true, y_proba, name=label, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_title("ROC curve")
    fig.tight_layout()
    if save_as:
        savefig(fig, save_as)
    return fig


def plot_pr_curves(curves: dict[str, tuple], save_as: str | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, (y_true, y_proba) in curves.items():
        PrecisionRecallDisplay.from_predictions(y_true, y_proba, name=label, ax=ax)
    ax.set_title("Precision-Recall curve")
    fig.tight_layout()
    if save_as:
        savefig(fig, save_as)
    return fig


def plot_confusion_matrix_at_threshold(
    y_true, y_proba, threshold: float, save_as: str | None = None
) -> plt.Figure:
    y_pred = (y_proba >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["No disease", "Disease"], cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion matrix @ threshold={threshold:.3f}")
    fig.tight_layout()
    if save_as:
        savefig(fig, save_as)
    return fig


def plot_calibration_curve(
    curves: dict[str, tuple], n_bins: int = 10, save_as: str | None = None
) -> plt.Figure:
    """curves: {label: (y_true, y_proba)}"""
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, (y_true, y_proba) in curves.items():
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_ids = np.clip(np.digitize(y_proba, bin_edges) - 1, 0, n_bins - 1)
        mean_pred = np.array(
            [
                y_proba[bin_ids == b].mean() if (bin_ids == b).any() else np.nan
                for b in range(n_bins)
            ]
        )
        frac_pos = np.array(
            [
                np.asarray(y_true)[bin_ids == b].mean() if (bin_ids == b).any() else np.nan
                for b in range(n_bins)
            ]
        )
        ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction positive")
    ax.set_title("Calibration curve")
    ax.legend()
    fig.tight_layout()
    if save_as:
        savefig(fig, save_as)
    return fig


def _onehot_feature_names(column_transformer) -> list[str]:
    return list(column_transformer.get_feature_names_out())


def plot_logreg_coefficients(pipeline, top_n: int = 20, save_as: str | None = None) -> plt.Figure:
    """Coefficients and odds ratios for a fitted logreg pipeline, largest
    |coefficient| first (top of the chart) -- matplotlib's barh default
    plots the first list item at the bottom, so the selected order is
    reversed before plotting to get the more conventional top-to-bottom
    "most important first" reading."""
    clf = pipeline.named_steps["classifier"]
    feature_names = _onehot_feature_names(pipeline.named_steps["preprocess"])
    coefs = clf.coef_[0]
    odds_ratios = np.exp(coefs)
    order = np.argsort(np.abs(coefs))[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(order))))
    ax.barh(
        [feature_names[i] for i in order][::-1],
        [odds_ratios[i] for i in order][::-1],
        color=["#C44E52" if coefs[i] > 0 else "#4C72B0" for i in order][::-1],
    )
    ax.axvline(1.0, color="black", linewidth=0.8)
    ax.set_xlabel("Odds ratio (exp(coefficient))")
    ax.set_title(
        f"Logistic Regression — odds ratios, top {len(order)} by |coefficient| (>1 raises disease odds)"
    )
    fig.tight_layout()
    if save_as:
        savefig(fig, save_as)
    return fig


def plot_rf_importance(pipeline, X_holdout, y_holdout, save_as: str | None = None) -> plt.Figure:
    """Impurity-based (left) and permutation (right) importance for a fitted
    Random Forest pipeline. Permutation importance is computed on the
    holdout set (not training data) to avoid the known overfitting bias of
    impurity-based importance on correlated / high-cardinality features."""
    clf = pipeline.named_steps["classifier"]
    feature_names = _onehot_feature_names(pipeline.named_steps["preprocess"])
    impurity = clf.feature_importances_

    # Threading backend: see the comment in models/train.py::main for why
    # (avoids noisy-but-harmless multiprocessing teardown logs on macOS/py3.13
    # for a job this small; not a workaround for an actual failure).
    with joblib.parallel_backend("threading"):
        perm = permutation_importance(
            pipeline,
            X_holdout,
            y_holdout,
            n_repeats=20,
            random_state=SEED,
            scoring="roc_auc",
            n_jobs=-1,
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, 0.3 * len(feature_names))))

    order_imp = np.argsort(impurity)[::-1][:20]
    axes[0].barh(
        [feature_names[i] for i in order_imp][::-1], impurity[order_imp][::-1], color="#55A868"
    )
    axes[0].set_title("Impurity-based importance (top 20)")

    perm_mean = perm.importances_mean
    order_perm = np.argsort(perm_mean)[::-1][:20]
    axes[1].barh(
        [
            MODEL_FEATURE_COLUMNS[i] if i < len(MODEL_FEATURE_COLUMNS) else str(i)
            for i in order_perm
        ][::-1],
        perm_mean[order_perm][::-1],
        xerr=perm.importances_std[order_perm][::-1],
        color="#4C72B0",
    )
    axes[1].set_title("Permutation importance on holdout (top 20, by raw column)")
    fig.tight_layout()
    if save_as:
        savefig(fig, save_as)
    return fig


# --- Leave-one-site-out --------------------------------------------------


@dataclass
class LosoResult:
    held_out_site: str
    n_train: int
    n_test: int
    metrics: dict


def leave_one_site_out(
    build_pipeline_fn, best_params: dict, df: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    """Train on 3 of the 4 sites, evaluate on the held-out 4th, for each
    site in turn. Uses the single globally-selected threshold (not
    re-tuned per fold) since the deployed model will use one fixed
    threshold regardless of which hospital a request comes from."""
    rows = []
    with joblib.parallel_backend("threading"):
        for held_out in SITES:
            train_df = df[df["site"] != held_out]
            test_df = df[df["site"] == held_out]

            X_train = train_df[MODEL_FEATURE_COLUMNS]
            y_train = train_df["target"]
            X_test = test_df[MODEL_FEATURE_COLUMNS]
            y_test = test_df["target"]

            pipeline = clone(build_pipeline_fn())
            pipeline.set_params(**best_params)
            pipeline.fit(X_train, y_train)
            y_proba = pipeline.predict_proba(X_test)[:, 1]

            metrics = compute_metrics(y_test, y_proba, threshold)
            rows.append(
                {
                    "held_out_site": held_out,
                    "n_train": len(train_df),
                    "n_test": len(test_df),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)
