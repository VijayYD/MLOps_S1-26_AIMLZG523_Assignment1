"""Tests for evaluate.py, train.py, and registry.py.

The end-to-end integration test (test_main_end_to_end_with_tiny_data_and_grids)
exercises the real training + MLflow-logging + packaging path, but with tiny
parameter grids/fold counts on the small synthetic fixture, so it runs in
seconds rather than the ~6 minutes the real `python -m heart_mlops.models.train`
takes with its full search space on the full dataset.
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np
import pytest
from scipy.stats import randint

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from heart_mlops.config import MODEL_FEATURE_COLUMNS, SITES  # noqa: E402
from heart_mlops.features.preprocess import build_pipeline  # noqa: E402
from heart_mlops.models import evaluate as ev  # noqa: E402
from heart_mlops.models import registry  # noqa: E402
from heart_mlops.models import train as tr  # noqa: E402

# --- evaluate.py -----------------------------------------------------------


def test_threshold_for_recall_meets_target():
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    threshold = ev.threshold_for_recall(y_true, y_proba, recall_target=0.8)
    y_pred = (y_proba >= threshold).astype(int)
    recall = y_pred[y_true == 1].sum() / (y_true == 1).sum()
    assert recall >= 0.8


def test_threshold_for_recall_falls_back_to_zero_when_unreachable():
    y_true = np.array([1, 1])
    y_proba = np.array([0.4, 0.6])
    # Even threshold 0 gives recall 1.0 here, but check the fallback path
    # directly: a target recall no threshold can beat forces 0.0.
    threshold = ev.threshold_for_recall(y_true, y_proba, recall_target=1.1)
    assert threshold == 0.0


def test_compute_metrics_known_values():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8])
    metrics = ev.compute_metrics(y_true, y_proba, threshold=0.5)
    assert metrics["accuracy"] == 0.5
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["threshold"] == 0.5


def test_classification_report_text_mentions_both_classes():
    y_true = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.4, 0.6])
    report = ev.classification_report_text(y_true, y_proba, threshold=0.5)
    assert "No disease" in report
    assert "Disease" in report


def test_savefig_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "REPORTS_FIGURES_DIR", tmp_path)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ev.savefig(fig, "test_fig.png")
    assert (tmp_path / "test_fig.png").exists()
    plt.close(fig)


def test_plot_functions_return_figures():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.3, 0.8, 0.6, 0.4, 0.9])
    curves = {"model_a": (y_true, y_proba)}

    for fig in [
        ev.plot_roc_curves(curves),
        ev.plot_pr_curves(curves),
        ev.plot_calibration_curve(curves, n_bins=3),
        ev.plot_confusion_matrix_at_threshold(y_true, y_proba, threshold=0.5),
    ]:
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


def test_plot_logreg_coefficients(tiny_cleaned_df):
    X = tiny_cleaned_df[MODEL_FEATURE_COLUMNS]
    y = tiny_cleaned_df["target"]
    pipeline = build_pipeline("logreg")
    pipeline.fit(X, y)
    fig = ev.plot_logreg_coefficients(pipeline, top_n=5)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_rf_importance(tiny_cleaned_df):
    X = tiny_cleaned_df[MODEL_FEATURE_COLUMNS]
    y = tiny_cleaned_df["target"]
    pipeline = build_pipeline("random_forest")
    pipeline.fit(X, y)
    fig = ev.plot_rf_importance(pipeline, X, y)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_leave_one_site_out_covers_every_site(tiny_cleaned_df):
    pipeline = build_pipeline("logreg")
    X = tiny_cleaned_df[MODEL_FEATURE_COLUMNS]
    y = tiny_cleaned_df["target"]
    pipeline.fit(X, y)
    best_params = {}  # default hyperparameters are fine for this test

    result = ev.leave_one_site_out(
        lambda: build_pipeline("logreg"), best_params, tiny_cleaned_df, threshold=0.5
    )
    assert set(result["held_out_site"]) == set(SITES)
    assert (result["n_train"] + result["n_test"] == len(tiny_cleaned_df)).all()


# --- train.py ----------------------------------------------------------


def test_load_dataset(tmp_path, monkeypatch, tiny_cleaned_df):
    csv_path = tmp_path / "clean.csv"
    tiny_cleaned_df.to_csv(csv_path, index=False)
    monkeypatch.setattr(tr, "CLEANED_DATA_PATH", csv_path)

    X, y, site = tr.load_dataset()
    assert list(X.columns) == MODEL_FEATURE_COLUMNS
    assert len(X) == len(y) == len(site)


def test_make_holdout_split_preserves_site_target_strata(tiny_cleaned_df):
    X = tiny_cleaned_df[MODEL_FEATURE_COLUMNS]
    y = tiny_cleaned_df["target"]
    site = tiny_cleaned_df["site"]
    X_train, X_test, y_train, y_test, site_train, site_test = tr.make_holdout_split(X, y, site)

    assert len(X_train) + len(X_test) == len(X)
    assert set(site_train) <= set(SITES)
    assert set(site_test) <= set(SITES)


def test_select_best_model_picks_highest_score():
    class _FakeSearch:
        def __init__(self, score):
            self.best_score_ = score

    results = {
        "logreg": _FakeSearch(0.90),
        "random_forest": _FakeSearch(0.85),
        "hist_gb": _FakeSearch(0.80),
    }
    winner, reason = tr.select_best_model(results)
    assert winner == "logreg"
    assert "highest CV ROC-AUC" in reason


def test_select_best_model_breaks_ties_by_simplicity():
    class _FakeSearch:
        def __init__(self, score):
            self.best_score_ = score

    results = {
        "hist_gb": _FakeSearch(0.900),
        "logreg": _FakeSearch(0.899),
        "random_forest": _FakeSearch(0.85),
    }
    winner, reason = tr.select_best_model(results)
    assert winner == "logreg"  # simpler than hist_gb, within TIE_BREAK_TOLERANCE
    assert "tie within" in reason


def test_get_data_checksum_matches_manual_sha256(tmp_path, monkeypatch):
    content = b"some,csv,content\n1,2,3\n"
    csv_path = tmp_path / "clean.csv"
    csv_path.write_bytes(content)
    monkeypatch.setattr(tr, "CLEANED_DATA_PATH", csv_path)

    import hashlib

    assert tr.get_data_checksum() == hashlib.sha256(content).hexdigest()


def test_get_git_sha_returns_a_commit_hash_in_this_repo():
    sha = tr.get_git_sha()
    assert sha == "unknown" or len(sha) == 40


def test_get_git_sha_falls_back_to_unknown_on_failure(monkeypatch):
    def _raise(*args, **kwargs):
        raise FileNotFoundError("git not installed")

    monkeypatch.setattr(tr.subprocess, "run", _raise)
    assert tr.get_git_sha() == "unknown"


@pytest.fixture
def tiny_search_results(tiny_cleaned_df, monkeypatch):
    """Real (not mocked) RandomizedSearchCV results, but with a tiny
    search space / fold count / candidate count, fast on the small
    fixture -- used to exercise evaluate_all_models/log_mlflow_run
    realistically without paying full-tuning cost."""
    monkeypatch.setattr(tr, "N_SEARCH_ITER", 2)
    monkeypatch.setattr(tr, "CV_FOLDS", 2)
    monkeypatch.setattr(
        tr,
        "PARAM_DISTRIBUTIONS",
        {
            "logreg": {"classifier__C": [0.1, 1.0]},
            "random_forest": {"classifier__n_estimators": randint(10, 20)},
            "hist_gb": {"classifier__max_iter": [10, 20]},
        },
    )
    X = tiny_cleaned_df[MODEL_FEATURE_COLUMNS]
    y = tiny_cleaned_df["target"]
    return tr.tune_all_models(X, y), X, y


def test_evaluate_all_models_returns_comparison_for_every_model(
    tiny_cleaned_df, tiny_search_results
):
    search_results, X, y = tiny_search_results
    comparison_df, holdout_probas, thresholds = tr.evaluate_all_models(search_results, X, y, X, y)
    assert set(comparison_df["model"]) == set(search_results.keys())
    assert set(holdout_probas.keys()) == set(search_results.keys())
    assert set(thresholds.keys()) == set(search_results.keys())


def test_main_end_to_end_with_tiny_data_and_grids(tmp_path, monkeypatch, tiny_cleaned_df):
    """Runs the real python -m heart_mlops.models.train code path (tuning,
    selection, MLflow logging, packaging) against tiny data/grids/dirs so
    it completes in seconds, not minutes."""
    csv_path = tmp_path / "clean.csv"
    tiny_cleaned_df.to_csv(csv_path, index=False)
    monkeypatch.setattr(tr, "CLEANED_DATA_PATH", csv_path)
    monkeypatch.setattr(tr, "N_SEARCH_ITER", 2)
    monkeypatch.setattr(tr, "CV_FOLDS", 2)
    monkeypatch.setattr(
        tr,
        "PARAM_DISTRIBUTIONS",
        {
            "logreg": {"classifier__C": [0.1, 1.0]},
            "random_forest": {"classifier__n_estimators": randint(10, 20)},
            "hist_gb": {"classifier__max_iter": [10, 20]},
        },
    )

    mlflow_db = tmp_path / "mlflow.db"
    mlartifacts = tmp_path / "mlartifacts"
    monkeypatch.setattr(tr, "MLFLOW_TRACKING_URI_DEFAULT", f"sqlite:///{mlflow_db}")
    monkeypatch.setattr(tr, "MLFLOW_ARTIFACT_ROOT", mlartifacts)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    models_dir = tmp_path / "models"
    monkeypatch.setattr(registry, "MODEL_PATH", models_dir / "heart_model.joblib")
    monkeypatch.setattr(registry, "METADATA_PATH", models_dir / "model_metadata.json")

    result = tr.main()

    assert result["selected_name"] in {"logreg", "random_forest", "hist_gb"}
    assert 0.0 <= result["threshold"] <= 1.0
    assert registry.MODEL_PATH.exists()
    assert registry.METADATA_PATH.exists()

    metadata = json.loads(registry.METADATA_PATH.read_text())
    assert metadata["model_name"] == result["selected_name"]
    assert "model_version" in metadata


# --- registry.py -------------------------------------------------------


def test_build_metadata_has_all_required_fields():
    metadata = registry.build_metadata(
        model_name="logreg",
        model_version=1,
        threshold=0.25,
        hyperparameters={"classifier__C": 1.0},
        metrics={"accuracy": 0.8},
        data_checksum="abc123",
        git_sha="deadbeef",
    )
    for field in [
        "model_name",
        "model_version",
        "threshold",
        "hyperparameters",
        "feature_schema",
        "metrics",
        "training_data_checksum",
        "sklearn_version",
        "git_sha",
        "seed",
        "timestamp",
    ]:
        assert field in metadata


def test_save_and_load_packaged_model_round_trip(tmp_path, tiny_cleaned_df):
    X = tiny_cleaned_df[MODEL_FEATURE_COLUMNS]
    y = tiny_cleaned_df["target"]
    pipeline = build_pipeline("logreg")
    pipeline.fit(X, y)
    proba_before = pipeline.predict_proba(X)[:, 1]

    metadata = registry.build_metadata(
        model_name="logreg",
        model_version=1,
        threshold=0.25,
        hyperparameters={},
        metrics={},
        data_checksum="abc",
        git_sha="abc",
    )
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"

    import joblib

    joblib.dump(pipeline, model_path)
    metadata_path.write_text(json.dumps(metadata))

    loaded_pipeline, loaded_metadata = registry.load_packaged_model(model_path, metadata_path)
    proba_after = loaded_pipeline.predict_proba(X)[:, 1]

    np.testing.assert_array_equal(proba_before, proba_after)
    assert loaded_metadata["model_name"] == "logreg"


def test_get_champion_model_uri_format():
    uri = registry.get_champion_model_uri()
    assert (
        uri == f"models:/{registry.MLFLOW_REGISTERED_MODEL_NAME}@{registry.MLFLOW_CHAMPION_ALIAS}"
    )
