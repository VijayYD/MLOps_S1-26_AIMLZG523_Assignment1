"""Short training run used only by scripts/lab_parity_check.sh, to prove
the real training pipeline actually fits on Python 3.12.9 (the lab's
version) -- not a real training session: tiny search space and fold
counts, and writes to a temp directory rather than overwriting the real
models/heart_model.joblib or this container's MLflow state.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from scipy.stats import randint

import heart_mlops.models.train as tr
from heart_mlops.models import registry

tr.N_SEARCH_ITER = 3
tr.CV_FOLDS = 2
tr.PARAM_DISTRIBUTIONS = {
    "logreg": {"classifier__C": [0.1, 1.0]},
    "random_forest": {"classifier__n_estimators": randint(10, 30)},
    "hist_gb": {"classifier__max_iter": [10, 20]},
}

with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_path = Path(tmp_dir)
    registry.MODEL_PATH = tmp_path / "heart_model.joblib"
    registry.METADATA_PATH = tmp_path / "model_metadata.json"
    tr.MLFLOW_TRACKING_URI_DEFAULT = f"sqlite:///{tmp_path / 'mlflow.db'}"
    tr.MLFLOW_ARTIFACT_ROOT = tmp_path / "mlartifacts"

    result = tr.main()
    print(
        f"Lab-parity training check OK -- selected {result['selected_name']}, "
        f"threshold={result['threshold']:.4f}"
    )
