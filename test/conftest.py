"""Shared pytest fixtures.

tiny_cleaned_df is a small, fast synthetic dataset shaped like the real
cleaned CSV -- same columns/dtypes/site values, deliberate missingness --
so preprocessing/model/API tests don't need the real 918-row dataset or a
slow fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heart_mlops.config import SITES


@pytest.fixture
def tiny_cleaned_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 80
    df = pd.DataFrame(
        {
            "age": rng.uniform(30, 75, n),
            "sex": rng.integers(0, 2, n),
            "cp": rng.integers(1, 5, n),
            "trestbps": rng.uniform(90, 180, n),
            "chol": rng.uniform(150, 350, n),
            "fbs": rng.integers(0, 2, n).astype(float),
            "restecg": rng.integers(0, 3, n).astype(float),
            "thalach": rng.uniform(80, 190, n),
            "exang": rng.integers(0, 2, n).astype(float),
            "oldpeak": rng.uniform(0, 4, n),
            "slope": rng.integers(1, 4, n).astype(float),
            "ca": rng.integers(0, 4, n).astype(float),
            "thal": rng.choice([3, 6, 7], n).astype(float),
            "num": rng.integers(0, 5, n),
        }
    )
    df["target"] = (df["num"] > 0).astype(int)
    df["site"] = rng.choice(SITES, n)

    for col in ["ca", "thal", "slope", "trestbps", "chol", "fbs", "restecg", "exang"]:
        mask = rng.random(n) < 0.3
        df.loc[mask, col] = np.nan

    return df
