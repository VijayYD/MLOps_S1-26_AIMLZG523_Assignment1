"""Combine the four downloaded UCI site files into a single cleaned dataset.

Usage:
    python -m heart_mlops.data.clean

Cleaning rules (see docs/DECISIONS.md for the evidence behind each):
  - '?' tokens -> NaN.
  - Physiologically impossible zeros -> NaN: chol == 0 (all 123 Switzerland
    rows and 49 VA rows) and trestbps == 0 (1 VA row).
  - Binary target: target = 1 if num > 0 else 0 (num itself is kept too).
  - Exact full-row duplicates within a site are dropped (2 pairs found:
    1 in Hungarian, 1 in VA).
  - Categorical/count columns are stored as pandas nullable Int64 so NaN and
    integer codes coexist without the "1.0" float rendering.

This module does NOT impute or drop any column for modelling purposes --
the cleaned CSV keeps every column and every NaN. Imputation and the
ca/thal/slope/site feature-inclusion decisions are made inside the model
pipeline (src/heart_mlops/features/preprocess.py, Phase 3) so they are fit
on training folds only, never on the full dataset.
"""

from __future__ import annotations

import logging

import pandas as pd

from heart_mlops.config import (
    CLEANED_DATA_PATH,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    RAW_COLUMNS,
    SITES,
)

logger = logging.getLogger(__name__)

# Columns genuinely continuous; kept as float64.
CONTINUOUS_COLUMNS = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# Columns that are integer-coded categories/counts; stored as pandas nullable
# Int64 so missingness and integer codes coexist cleanly.
CATEGORICAL_INT_COLUMNS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "num"]

# Implausible-zero rules: (column, value treated as missing).
IMPLAUSIBLE_ZERO_COLUMNS = ["chol", "trestbps"]


def _load_site(site: str) -> pd.DataFrame:
    path = DATA_RAW_DIR / f"processed.{site}.data"
    df = pd.read_csv(path, header=None, names=RAW_COLUMNS, na_values="?")
    df["site"] = site
    return df


def load_combined_raw() -> pd.DataFrame:
    frames = [_load_site(site) for site in SITES]
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in RAW_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in IMPLAUSIBLE_ZERO_COLUMNS:
        n_zero = int((df[col] == 0).sum())
        if n_zero:
            logger.info(
                "%s: treating %d zero value(s) as missing (physiologically impossible)", col, n_zero
            )
        df.loc[df[col] == 0, col] = pd.NA

    before = len(df)
    df = df.drop_duplicates(subset=RAW_COLUMNS + ["site"], keep="first").reset_index(drop=True)
    n_dropped = before - len(df)
    if n_dropped:
        logger.info("Dropped %d exact duplicate row(s)", n_dropped)

    df["target"] = (df["num"] > 0).astype("int8")

    for col in CONTINUOUS_COLUMNS:
        df[col] = df[col].astype("float64")
    for col in CATEGORICAL_INT_COLUMNS:
        df[col] = df[col].astype("Int64")

    ordered_columns = RAW_COLUMNS[:-1] + ["num", "target", "site"]
    return df[ordered_columns]


def data_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Rows, missing % per column, and class balance -- one row per site plus
    an 'ALL' row, long-format so it's easy to both log and save as a table."""
    feature_cols = [c for c in RAW_COLUMNS if c != "num"]
    rows = []
    for site_name, group in list(df.groupby("site")) + [("ALL", df)]:
        row = {"site": site_name, "n_rows": len(group)}
        for col in feature_cols:
            row[f"missing_pct_{col}"] = round(group[col].isna().mean() * 100, 1)
        row["target_positive_pct"] = round(group["target"].mean() * 100, 1)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    raw = load_combined_raw()
    logger.info("Loaded %d raw rows across %d sites", len(raw), raw["site"].nunique())

    cleaned = clean(raw)
    logger.info("Cleaned dataset: %d rows, %d columns", *cleaned.shape)

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(CLEANED_DATA_PATH, index=False)
    logger.info("Wrote %s", CLEANED_DATA_PATH)

    summary = data_quality_summary(cleaned)
    summary_path = DATA_PROCESSED_DIR / "data_quality_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Wrote %s", summary_path)
    logger.info("\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
