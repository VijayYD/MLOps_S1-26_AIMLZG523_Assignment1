# Data

## Provenance and licence

Source: [UCI Machine Learning Repository, Heart Disease dataset (id 45)](https://archive.ics.uci.edu/dataset/45/heart+disease),
donated by Andras Janosi (Hungarian Institute of Cardiology, Budapest),
William Steinbrunn (University Hospital, Zurich), Matthias Pfisterer
(University Hospital, Basel) and Robert Detrano (V.A. Medical Center, Long
Beach and Cleveland Clinic Foundation). Licensed under CC BY 4.0.

This project uses **all four sites combined** (~920 rows total), not just the
usual 303-row Cleveland-only subset, to get realistic site-dependent
missingness and distribution shift (see `docs/DECISIONS.md` and
`notebooks/01_eda.ipynb` for why).

Downloaded by `python -m heart_mlops.data.download`:

| Site | Primary URL | Rows |
|---|---|---|
| Cleveland | `https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data` | 303 |
| Hungarian | `https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data` | 294 |
| Switzerland | `https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data` | 123 |
| VA Long Beach | `https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data` | 200 |

Fallback (used automatically if a primary URL fails): the UCI static zip for
dataset id 45, `https://archive.ics.uci.edu/static/public/45/heart+disease.zip`,
which bundles byte-identical copies of the same four files (verified by diff).

SHA-256 checksums (verified automatically on every download; a mismatch
aborts with a non-zero exit rather than proceeding with unexpected data):

| File | SHA-256 |
|---|---|
| `processed.cleveland.data` | `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8` |
| `processed.hungarian.data` | `d1ad108f785768cd3d7e82dc522e6f5a61eea93cccfb3a46ee8076f73fc3d796` |
| `processed.switzerland.data` | `834a405ccf5b66ab4056bb77794adc8df0b7125186454c0a1d002d33c6c3b314` |
| `processed.va.data` | `e7c93d8d0d2acdadfa4c5e8de768e2191e7f618b952e29623f1f0d5949ff6b8f` |

`data/raw/` itself is gitignored; only the cleaned, combined dataset below is
committed, so a fresh clone can retrain even if UCI is unreachable.

## Column dictionary

| Column | Meaning | Type |
|---|---|---|
| `age` | Age in years | continuous |
| `sex` | 1 = male, 0 = female | categorical |
| `cp` | Chest pain type (1-4) | categorical |
| `trestbps` | Resting blood pressure (mm Hg) | continuous |
| `chol` | Serum cholesterol (mg/dl) | continuous |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) | categorical |
| `restecg` | Resting ECG results (0-2) | categorical |
| `thalach` | Maximum heart rate achieved | continuous |
| `exang` | Exercise-induced angina (1 = yes) | categorical |
| `oldpeak` | ST depression induced by exercise relative to rest | continuous |
| `slope` | Slope of the peak exercise ST segment (1-3) | categorical |
| `ca` | Number of major vessels (0-3) coloured by fluoroscopy | categorical |
| `thal` | 3 = normal, 6 = fixed defect, 7 = reversible defect | categorical |
| `num` | Original diagnosis of heart disease, 0 (absent) to 4 | ordinal, kept for reference |
| `target` | Binarised label: `1` if `num > 0` else `0` | **model target** |
| `site` | Source site (`cleveland`/`hungarian`/`switzerland`/`va`) | grouping only -- see below |

## Cleaning rules applied (`src/heart_mlops/data/clean.py`)

1. `?` tokens -> `NaN` (the UCI convention for missing values).
2. **Physiologically impossible zeros -> `NaN`**, based on evidence from the
   raw data, not assumption:
   - `chol == 0`: **all 123 Switzerland rows** and **49 of 200 VA rows**
     (172 total). Serum cholesterol cannot be zero in a living patient --
     this is a site-specific recording convention for "not measured", not a
     true reading.
   - `trestbps == 0`: 1 VA row. Resting blood pressure cannot be zero.
3. **Exact duplicate rows dropped**: 2 pairs found (identical across all 14
   raw columns, including `site`) -- one pair in Hungarian, one in VA.
   918 rows remain (from 920).
4. **Target binarisation**: `target = 1 if num > 0 else 0`. The original
   `num` (0-4 severity) is kept alongside `target` for reference.
5. **The cleaned CSV keeps every `NaN`** -- imputation is deliberately *not*
   done here. It happens inside the model pipeline (Phase 3), fit on
   training folds only, so no information leaks from validation/test rows
   into imputed values.

## Missingness and class balance by site (from `python -m heart_mlops.data.clean`)

| site | n_rows | trestbps | chol | slope | ca | thal | target positive % |
|---|---|---|---|---|---|---|---|
| cleveland | 303 | 0.0% | 0.0% | 0.0% | 1.3% | 0.7% | 45.9% |
| hungarian | 293 | 0.3% | 7.5% | 64.5% | 99.0% | 90.4% | 36.2% |
| switzerland | 123 | 1.6% | 100.0%\* | 4.9% | 95.9% | 42.3% | 93.5% |
| va | 199 | 28.6% | 28.1% | 50.8% | 99.0% | 82.9% | 74.4% |
| **ALL** | **918** | 6.5% | 21.9% | 6.8%\*\* | 66.3% | 52.7% | 55.3% |

\* Switzerland's 100% `chol` missingness is exactly the 123 zero-values
described above, now correctly flagged as missing rather than treated as
real readings.
\*\* `slope`'s low ALL-row missing % is misleading: Cleveland (the largest
site) has 0% missing, which pulls the combined average down even though
3 of 4 sites are missing `slope` roughly half the time.

Full per-column, per-site breakdown is regenerated by
`python -m heart_mlops.data.clean` into `data/processed/data_quality_summary.csv`.

## `ca`, `thal`, `slope`, `site` -- feature-inclusion decisions (confirmed with the project owner; implemented in Phase 3)

- **`ca`**: non-missing in 98.7% of Cleveland rows but only 1-4% of rows in
  the other three sites -- a missing-indicator flag on `ca` mostly encodes
  "is this a Cleveland row", i.e. a site-leakage risk. This was flagged, and
  the project owner chose to **keep `ca`** (with a missing-indicator flag)
  specifically so the leave-one-site-out evaluation can show empirically
  whether it hurts cross-site generalisation, rather than assuming it will
  and dropping it pre-emptively.
- **`thal`** and **`slope`**: moderate coverage in 2-3 sites (`thal` 17-90%
  missing outside Cleveland, `slope` 35-65% missing outside Cleveland) --
  enough real signal to be worth keeping, but still a leakage risk.
  **Confirmed: keep both, imputed with missing-indicator flags** inside the
  pipeline, and watch their importance scores plus the leave-one-site-out
  results for signs the model is leaning on "was this measured" rather than
  the value itself.
- **`site`**: **confirmed excluded as a model input feature.** It is used
  only as a grouping variable for the leave-one-site-out evaluation and for
  reporting per-site metrics. Rationale: a model deployed on a genuinely new
  hospital has no value for a `site` feature it was trained on, and
  including it risks the model learning site-specific shortcuts (e.g.
  "Hungarian-shaped missingness pattern -> predict X") instead of clinically
  meaningful relationships. This also means the leave-one-site-out
  evaluation is a fair test of generalisation, not something the model could
  solve by memorising a site code.

See `docs/DECISIONS.md` D2.5 for the full rationale, including the explicit
plan to report (not silently act on) `ca`'s importance/leakage signal in
Phase 3.
