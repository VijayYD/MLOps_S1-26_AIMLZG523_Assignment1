# Data

**Status:** populated in Phase 2 (data acquisition and EDA).

This will document:
- Provenance and licence of the UCI Heart Disease dataset (four sites:
  Cleveland, Hungarian, Switzerland, VA Long Beach).
- The column dictionary (14 standard clinical attributes + `site` + `target`).
- SHA-256 checksums of each raw source file, verified by
  `python -m heart_mlops.data.download` on every run.
- The exact cleaning rules applied in `src/heart_mlops/data/clean.py` (missing
  value handling, implausible-zero handling, target binarisation).

`data/raw/` is gitignored (downloaded fresh via the download script); only
`data/processed/heart_combined_clean.csv` (the cleaned, de-duplicated dataset
with NaNs preserved) is committed, so a fresh clone can retrain without
depending on the UCI servers being reachable.
