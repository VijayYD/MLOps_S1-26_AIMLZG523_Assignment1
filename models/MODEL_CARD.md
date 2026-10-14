# Model Card — Heart Disease Risk Classifier

**Packaged artifact:** `models/heart_model.joblib` (full preprocessing +
classifier pipeline) + `models/model_metadata.json` (version, threshold,
feature schema, metrics, checksums, sklearn version, git SHA, timestamp --
regenerate both with `make train`). MLflow Model Registry:
`heart_disease_classifier`, alias `champion`. Both are the *same* fitted
pipeline in two serialisation forms (docs/DECISIONS.md D5.2); the API
(Phase 7) loads the joblib + metadata pair directly, not the MLflow
model/registry (D5.1).

## Intended use

Predicts the presence (vs. absence) of heart disease from 13 clinical
attributes, as a **screening-support** tool: it flags patients who may
warrant further clinical investigation. It does not diagnose, and its
output is a probability plus a threshold-based label, not a clinical
verdict.

## Not for clinical use

**This model is a course project artefact, not a validated medical
device.** It must not be used to make or inform real patient-care
decisions. It has not been validated prospectively, was trained on a
public research dataset collected in the 1980s-90s at four specific
hospitals, and carries all the limitations described below.

## Data

Combined UCI Heart Disease dataset, four sites (Cleveland, Hungarian,
Switzerland, VA Long Beach), 918 rows after cleaning (see
[`data/README.md`](../data/README.md) for provenance, checksums, and the
cleaning rules applied). 13 features used for modelling: `age`, `sex`,
`cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`,
`slope`, `ca`, `thal`. `site` is **not** a model input (grouping-only; see
`docs/DECISIONS.md` D2.5).

## Model and training

- **Selected model:** Logistic Regression (`liblinear` solver, L1
  penalty), selected by the pre-registered rule in `docs/DECISIONS.md`
  D3.1 (highest 5-fold cross-validated ROC-AUC among Logistic Regression,
  Random Forest, and HistGradientBoostingClassifier).
- **Preprocessing:** numeric features median-imputed with missing
  indicators (scaled, since Logistic Regression's coefficients and L1
  penalty require it); categorical features imputed with an explicit
  "missing" sentinel category, one-hot encoded.
- **Decision threshold:** 0.2419, chosen from out-of-fold predicted
  probabilities to meet a 0.95 recall target on the positive (disease)
  class (`docs/DECISIONS.md` D0.4) -- missed disease cases are treated as
  costlier than false alarms.

## Metrics (hold-out set, 184 rows, 20% stratified split)

| Metric | Value |
|---|---|
| Accuracy | 0.804 |
| Precision | 0.754 |
| Recall | 0.961 |
| F1 | 0.845 |
| ROC-AUC | 0.904 |
| PR-AUC | 0.915 |
| Brier score | 0.120 |

## Per-site performance (leave-one-site-out: trained on 3 sites, evaluated entirely on the 4th)

| Held-out site | Recall | ROC-AUC | Precision |
|---|---|---|---|
| Cleveland | 0.892 | 0.868 | 0.642 |
| Hungarian | 0.953 | 0.903 | 0.518 |
| Switzerland | 0.948 | 0.726 | 0.965 |
| VA | 0.926 | 0.736 | 0.774 |

Recall holds near or above the 0.95 target for 3 of 4 sites; Cleveland is
the exception. ROC-AUC and precision vary considerably more by site than
recall does -- see `notebooks/02_modelling.ipynb` section 10 for the full
discussion (base-rate shift for Hungarian, distributional shift for
Switzerland/VA).

## Limitations

- Training data is small (918 rows) and decades old; it may not reflect
  current patient populations, diagnostic practices, or equipment.
- `ca` and `thal` are >60% missing outside Cleveland; the model's behaviour
  on sites/populations with a different missingness pattern than the
  training mix is not guaranteed (see the leave-one-site-out results).
- The 0.95 recall target was chosen for this project, not validated
  against any real clinical cost-benefit analysis.
- No subgroup fairness analysis (e.g. by sex) has been performed beyond
  the exploratory data analysis in `notebooks/01_eda.ipynb`.

## Ethical considerations

- False negatives in a real deployment would mean missed disease cases;
  the recall-first threshold is a deliberate mitigation, not a guarantee.
- The training data's site/hospital origin correlates with factors (era,
  region, equipment) that could encode non-clinical bias; `site` is
  excluded as a feature specifically to reduce (not eliminate) this risk.
- This model must not be used for insurance, employment, or any
  decision with legal or financial consequences for a person.
