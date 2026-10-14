"""Pydantic v2 request/response models for the API.

Field bounds are evidence-based, not guessed: taken from the observed
min/max in data/processed/heart_combined_clean.csv with headroom for
clinically plausible values outside this particular 918-row sample (e.g.
`trestbps`/`chol`/`thalach` observed ranges are widened; `oldpeak`'s lower
bound of -2.6 in the data shows ST depression really can read negative,
so -3.0 is used rather than a naive >=0).

All 13 fields are required (not optional/nullable) -- the trained pipeline
can handle missing values internally, but a client-facing API accepting
partial clinical records adds real complexity (which fields may be
omitted together, how that interacts with validation) for a feature this
project doesn't otherwise need; see docs/DECISIONS.md D6.x.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PatientFeatures(BaseModel):
    age: float = Field(ge=1, le=120, description="Age in years", examples=[63])
    sex: Literal[0, 1] = Field(description="1 = male, 0 = female", examples=[1])
    cp: Literal[1, 2, 3, 4] = Field(description="Chest pain type (1-4)", examples=[1])
    trestbps: float = Field(
        ge=50, le=260, description="Resting blood pressure (mm Hg)", examples=[145]
    )
    chol: float = Field(ge=50, le=700, description="Serum cholesterol (mg/dl)", examples=[233])
    fbs: Literal[0, 1] = Field(description="Fasting blood sugar > 120 mg/dl", examples=[1])
    restecg: Literal[0, 1, 2] = Field(description="Resting ECG result (0-2)", examples=[2])
    thalach: float = Field(ge=50, le=250, description="Max heart rate achieved", examples=[150])
    exang: Literal[0, 1] = Field(description="Exercise-induced angina", examples=[0])
    oldpeak: float = Field(
        ge=-3,
        le=10,
        description="ST depression induced by exercise relative to rest",
        examples=[2.3],
    )
    slope: Literal[1, 2, 3] = Field(
        description="Slope of peak exercise ST segment (1-3)", examples=[3]
    )
    ca: Literal[0, 1, 2, 3] = Field(
        description="Number of major vessels coloured by fluoroscopy (0-3)", examples=[0]
    )
    thal: Literal[3, 6, 7] = Field(
        description="3 = normal, 6 = fixed defect, 7 = reversible defect", examples=[6]
    )


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction: Literal[0, 1]
    label: Literal["No disease", "Disease"]
    confidence: float = Field(description="Predicted probability of the predicted class")
    risk_probability: float = Field(description="Predicted probability of disease (class 1)")
    threshold: float
    model_version: str
    request_id: str


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    threshold: float
    feature_schema: dict
    metrics: dict
    sklearn_version: str
    git_sha: str
    timestamp: str
