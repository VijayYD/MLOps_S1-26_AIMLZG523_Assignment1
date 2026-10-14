"""FastAPI serving app for the heart disease risk model.

Loads the packaged joblib + metadata artifact directly (docs/DECISIONS.md
D5.1) -- no MLflow server needed at serving time.
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from heart_mlops.api.logging_config import configure_logging, log_request
from heart_mlops.api.metrics import (
    INFERENCE_ERRORS,
    PREDICTION_COUNT,
    REQUEST_COUNT,
    REQUEST_LATENCY_SECONDS,
    RISK_PROBABILITY,
)
from heart_mlops.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PatientFeatures,
    PredictionResponse,
    ReadyResponse,
)
from heart_mlops.config import MODEL_FEATURE_COLUMNS
from heart_mlops.models.registry import load_packaged_model

logger = configure_logging(level=os.environ.get("LOG_LEVEL", "INFO"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        pipeline, metadata = load_packaged_model()
    except FileNotFoundError:
        pipeline, metadata = None, None

    if metadata is not None:
        # PREDICTION_THRESHOLD lets ops override the trained threshold
        # (e.g. via the k8s ConfigMap) without retraining/repackaging --
        # the packaged model_metadata.json value is still the default.
        override = os.environ.get("PREDICTION_THRESHOLD")
        if override is not None:
            metadata = dict(metadata, threshold=float(override))
            logger.info("Threshold overridden via PREDICTION_THRESHOLD: %s", override)

    app.state.pipeline = pipeline
    app.state.metadata = metadata
    yield


app = FastAPI(
    title="Heart Disease Risk API",
    description="Screening-support risk classifier -- not for clinical use. See models/MODEL_CARD.md.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe -- always OK if the process is up."""
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse)
def ready(request: Request) -> ReadyResponse:
    """Readiness probe -- OK only once the model artifact has loaded."""
    if request.app.state.pipeline is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return ReadyResponse(status="ready")


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    metadata = request.app.state.metadata
    if metadata is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return ModelInfoResponse(
        model_name=metadata["model_name"],
        model_version=str(metadata["model_version"]),
        threshold=metadata["threshold"],
        feature_schema=metadata["feature_schema"],
        metrics=metadata["metrics"],
        sklearn_version=metadata["sklearn_version"],
        git_sha=metadata["git_sha"],
        timestamp=metadata["timestamp"],
    )


def _features_to_row(features: PatientFeatures) -> pd.DataFrame:
    # astype float64: the fitted preprocessor saw float64 for every column
    # at training time (pandas upcasts a mix of int64/float64 columns to
    # float64 when sklearn's ColumnTransformer extracts them into one
    # array), but Pydantic's Literal[int] fields (sex, cp, fbs, ...) give
    # plain Python ints here, which pandas would otherwise store as int64
    # -- a dtype sklearn's imputer correctly refuses to accept as a
    # mismatch from what it was fit on.
    return pd.DataFrame([features.model_dump()])[MODEL_FEATURE_COLUMNS].astype("float64")


def _risk_to_response(
    risk_probability: float, metadata: dict, request_id: str
) -> PredictionResponse:
    threshold = metadata["threshold"]
    prediction = int(risk_probability >= threshold)
    label = "Disease" if prediction == 1 else "No disease"
    confidence = risk_probability if prediction == 1 else 1.0 - risk_probability
    return PredictionResponse(
        prediction=prediction,
        label=label,
        confidence=confidence,
        risk_probability=risk_probability,
        threshold=threshold,
        model_version=str(metadata["model_version"]),
        request_id=request_id,
    )


def _require_loaded_model(request: Request, endpoint: str) -> tuple:
    pipeline = request.app.state.pipeline
    metadata = request.app.state.metadata
    if pipeline is None:
        REQUEST_COUNT.labels(method="POST", endpoint=endpoint, status="503").inc()
        INFERENCE_ERRORS.inc()
        raise HTTPException(status_code=503, detail="model not loaded")
    return pipeline, metadata


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures, request: Request) -> PredictionResponse:
    start = time.perf_counter()
    request_id = str(uuid.uuid4())
    pipeline, metadata = _require_loaded_model(request, "/predict")

    row = _features_to_row(features)
    try:
        risk_probability = float(pipeline.predict_proba(row)[0, 1])
    except Exception as exc:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="500").inc()
        INFERENCE_ERRORS.inc()
        raise HTTPException(status_code=500, detail="inference failed") from exc

    response = _risk_to_response(risk_probability, metadata, request_id)

    latency_ms = (time.perf_counter() - start) * 1000
    REQUEST_COUNT.labels(method="POST", endpoint="/predict", status="200").inc()
    REQUEST_LATENCY_SECONDS.labels(endpoint="/predict").observe(latency_ms / 1000)
    PREDICTION_COUNT.labels(predicted_class=str(response.prediction)).inc()
    RISK_PROBABILITY.observe(risk_probability)
    log_request(
        logger,
        request_id=request_id,
        path="/predict",
        status_code=200,
        latency_ms=latency_ms,
        predicted_class=response.prediction,
        model_version=response.model_version,
    )

    return response


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(batch: BatchPredictRequest, request: Request) -> BatchPredictResponse:
    start = time.perf_counter()
    request_id = str(uuid.uuid4())
    pipeline, metadata = _require_loaded_model(request, "/predict/batch")

    rows = pd.concat([_features_to_row(f) for f in batch.records], ignore_index=True)
    try:
        risk_probabilities = pipeline.predict_proba(rows)[:, 1]
    except Exception as exc:
        REQUEST_COUNT.labels(method="POST", endpoint="/predict/batch", status="500").inc()
        INFERENCE_ERRORS.inc()
        raise HTTPException(status_code=500, detail="inference failed") from exc

    responses = [
        _risk_to_response(float(p), metadata, f"{request_id}-{i}")
        for i, p in enumerate(risk_probabilities)
    ]

    latency_ms = (time.perf_counter() - start) * 1000
    REQUEST_COUNT.labels(method="POST", endpoint="/predict/batch", status="200").inc()
    REQUEST_LATENCY_SECONDS.labels(endpoint="/predict/batch").observe(latency_ms / 1000)
    for response, risk_probability in zip(responses, risk_probabilities, strict=True):
        PREDICTION_COUNT.labels(predicted_class=str(response.prediction)).inc()
        RISK_PROBABILITY.observe(float(risk_probability))
    log_request(
        logger,
        request_id=request_id,
        path="/predict/batch",
        status_code=200,
        latency_ms=latency_ms,
        predicted_class=None,
        model_version=responses[0].model_version,
    )

    return BatchPredictResponse(predictions=responses, request_id=request_id)


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
