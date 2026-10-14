"""Tests for the FastAPI app (src/heart_mlops/api/main.py).

Uses a tiny fitted pipeline fixture, not the committed models/heart_model.joblib,
so these tests stay fast and independent of whatever model happens to be
packaged at the time -- see the api_client fixture below.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from heart_mlops.config import MODEL_FEATURE_COLUMNS
from heart_mlops.features.preprocess import build_pipeline

VALID_PAYLOAD = {
    "age": 63,
    "sex": 1,
    "cp": 1,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": 6,
}


@pytest.fixture
def api_client(tiny_cleaned_df, monkeypatch):
    pipeline = build_pipeline("logreg")
    X = tiny_cleaned_df[MODEL_FEATURE_COLUMNS]
    y = tiny_cleaned_df["target"]
    pipeline.fit(X, y)
    metadata = {
        "model_name": "logreg",
        "model_version": "1",
        "threshold": 0.25,
        "feature_schema": {"numeric": [], "categorical": []},
        "metrics": {"accuracy": 0.8},
        "sklearn_version": "1.5.2",
        "git_sha": "testsha",
        "timestamp": "2026-01-01T00:00:00+00:00",
    }

    monkeypatch.setattr("heart_mlops.api.main.load_packaged_model", lambda: (pipeline, metadata))

    from heart_mlops.api.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
def api_client_model_not_loaded(monkeypatch):
    def _raise():
        raise FileNotFoundError("no packaged model")

    monkeypatch.setattr("heart_mlops.api.main.load_packaged_model", _raise)

    from heart_mlops.api.main import app

    with TestClient(app) as client:
        yield client


def test_health_always_ok(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_when_model_loaded(api_client):
    response = api_client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_ready_returns_503_when_model_not_loaded(api_client_model_not_loaded):
    response = api_client_model_not_loaded.get("/ready")
    assert response.status_code == 503


def test_model_info(api_client):
    response = api_client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "logreg"
    assert body["threshold"] == 0.25


def test_predict_happy_path(api_client):
    response = api_client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("No disease", "Disease")
    assert 0.0 <= body["confidence"] <= 1.0
    assert 0.0 <= body["risk_probability"] <= 1.0
    assert body["threshold"] == 0.25
    assert body["model_version"] == "1"
    assert "request_id" in body


def test_predict_label_matches_prediction(api_client):
    response = api_client.post("/predict", json=VALID_PAYLOAD)
    body = response.json()
    expected_label = "Disease" if body["prediction"] == 1 else "No disease"
    assert body["label"] == expected_label


def test_predict_returns_503_when_model_not_loaded(api_client_model_not_loaded):
    response = api_client_model_not_loaded.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 503


def test_predict_batch_happy_path(api_client):
    response = api_client.post("/predict/batch", json={"records": [VALID_PAYLOAD, VALID_PAYLOAD]})
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2
    for pred in body["predictions"]:
        assert pred["prediction"] in (0, 1)
        assert pred["request_id"].startswith(body["request_id"])


def test_predict_batch_rejects_empty_list(api_client):
    response = api_client.post("/predict/batch", json={"records": []})
    assert response.status_code == 422


def test_predict_batch_rejects_over_max_length(api_client):
    response = api_client.post("/predict/batch", json={"records": [VALID_PAYLOAD] * 101})
    assert response.status_code == 422


def test_predict_batch_rejects_invalid_record_inside_batch(api_client):
    bad_record = dict(VALID_PAYLOAD, sex=2)
    response = api_client.post("/predict/batch", json={"records": [VALID_PAYLOAD, bad_record]})
    assert response.status_code == 422


def test_predict_batch_returns_503_when_model_not_loaded(api_client_model_not_loaded):
    response = api_client_model_not_loaded.post("/predict/batch", json={"records": [VALID_PAYLOAD]})
    assert response.status_code == 503


def test_prediction_threshold_env_var_overrides_packaged_threshold(tiny_cleaned_df, monkeypatch):
    pipeline = build_pipeline("logreg")
    X = tiny_cleaned_df[MODEL_FEATURE_COLUMNS]
    y = tiny_cleaned_df["target"]
    pipeline.fit(X, y)
    metadata = {
        "model_name": "logreg",
        "model_version": "1",
        "threshold": 0.25,
        "feature_schema": {"numeric": [], "categorical": []},
        "metrics": {"accuracy": 0.8},
        "sklearn_version": "1.5.2",
        "git_sha": "testsha",
        "timestamp": "2026-01-01T00:00:00+00:00",
    }
    monkeypatch.setattr("heart_mlops.api.main.load_packaged_model", lambda: (pipeline, metadata))
    monkeypatch.setenv("PREDICTION_THRESHOLD", "0.9")

    from heart_mlops.api.main import app

    with TestClient(app) as client:
        response = client.get("/model-info")

    assert response.json()["threshold"] == 0.9


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("sex", 2),  # outside Literal[0, 1]
        ("cp", 5),  # outside Literal[1, 2, 3, 4]
        ("age", -1),  # below ge=1
        ("age", 500),  # above le=120
        ("thal", 4),  # outside Literal[3, 6, 7]
        ("trestbps", 1000),  # above le=260
    ],
)
def test_predict_rejects_out_of_range_values(api_client, field, bad_value):
    payload = dict(VALID_PAYLOAD, **{field: bad_value})
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_missing_field(api_client):
    payload = dict(VALID_PAYLOAD)
    del payload["age"]
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_wrong_type(api_client):
    payload = dict(VALID_PAYLOAD, age="not-a-number")
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 422


def test_metrics_endpoint_exposes_custom_metrics(api_client):
    api_client.post("/predict", json=VALID_PAYLOAD)  # generate at least one data point
    response = api_client.get("/metrics")
    assert response.status_code == 200
    body = response.text
    for metric_name in [
        "http_requests_total",
        "http_request_duration_seconds",
        "predictions_total",
        "risk_probability",
        "inference_errors_total",
    ]:
        assert metric_name in body


def test_predict_does_not_log_raw_patient_features(api_client, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="heart_mlops.api"):
        api_client.post("/predict", json=VALID_PAYLOAD)

    logged_text = " ".join(record.getMessage() for record in caplog.records)
    # None of the distinctive raw feature values should appear in log output.
    assert "233" not in logged_text  # chol
    assert "145" not in logged_text  # trestbps


def test_openapi_docs_available(api_client):
    response = api_client.get("/docs")
    assert response.status_code == 200
    openapi = api_client.get("/openapi.json")
    assert openapi.status_code == 200
    assert "/predict" in openapi.json()["paths"]
