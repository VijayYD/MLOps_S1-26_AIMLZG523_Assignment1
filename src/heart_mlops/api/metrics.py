"""Custom Prometheus metrics for the API.

Kept as a separate module (rather than defined inline in main.py) so
test/test_api.py and any future script can import and inspect them
without importing the whole FastAPI app.
"""

from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["endpoint"],
)

PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total predictions made, by predicted class",
    ["predicted_class"],
)

RISK_PROBABILITY = Histogram(
    "risk_probability",
    "Predicted probability of disease (class 1) -- a simple drift signal",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

INFERENCE_ERRORS = Counter(
    "inference_errors_total",
    "Total inference errors (model raised during predict_proba)",
)
