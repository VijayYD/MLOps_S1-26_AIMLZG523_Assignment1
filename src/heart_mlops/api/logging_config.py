"""Structured JSON logging for the API.

Deliberately logs request id, path, status, latency, predicted class and
model version per request -- and deliberately does **not** log the raw
patient feature values. Patient-level clinical data (age, blood pressure,
cholesterol, etc.) is sensitive; nothing about serving predictions
requires it to sit in log files, which are typically retained far longer
and read by a wider audience than the request path itself. See
docs/DECISIONS.md D6.x.
"""

from __future__ import annotations

import logging
import sys

from pythonjsonlogger import jsonlogger

LOGGER_NAME = "heart_mlops.api"


def configure_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s", rename_fields={"asctime": "timestamp"}
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def log_request(
    logger: logging.Logger,
    *,
    request_id: str,
    path: str,
    status_code: int,
    latency_ms: float,
    predicted_class: int | None = None,
    model_version: str | None = None,
) -> None:
    """Logs request metadata only -- never the request body (patient
    features)."""
    logger.info(
        "request",
        extra={
            "request_id": request_id,
            "path": path,
            "status_code": status_code,
            "latency_ms": round(latency_ms, 2),
            "predicted_class": predicted_class,
            "model_version": model_version,
        },
    )
