# Base matches the lab's Python exactly (3.12.9) -- see docs/DECISIONS.md.
# linux/amd64 explicitly: both the Mac (Intel) and the lab are amd64, so no
# multi-arch/buildx is needed, but pinning the platform avoids any surprise
# if this is ever built on an arm64 machine (e.g. Apple Silicon) by mistake.

# ---- builder: resolve + download wheels, nothing else ----
FROM --platform=linux/amd64 python:3.12.9-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ---- runtime: only what's needed to serve ----
FROM --platform=linux/amd64 python:3.12.9-slim AS runtime

ARG GIT_SHA=unknown
ARG VERSION=0.1.0
LABEL org.opencontainers.image.title="Heart Disease Risk API" \
      org.opencontainers.image.description="Screening-support risk classifier -- not for clinical use" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.source="https://github.com/VijayYD/MLOps_S1-26_AIMLZG523_Assignment1"

RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin appuser

WORKDIR /app

COPY --from=builder /root/.local /home/appuser/.local
COPY src/ src/
COPY models/heart_model.joblib models/model_metadata.json models/MODEL_CARD.md models/

RUN chown -R appuser:appuser /app

ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    API_PORT=8000 \
    UVICORN_WORKERS=2

USER appuser

EXPOSE 8000

# No curl in the slim image on purpose (keeps the image to runtime deps
# only) -- urllib from the stdlib does the same job for a healthcheck.
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"API_PORT\"]}/health', timeout=2)" || exit 1

CMD ["sh", "-c", "uvicorn heart_mlops.api.main:app --host 0.0.0.0 --port ${API_PORT} --workers ${UVICORN_WORKERS}"]
