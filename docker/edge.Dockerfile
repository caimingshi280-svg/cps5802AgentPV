# syntax=docker/dockerfile:1
# Edge inference service (Component 2 / 6).
# Stack: ONNX Runtime (CPU only) + FastAPI.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# `curl` is needed for the docker-compose healthcheck against /healthz.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Install pinned deps (rule §23). Edge needs:
#   - ml.txt        (numpy / onnxruntime / pandas)
#   - agent.txt     (fastapi / uvicorn / httpx — shared HTTP stack)
COPY requirements ./requirements
RUN pip install --upgrade pip \
 && pip install -r requirements/ml.txt \
 && pip install -r requirements/agent.txt

COPY api ./api
COPY configs ./configs
COPY inference ./inference
COPY quantization ./quantization
COPY simulation ./simulation
COPY utils ./utils

EXPOSE 8000

CMD ["uvicorn", "api.edge_service:app", "--host", "0.0.0.0", "--port", "8000"]
