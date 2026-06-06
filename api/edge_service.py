"""FastAPI edge inference service (Component 6 — edge layer).

Endpoints
---------
* ``POST /predict``   — :class:`SensorWindow` JSON → :class:`Alert` JSON
* ``GET  /healthz``   — service health (rule §22)
* ``GET  /metrics``   — last benchmark stats per loaded model

Model loading is **lazy + per system_type**: PV and BESS ONNX files are
loaded once on startup and cached in memory. The ``/predict`` endpoint
routes the request to the right classifier based on
``SensorWindow.system_type``. If the file is missing, the service still
starts in ``degraded`` mode so a partial deployment (e.g. PV only) works.

中文说明
--------
边缘 HTTP 服务：接收滑动窗口传感器向量，经 ONNX 推理后输出符合
``docs/alert_schema.json`` 的结构化告警，供云端智能体或编排器消费。
"""
from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from api.schemas import (
    Alert,
    ErrorResponse,
    HealthResponse,
    SensorWindow,
    SystemType,
)
from configs.settings import get_settings
from inference.onnx_runner import OnnxClassifier
from utils.logging_config import get_logger

log = get_logger(__name__)

SERVICE_NAME = "edge"


# ---------------------------------------------------------------------------
# Model registry — populated on startup
# ---------------------------------------------------------------------------


def _resolve_model_paths(artifacts_dir: Path) -> dict[SystemType, Path]:
    """Locate the ONNX file for each system type."""

    return {
        SystemType.PV: artifacts_dir / "cnn1d_pv.onnx",
        SystemType.BESS: artifacts_dir / "cnn1d_bess.onnx",
    }


def _load_classifiers(artifacts_dir: Path) -> dict[SystemType, OnnxClassifier]:
    """Load whichever ONNX files exist; return a registry keyed by system type."""

    registry: dict[SystemType, OnnxClassifier] = {}
    for system_type, path in _resolve_model_paths(artifacts_dir).items():
        if not path.exists():
            log.warning(
                "edge_model_missing",
                extra={"system_type": system_type.value, "path": str(path)},
            )
            continue
        registry[system_type] = OnnxClassifier(path)
    return registry


# ---------------------------------------------------------------------------
# FastAPI lifespan — replaces deprecated on_event hooks (FastAPI 0.110+).
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    artifacts_dir = settings.artifacts_dir
    log.info(
        "edge_service_starting",
        extra={"artifacts_dir": str(artifacts_dir)},
    )
    registry = _load_classifiers(artifacts_dir)
    app.state.classifiers = registry
    app.state.artifacts_dir = artifacts_dir
    log.info(
        "edge_service_started",
        extra={
            "loaded_systems": [s.value for s in registry.keys()],
        },
    )
    yield
    log.info("edge_service_stopped")


app = FastAPI(
    title="AgentPV Edge Inference",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_classifier(request: Request, system_type: SystemType) -> OnnxClassifier:
    """Return the loaded classifier or raise 503 if unavailable."""

    classifiers: dict[SystemType, OnnxClassifier] = request.app.state.classifiers
    classifier = classifiers.get(system_type)
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"No model loaded for system_type={system_type.value}",
        )
    return classifier


def _error_response(
    *,
    status_code: int,
    error_code: str,
    message: str,
    trace_id: str,
    retry_after_s: int | None = None,
) -> JSONResponse:
    """Build a structured ErrorResponse JSON reply."""

    payload = ErrorResponse(
        error_code=error_code,
        message=message,
        trace_id=trace_id,
        retry_after_s=retry_after_s,
    ).model_dump(mode="json", exclude_none=True)
    return JSONResponse(status_code=status_code, content=payload)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/healthz", response_model=HealthResponse)
async def healthz(request: Request) -> HealthResponse:
    """Health check (rule §22 — every service has /healthz)."""

    classifiers: dict[SystemType, OnnxClassifier] = request.app.state.classifiers
    expected = {SystemType.PV, SystemType.BESS}
    status_str = "ok" if expected.issubset(classifiers.keys()) else "degraded"
    return HealthResponse(status=status_str, service=SERVICE_NAME)


@app.post(
    "/predict",
    response_model=Alert,
    responses={
        503: {"model": ErrorResponse, "description": "Model not loaded"},
        422: {"model": ErrorResponse, "description": "Validation error"},
    },
)
async def predict(window: SensorWindow, request: Request) -> Alert:
    """Run edge inference on a single sensor window."""

    trace_id = str(uuid.uuid4())
    log.info(
        "edge_predict_request",
        extra={
            "trace_id": trace_id,
            "system_id": window.system_id,
            "system_type": window.system_type.value,
            "window_size": window.window_size,
        },
    )
    classifier = _get_classifier(request, window.system_type)

    started = datetime.now(UTC)
    alert = classifier.predict_window(window)
    elapsed_ms = (datetime.now(UTC) - started).total_seconds() * 1000.0

    log.info(
        "edge_predict_response",
        extra={
            "trace_id": trace_id,
            "fault_class": alert.fault_class,
            "severity": alert.severity.value,
            "confidence": alert.confidence,
            "elapsed_ms": round(elapsed_ms, 3),
        },
    )
    return alert


@app.get("/metrics")
async def metrics(request: Request) -> dict:
    """Latency benchmark per loaded model (synthetic, n=50).

    Endpoint exists so the dashboard / orchestrator can surface real
    perf-budget numbers without bundling its own benchmark code.
    """

    classifiers: dict[SystemType, OnnxClassifier] = request.app.state.classifiers
    out: dict[str, dict] = {}
    for st, clf in classifiers.items():
        out[st.value] = clf.benchmark(n=50).to_dict()
    return out
