"""FastAPI cloud agent service (Component 4 / 6 — agent layer).

Endpoints
---------
* ``POST /recommend`` — :class:`Alert` JSON → :class:`Recommendation` JSON
* ``GET  /healthz``   — service health (rule §22)

Wiring on startup:

* Build :class:`tools.retrieve_knowledge.RetrieveKnowledgeTool` (TF-IDF by default in tests; Chroma + sentence-transformers when an index exists and ``rag_retrieval`` is ``auto``/``chroma``).
* Build the other three tools (mock / rule-based backends as today).
* Build an :class:`agent.orchestration.llm_client.LlmClient` from ``llm_backend`` (``mock`` or ``ollama``).
* Build :class:`agent.workflows.react.ReActAgent` and stash on app.state.

中文说明
--------
云端智能体 HTTP 入口：读入边缘告警 JSON，执行 ReAct + RAG + 工具，返回含
``recommended_action``、``reasoning_trace``、``knowledge_sources`` 等字段的结构化推荐。
"""
from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from agent.orchestration.llm_client import build_llm_client
from agent.workflows.react import ReActAgent
from api.schemas import (
    Alert,
    ErrorResponse,
    HealthResponse,
    Recommendation,
)
from configs.settings import get_settings
from tools.escalate_alert import EscalateAlertTool
from tools.estimate_rul import EstimateRulTool
from tools.retrieve_knowledge import build_default_tool as build_retrieve_tool
from tools.system_history import SystemHistoryTool
from utils.logging_config import get_logger

log = get_logger(__name__)

SERVICE_NAME = "agent"


def _build_react_agent(settings):
    """Wire all tools + LLM + workflow into a single :class:`ReActAgent`."""

    retrieve = build_retrieve_tool(settings.knowledge_base_dir, settings=settings)
    history = SystemHistoryTool()
    rul = EstimateRulTool()
    escalate = EscalateAlertTool()
    llm = build_llm_client(settings=settings)
    return ReActAgent(
        tools={
            retrieve.name: retrieve,
            history.name: history,
            rul.name: rul,
            escalate.name: escalate,
        },
        llm=llm,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    log.info(
        "agent_service_starting",
        extra={"knowledge_base_dir": str(settings.knowledge_base_dir)},
    )
    if not settings.knowledge_base_dir.exists():
        log.warning(
            "agent_service_kb_missing",
            extra={"path": str(settings.knowledge_base_dir)},
        )
        app.state.react_agent = None
        app.state.startup_error = (
            f"knowledge_base_dir missing: {settings.knowledge_base_dir}"
        )
    else:
        try:
            app.state.react_agent = _build_react_agent(settings)
            app.state.startup_error = None
            log.info(
                "agent_service_started",
                extra={
                    "llm_backend": settings.llm_backend,
                    "rag_retrieval": settings.rag_retrieval,
                },
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("agent_service_startup_failed")
            app.state.react_agent = None
            app.state.startup_error = str(exc)
    yield
    log.info("agent_service_stopped")


app = FastAPI(
    title="AgentPV Cloud Agent",
    version="0.1.0",
    lifespan=lifespan,
)


def _error(
    *, status_code: int, error_code: str, message: str, trace_id: str
) -> JSONResponse:
    payload = ErrorResponse(
        error_code=error_code, message=message, trace_id=trace_id
    ).model_dump(mode="json", exclude_none=True)
    return JSONResponse(status_code=status_code, content=payload)


@app.get("/healthz", response_model=HealthResponse)
async def healthz(request: Request) -> HealthResponse:
    """Service health (rule §22)."""

    react_agent = getattr(request.app.state, "react_agent", None)
    status_str = "ok" if react_agent is not None else "degraded"
    return HealthResponse(status=status_str, service=SERVICE_NAME)


@app.post(
    "/recommend",
    response_model=Recommendation,
    responses={
        503: {"model": ErrorResponse, "description": "Agent not initialized"},
        422: {"model": ErrorResponse, "description": "Validation error"},
    },
)
async def recommend(alert: Alert, request: Request) -> Recommendation:
    """Run the ReAct loop on an alert and return a structured recommendation."""

    trace_id = uuid.uuid4().hex
    react_agent: ReActAgent | None = getattr(request.app.state, "react_agent", None)
    if react_agent is None:
        startup_error = getattr(request.app.state, "startup_error", "unknown") or "unknown"
        log.warning(
            "agent_recommend_unavailable",
            extra={"trace_id": trace_id, "reason": startup_error},
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Agent not initialized: {startup_error}",
        )

    log.info(
        "agent_recommend_request",
        extra={
            "trace_id": trace_id,
            "system_id": alert.system_id,
            "fault_class": alert.fault_class,
            "severity": alert.severity.value,
        },
    )
    recommendation = await react_agent.run(alert)
    log.info(
        "agent_recommend_response",
        extra={
            "trace_id": trace_id,
            "n_steps": len(recommendation.reasoning_trace),
            "confidence": recommendation.confidence.value,
            "urgency": recommendation.urgency.value,
            "n_sources": len(recommendation.knowledge_sources),
        },
    )
    return recommendation
