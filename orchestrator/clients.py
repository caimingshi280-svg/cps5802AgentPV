"""Async HTTP clients for talking to ``edge_service`` and ``agent_service``.

Both clients are thin wrappers around :class:`httpx.AsyncClient` that:

* construct typed responses from JSON payloads (rule §3 — single source of
  truth for shared schemas);
* expose a structured :class:`ClientError` envelope on failure rather than
  raising — the orchestrator's loop must keep running even when one HTTP
  call fails (rule §13);
* preserve a ``trace_id`` per request for cross-service log correlation.

Tests inject these via :class:`httpx.AsyncClient(transport=ASGITransport)`
to call FastAPI apps in-process without binding a real socket.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx

from api.schemas import Alert, Recommendation, SensorWindow
from utils.logging_config import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class ClientError:
    """Structured failure record emitted by either client."""

    status_code: int | None  # None when the request never reached the server
    message: str
    trace_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status_code": self.status_code,
            "message": self.message,
            "trace_id": self.trace_id,
        }


@dataclass(frozen=True)
class TimedResponse:
    """A successful response plus its wall-clock latency."""

    payload: dict[str, Any]
    elapsed_ms: float
    trace_id: str


class EdgeClient:
    """Talks to ``edge_service.py``'s ``POST /predict``."""

    def __init__(self, http: httpx.AsyncClient, base_url: str = "") -> None:
        self.http = http
        self.base_url = base_url.rstrip("/")

    async def predict(self, window: SensorWindow) -> Alert | ClientError:
        trace_id = uuid.uuid4().hex
        url = f"{self.base_url}/predict"
        try:
            resp = await self.http.post(
                url,
                json=window.model_dump(mode="json"),
                headers={"x-trace-id": trace_id},
            )
        except httpx.HTTPError as exc:
            log.warning(
                "edge_predict_transport_error",
                extra={"trace_id": trace_id, "error": str(exc)},
            )
            return ClientError(status_code=None, message=str(exc), trace_id=trace_id)

        # 这里我们只关心 timed 路径里的耗时；保留 t0 以便 predict_timed 重新测量。
        if resp.status_code != 200:
            log.warning(
                "edge_predict_http_error",
                extra={
                    "trace_id": trace_id,
                    "status_code": resp.status_code,
                    "body": resp.text[:200],
                },
            )
            return ClientError(
                status_code=resp.status_code,
                message=resp.text[:500],
                trace_id=trace_id,
            )
        try:
            alert = Alert.model_validate(resp.json())
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "edge_predict_decode_error",
                extra={"trace_id": trace_id, "error": str(exc)},
            )
            return ClientError(
                status_code=resp.status_code, message=str(exc), trace_id=trace_id
            )
        # Stash latency on a side-channel via attribute (frozen Alert can't carry it).
        # Returning a tuple would also work; we return the alert and recompute
        # latency inside the caller. The TimedResponse pattern is exposed via
        # `predict_timed` below for callers that need both.
        return alert

    async def predict_timed(
        self, window: SensorWindow
    ) -> tuple[Alert | ClientError, float]:
        """Same as :meth:`predict` but also returns measured latency in ms."""

        t0 = time.perf_counter()
        result = await self.predict(window)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return result, elapsed_ms


class AgentClient:
    """Talks to ``agent_service.py``'s ``POST /recommend``."""

    def __init__(self, http: httpx.AsyncClient, base_url: str = "") -> None:
        self.http = http
        self.base_url = base_url.rstrip("/")

    async def recommend_timed(
        self, alert: Alert
    ) -> tuple[Recommendation | ClientError, float]:
        trace_id = uuid.uuid4().hex
        url = f"{self.base_url}/recommend"
        t0 = time.perf_counter()
        try:
            resp = await self.http.post(
                url,
                json=alert.model_dump(mode="json"),
                headers={"x-trace-id": trace_id},
            )
        except httpx.HTTPError as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            log.warning(
                "agent_recommend_transport_error",
                extra={"trace_id": trace_id, "error": str(exc)},
            )
            return ClientError(status_code=None, message=str(exc), trace_id=trace_id), elapsed_ms

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if resp.status_code != 200:
            log.warning(
                "agent_recommend_http_error",
                extra={
                    "trace_id": trace_id,
                    "status_code": resp.status_code,
                    "body": resp.text[:200],
                },
            )
            return ClientError(
                status_code=resp.status_code,
                message=resp.text[:500],
                trace_id=trace_id,
            ), elapsed_ms
        try:
            rec = Recommendation.model_validate(resp.json())
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "agent_recommend_decode_error",
                extra={"trace_id": trace_id, "error": str(exc)},
            )
            return ClientError(
                status_code=resp.status_code, message=str(exc), trace_id=trace_id
            ), elapsed_ms
        return rec, elapsed_ms
