"""Interactive fault-injection helper for the dashboard (Component 7 demo).

The orchestrator (`python -m orchestrator`) drives long-running stochastic
fault generation. The *demo* path implemented here lets a dashboard user
trigger **one** specific fault on demand:

1. Generate a clean window via :class:`simulation.pv_simulator.PVSimulator`
   or :class:`simulation.battery_simulator.BatterySimulator`.
2. Inject the chosen fault label (or none for the *Normal* class) via
   :func:`simulation.fault_injector.inject_fault`.
3. POST the resulting :class:`api.schemas.SensorWindow` to the running
   ``edge_service`` (``POST /predict``) to obtain an :class:`Alert`.
4. If the alert severity warrants it (and ``skip_agent`` is False), POST
   the alert to the running ``agent_service`` (``POST /recommend``) to
   obtain a :class:`Recommendation` enriched with RAG knowledge sources.
5. Wrap everything into a single :class:`OrchestratorEvent` and append
   to the dashboard's JSONL feed so the regular tabs immediately pick it
   up after a refresh.

The function is **pure synchronous** (uses :class:`httpx.Client`) so it can
run inside Streamlit's request thread without needing an event loop. Tests
inject a :class:`httpx.MockTransport` via the ``http_client`` parameter.
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from api.schemas import (
    BESS_FAULT_CLASSES,
    PV_FAULT_CLASSES,
    Alert,
    OperatingCondition,
    OrchestratorEvent,
    Recommendation,
    SensorWindow,
    Severity,
    SystemType,
)
from orchestrator.event_log import JsonlEventWriter, make_default_path
from simulation.battery_simulator import BESS_FEATURE_NAMES, BatterySimulator
from simulation.fault_injector import inject_fault
from simulation.pv_simulator import PV_FEATURE_NAMES, PVSimulator
from utils.logging_config import get_logger

log = get_logger(__name__)

_PV_NORMAL = "PV_Normal"
_BESS_NORMAL = "BESS_Normal"

_AGENT_TRIGGER_SEVERITIES: frozenset[Severity] = frozenset(
    {Severity.WARNING, Severity.CRITICAL}
)


# ---------------------------------------------------------------------------
# Result envelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InjectionResult:
    """Outcome of a single demo injection.

    The ``event`` field is always populated; a non-``None`` ``edge_error``
    means the edge service refused the request and ``event.alert`` is
    ``None``. ``agent_error`` is only meaningful when the agent was
    actually called (severity ≥ warning, ``skip_agent=False``).
    """

    event: OrchestratorEvent
    edge_ms: float | None
    agent_ms: float | None
    edge_error: str | None
    agent_error: str | None
    agent_called: bool

    @property
    def ok(self) -> bool:
        if self.edge_error is not None:
            return False
        if self.agent_called and self.agent_error is not None:
            return False
        return self.event.alert is not None


# ---------------------------------------------------------------------------
# Pure validation
# ---------------------------------------------------------------------------


def fault_choices_for(system_type: SystemType) -> tuple[str, ...]:
    """Return the catalogue of valid labels (incl. *Normal*) for one system."""

    if system_type is SystemType.PV:
        return PV_FAULT_CLASSES
    return BESS_FAULT_CLASSES


def validate_request(
    *,
    system_type: SystemType,
    fault_class: str,
    system_id: str,
    window_size: int,
) -> None:
    """Pre-flight check; raises :class:`ValueError` on bad input."""

    catalogue = fault_choices_for(system_type)
    if fault_class not in catalogue:
        raise ValueError(
            f"fault_class={fault_class!r} not valid for {system_type.value}; "
            f"choose one of {list(catalogue)}"
        )
    if not system_id or not system_id.strip():
        raise ValueError("system_id must be a non-empty string")
    if not 10 <= window_size <= 600:
        raise ValueError(f"window_size must be 10..600, got {window_size}")


# ---------------------------------------------------------------------------
# Window synthesis (deterministic w.r.t. seed)
# ---------------------------------------------------------------------------


def _build_window(
    *,
    system_type: SystemType,
    fault_class: str,
    operating_condition: OperatingCondition,
    system_id: str,
    window_size: int,
    seed: int,
    now_fn: Callable[[], datetime],
) -> tuple[SensorWindow, str]:
    """Deterministic window generation given (system, fault, op, seed)."""

    rng = np.random.default_rng(seed)
    if system_type is SystemType.PV:
        sim = PVSimulator(seed=seed)
        clean = sim.simulate(
            condition=operating_condition, window_size=window_size
        )
        feature_names = list(PV_FEATURE_NAMES)
        normal_label = _PV_NORMAL
    else:
        sim = BatterySimulator(seed=seed)
        clean = sim.simulate(
            condition=operating_condition, window_size=window_size
        )
        feature_names = list(BESS_FEATURE_NAMES)
        normal_label = _BESS_NORMAL

    if fault_class != normal_label:
        arr = inject_fault(clean, label=fault_class, rng=rng)
    else:
        arr = clean

    window = SensorWindow(
        timestamp_start=now_fn(),
        system_id=system_id,
        system_type=system_type,
        sample_rate_hz=1.0,
        window_size=window_size,
        feature_names=feature_names,
        values=arr.astype(float).tolist(),
        operating_condition=operating_condition,
        injected_fault=None if fault_class == normal_label else fault_class,
    )
    return window, fault_class


# ---------------------------------------------------------------------------
# Live edge / agent calls
# ---------------------------------------------------------------------------


def _post_json(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    *,
    trace_id: str,
) -> tuple[dict | None, str | None, float]:
    """POST JSON; return ``(payload_or_None, error_str_or_None, elapsed_ms)``."""

    t0 = time.perf_counter()
    try:
        resp = client.post(
            url, json=payload, headers={"x-trace-id": trace_id}
        )
    except httpx.HTTPError as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return None, f"{type(exc).__name__}: {exc}" or type(exc).__name__, elapsed_ms

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if resp.status_code != 200:
        snippet = resp.text[:200]
        return None, f"HTTP {resp.status_code}: {snippet}", elapsed_ms
    try:
        body = resp.json()
    except ValueError as exc:
        return None, f"non-JSON body: {exc}", elapsed_ms
    if not isinstance(body, dict):
        return None, f"expected JSON object, got {type(body).__name__}", elapsed_ms
    return body, None, elapsed_ms


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def inject_fault_demo(
    *,
    system_type: SystemType,
    fault_class: str,
    operating_condition: OperatingCondition = OperatingCondition.HIGH_IRRADIANCE,
    system_id: str = "DEMO-PV-001",
    edge_url: str = "http://127.0.0.1:8000",
    agent_url: str = "http://127.0.0.1:8001",
    events_path: Path | None = None,
    seed: int = 4242,
    window_size: int = 60,
    skip_agent: bool = False,
    timeout_s: float = 120.0,
    http_client: httpx.Client | None = None,
    now_fn: Callable[[], datetime] | None = None,
    node_id: str = "demo-injector",
    persist: bool = True,
) -> InjectionResult:
    """Generate a labeled window, run it end-to-end, persist the event.

    Parameters
    ----------
    system_type, fault_class, operating_condition, system_id
        Window specification. ``fault_class`` must belong to the catalogue
        returned by :func:`fault_choices_for(system_type)`.
    edge_url, agent_url
        Base URLs of the running ``edge_service`` / ``agent_service``.
    events_path
        Where to append the resulting :class:`OrchestratorEvent`. Defaults
        to :func:`orchestrator.event_log.make_default_path`. Set
        ``persist=False`` to skip the JSONL write entirely.
    seed
        Deterministic seed feeding the simulator + fault injector.
    skip_agent
        If True, never call ``/recommend`` (forces the *edge_only* path,
        useful for demo'ing graceful degradation).
    timeout_s
        Per-request HTTP timeout. Default 120 s aligns with orchestrator
        ``--http-timeout`` and full-mode agent P95 (~9.5 s) under load
        (see ``reports/integration_eval.md``).
    http_client
        Override for unit tests (``httpx.MockTransport``). When None we
        build a fresh :class:`httpx.Client` and close it at the end.
    now_fn
        Override for the ``timestamp_start`` clock. Defaults to
        ``datetime.now(UTC)``; tests inject a frozen clock.
    persist
        If False, the event is built and returned but **not** appended to
        the JSONL — useful for dry-run previews.
    """

    validate_request(
        system_type=system_type,
        fault_class=fault_class,
        system_id=system_id,
        window_size=window_size,
    )
    now_fn = now_fn or (lambda: datetime.now(UTC))

    window, label = _build_window(
        system_type=system_type,
        fault_class=fault_class,
        operating_condition=operating_condition,
        system_id=system_id,
        window_size=window_size,
        seed=seed,
        now_fn=now_fn,
    )

    owns_client = http_client is None
    client = http_client or httpx.Client(timeout=timeout_s)
    try:
        edge_trace = uuid.uuid4().hex
        edge_payload, edge_error, edge_ms = _post_json(
            client,
            f"{edge_url.rstrip('/')}/predict",
            window.model_dump(mode="json"),
            trace_id=edge_trace,
        )

        alert: Alert | None = None
        if edge_payload is not None:
            try:
                alert = Alert.model_validate(edge_payload)
            except Exception as exc:  # noqa: BLE001
                edge_error = f"alert decode error: {exc}"

        agent_called = False
        agent_ms: float | None = None
        agent_error: str | None = None
        recommendation: Recommendation | None = None

        if alert is not None and not skip_agent:
            if alert.severity in _AGENT_TRIGGER_SEVERITIES:
                agent_called = True
                agent_trace = uuid.uuid4().hex
                agent_payload, agent_error, agent_ms = _post_json(
                    client,
                    f"{agent_url.rstrip('/')}/recommend",
                    alert.model_dump(mode="json"),
                    trace_id=agent_trace,
                )
                if agent_payload is not None:
                    try:
                        recommendation = Recommendation.model_validate(
                            agent_payload
                        )
                    except Exception as exc:  # noqa: BLE001
                        agent_error = f"recommendation decode error: {exc}"
    finally:
        if owns_client:
            client.close()

    event_error: str | None = None
    if edge_error is not None:
        event_error = f"edge_predict_failed: {edge_error}"
    elif agent_called and agent_error is not None:
        event_error = f"agent_recommend_failed: {agent_error}"

    event = OrchestratorEvent(
        event_id=uuid.uuid4().hex,
        timestamp=now_fn(),
        node_id=node_id,
        system_id=system_id,
        system_type=system_type,
        step_number=0,
        ground_truth_label=label,
        alert=alert,
        recommendation=recommendation,
        error=event_error,
        edge_elapsed_ms=edge_ms if edge_ms is not None else 0.0,
        agent_elapsed_ms=agent_ms,
    )

    if persist:
        target = events_path or make_default_path()
        writer = JsonlEventWriter(target)
        writer.append(event)
        log.info(
            "dashboard_demo_injection_persisted",
            extra={
                "events_path": str(target),
                "event_id": event.event_id,
                "fault_class": fault_class,
                "system_type": system_type.value,
                "edge_ms": edge_ms,
                "agent_ms": agent_ms,
                "ok": event_error is None and alert is not None,
            },
        )

    return InjectionResult(
        event=event,
        edge_ms=edge_ms,
        agent_ms=agent_ms,
        edge_error=edge_error,
        agent_error=agent_error,
        agent_called=agent_called,
    )
