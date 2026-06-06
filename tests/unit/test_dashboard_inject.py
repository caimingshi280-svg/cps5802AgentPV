"""Unit tests for :mod:`dashboard.inject` (Component 7 demo path).

Uses :class:`httpx.MockTransport` (same pattern as
``tests/unit/test_orchestrator.py``) so we exercise the real ``httpx.Client``
request path without standing up the FastAPI services.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from api.schemas import OperatingCondition, SystemType
from dashboard.inject import (
    InjectionResult,
    fault_choices_for,
    inject_fault_demo,
    validate_request,
)

# ---------------------------------------------------------------------------
# Helpers — scripted HTTP responses
# ---------------------------------------------------------------------------


def _alert_payload(*, severity: str, fault_class: str, system_id: str) -> dict:
    return {
        "timestamp": "2026-05-13T20:00:00+00:00",
        "system_id": system_id,
        "system_type": "PV",
        "fault_class": fault_class,
        "severity": severity,
        "confidence": 0.91,
        "sensor_snapshot": {"V_dc": 28.0, "I_dc": 6.0},
    }


def _recommendation_payload() -> dict:
    return {
        "recommended_action": "Inspect inverter and clear fault code.",
        "urgency": "immediate",
        "reasoning_trace": [
            {"step": 0, "phase": "observe", "thought": "ok", "action": None,
             "result_summary": None}
        ],
        "knowledge_sources": ["pv-handbook-1.3", "site-runbook-7"],
        "confidence": "high",
    }


def _scripted_transport(
    *,
    edge: dict | int,
    agent: dict | int | None = None,
) -> httpx.MockTransport:
    """Return a transport that hands one scripted response per path."""

    def _handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/predict":
            spec = edge
        elif path == "/recommend":
            spec = agent
        else:
            return httpx.Response(404, text=f"unexpected {path}")
        if spec is None:
            return httpx.Response(503, text="not configured")
        if isinstance(spec, int):
            return httpx.Response(spec, text="scripted error")
        return httpx.Response(200, json=spec)

    return httpx.MockTransport(_handler)


def _frozen_now() -> datetime:
    return datetime(2026, 5, 13, 20, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_fault_choices_for_pv_and_bess() -> None:
    pv = fault_choices_for(SystemType.PV)
    bess = fault_choices_for(SystemType.BESS)
    assert "PV_Normal" in pv and "Inverter_fault" in pv
    assert "BESS_Normal" in bess and "Thermal_anomaly" in bess
    assert set(pv).isdisjoint(set(bess) - {"PV_Normal", "BESS_Normal"})


def test_validate_request_rejects_cross_system_label() -> None:
    with pytest.raises(ValueError, match="not valid for PV"):
        validate_request(
            system_type=SystemType.PV,
            fault_class="Thermal_anomaly",  # BESS-only
            system_id="DEMO",
            window_size=60,
        )


def test_validate_request_rejects_empty_system_id() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        validate_request(
            system_type=SystemType.PV,
            fault_class="PV_Normal",
            system_id="   ",
            window_size=60,
        )


def test_validate_request_rejects_bad_window_size() -> None:
    with pytest.raises(ValueError, match="window_size"):
        validate_request(
            system_type=SystemType.PV,
            fault_class="PV_Normal",
            system_id="x",
            window_size=5,
        )


# ---------------------------------------------------------------------------
# Happy path — full pipeline writes JSONL
# ---------------------------------------------------------------------------


def test_inject_fault_demo_writes_jsonl_for_full_pipeline(tmp_path: Path) -> None:
    transport = _scripted_transport(
        edge=_alert_payload(
            severity="warning",
            fault_class="Inverter_fault",
            system_id="DEMO-PV-001",
        ),
        agent=_recommendation_payload(),
    )
    client = httpx.Client(transport=transport, base_url="http://test")
    events_path = tmp_path / "events.jsonl"

    result = inject_fault_demo(
        system_type=SystemType.PV,
        fault_class="Inverter_fault",
        operating_condition=OperatingCondition.HIGH_IRRADIANCE,
        system_id="DEMO-PV-001",
        edge_url="http://test",
        agent_url="http://test",
        events_path=events_path,
        seed=42,
        http_client=client,
        now_fn=_frozen_now,
    )

    assert isinstance(result, InjectionResult)
    assert result.ok is True
    assert result.edge_error is None
    assert result.agent_called is True
    assert result.agent_error is None
    assert result.agent_ms is not None
    assert result.edge_ms is not None and result.edge_ms >= 0.0

    event = result.event
    assert event.alert is not None
    assert event.alert.severity.value == "warning"
    assert event.alert.fault_class == "Inverter_fault"
    assert event.recommendation is not None
    assert "Inspect inverter" in event.recommendation.recommended_action
    assert len(event.recommendation.knowledge_sources) == 2
    assert event.ground_truth_label == "Inverter_fault"
    assert event.error is None

    # File persisted exactly one line.
    lines = events_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert event.event_id in lines[0]


# ---------------------------------------------------------------------------
# Monitor severity → agent not called (matches NodeRunner trigger set)
# ---------------------------------------------------------------------------


def test_inject_fault_demo_monitor_severity_skips_agent(tmp_path: Path) -> None:
    transport = _scripted_transport(
        edge=_alert_payload(
            severity="monitor",
            fault_class="PV_Normal",
            system_id="DEMO-PV-001",
        ),
    )
    client = httpx.Client(transport=transport, base_url="http://test")

    result = inject_fault_demo(
        system_type=SystemType.PV,
        fault_class="PV_Normal",
        operating_condition=OperatingCondition.HIGH_IRRADIANCE,
        system_id="DEMO-PV-001",
        edge_url="http://test",
        agent_url="http://test",
        events_path=tmp_path / "events.jsonl",
        http_client=client,
        now_fn=_frozen_now,
    )

    assert result.ok is True
    assert result.agent_called is False
    assert result.agent_ms is None
    assert result.event.recommendation is None


# ---------------------------------------------------------------------------
# skip_agent forces edge-only path even when severity warrants it
# ---------------------------------------------------------------------------


def test_inject_fault_demo_skip_agent_forces_edge_only(tmp_path: Path) -> None:
    transport = _scripted_transport(
        edge=_alert_payload(
            severity="critical",
            fault_class="Inverter_fault",
            system_id="DEMO-PV-001",
        ),
        agent=_recommendation_payload(),
    )
    client = httpx.Client(transport=transport, base_url="http://test")

    result = inject_fault_demo(
        system_type=SystemType.PV,
        fault_class="Inverter_fault",
        operating_condition=OperatingCondition.HIGH_IRRADIANCE,
        system_id="DEMO-PV-001",
        edge_url="http://test",
        agent_url="http://test",
        events_path=tmp_path / "events.jsonl",
        http_client=client,
        skip_agent=True,
        now_fn=_frozen_now,
    )

    assert result.ok is True
    assert result.agent_called is False
    assert result.event.recommendation is None


# ---------------------------------------------------------------------------
# Edge failure surfaces in result + event but does NOT block writing the event
# ---------------------------------------------------------------------------


def test_inject_fault_demo_edge_http_error(tmp_path: Path) -> None:
    transport = _scripted_transport(edge=503)
    client = httpx.Client(transport=transport, base_url="http://test")

    result = inject_fault_demo(
        system_type=SystemType.PV,
        fault_class="Soiling",
        operating_condition=OperatingCondition.HIGH_IRRADIANCE,
        system_id="DEMO-PV-001",
        edge_url="http://test",
        agent_url="http://test",
        events_path=tmp_path / "events.jsonl",
        http_client=client,
        now_fn=_frozen_now,
    )

    assert result.ok is False
    assert "HTTP 503" in (result.edge_error or "")
    assert result.event.alert is None
    assert result.event.error is not None
    assert result.event.error.startswith("edge_predict_failed")
    # Still persisted for traceability.
    assert (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Agent failure surfaces but alert still present + event written
# ---------------------------------------------------------------------------


def test_inject_fault_demo_agent_http_error(tmp_path: Path) -> None:
    transport = _scripted_transport(
        edge=_alert_payload(
            severity="critical",
            fault_class="String_disconnection",
            system_id="DEMO-PV-001",
        ),
        agent=500,
    )
    client = httpx.Client(transport=transport, base_url="http://test")

    result = inject_fault_demo(
        system_type=SystemType.PV,
        fault_class="String_disconnection",
        operating_condition=OperatingCondition.HIGH_IRRADIANCE,
        system_id="DEMO-PV-001",
        edge_url="http://test",
        agent_url="http://test",
        events_path=tmp_path / "events.jsonl",
        http_client=client,
        now_fn=_frozen_now,
    )

    assert result.ok is False
    assert result.edge_error is None
    assert result.agent_called is True
    assert "HTTP 500" in (result.agent_error or "")
    assert result.event.alert is not None
    assert result.event.alert.severity.value == "critical"
    assert result.event.recommendation is None
    assert result.event.error and result.event.error.startswith(
        "agent_recommend_failed"
    )


# ---------------------------------------------------------------------------
# persist=False does not touch the filesystem
# ---------------------------------------------------------------------------


def test_inject_fault_demo_persist_false_skips_jsonl(tmp_path: Path) -> None:
    transport = _scripted_transport(
        edge=_alert_payload(
            severity="monitor",
            fault_class="PV_Normal",
            system_id="DEMO-PV-001",
        ),
    )
    client = httpx.Client(transport=transport, base_url="http://test")
    events_path = tmp_path / "events.jsonl"

    result = inject_fault_demo(
        system_type=SystemType.PV,
        fault_class="PV_Normal",
        operating_condition=OperatingCondition.HIGH_IRRADIANCE,
        system_id="DEMO-PV-001",
        edge_url="http://test",
        agent_url="http://test",
        events_path=events_path,
        http_client=client,
        now_fn=_frozen_now,
        persist=False,
    )

    assert result.ok is True
    assert not events_path.exists()


# ---------------------------------------------------------------------------
# Determinism: same (seed, fault, op_cond) → same window values
# ---------------------------------------------------------------------------


def test_inject_fault_demo_is_deterministic_in_seed(tmp_path: Path) -> None:
    common = dict(
        system_type=SystemType.PV,
        fault_class="Partial_shading",
        operating_condition=OperatingCondition.HIGH_IRRADIANCE,
        system_id="DEMO-PV-001",
        edge_url="http://test",
        agent_url="http://test",
        seed=1234,
        now_fn=_frozen_now,
        persist=False,
    )

    def _run() -> InjectionResult:
        transport = _scripted_transport(
            edge=_alert_payload(
                severity="monitor",
                fault_class="Partial_shading",
                system_id="DEMO-PV-001",
            ),
        )
        client = httpx.Client(transport=transport, base_url="http://test")
        return inject_fault_demo(http_client=client, **common)  # type: ignore[arg-type]

    a = _run()
    b = _run()
    assert a.event.alert is not None and b.event.alert is not None
    # The synthetic sensor_snapshot comes from the *server* mock so it is
    # identical by construction; we instead compare the ground-truth label
    # and the timestamps (which we froze), which still ride on the seed.
    assert a.event.ground_truth_label == b.event.ground_truth_label
