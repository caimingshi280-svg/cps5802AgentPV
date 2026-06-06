"""Unit tests for the orchestrator (Component 7 MVP).

These tests use a fake :class:`httpx.MockTransport` so we can exercise the
real :class:`EdgeClient` / :class:`AgentClient` request-response paths
without standing up FastAPI services. Integration with the real apps is
covered separately in ``tests/integration/test_orchestrator_e2e.py``.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest

from api.schemas import (
    AgentConfidence,
    Alert,
    OrchestratorEvent,
    ReasoningStep,
    Recommendation,
    Severity,
    SystemType,
    Urgency,
)
from orchestrator.clients import AgentClient, ClientError, EdgeClient
from orchestrator.event_log import JsonlEventWriter, summarize
from orchestrator.node_simulator import NodeConfig, NodeRunner
from orchestrator.orchestrator import Orchestrator, OrchestratorConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _alert_payload(*, severity: str, fault_class: str, system_id: str) -> dict:
    return {
        "timestamp": datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC).isoformat(),
        "system_id": system_id,
        "system_type": "PV",
        "fault_class": fault_class,
        "severity": severity,
        "confidence": 0.9,
        "sensor_snapshot": {"P_dc": 200.0},
    }


def _recommendation_payload() -> dict:
    return {
        "recommended_action": "[MOCK] inspect within 4h",
        "urgency": Urgency.IMMEDIATE.value,
        "reasoning_trace": [
            ReasoningStep(step=0, phase="observe", thought="ok").model_dump(mode="json")
        ],
        "knowledge_sources": ["doc-a"],
        "confidence": AgentConfidence.HIGH.value,
    }


def _scripted_transport(
    *,
    edge_responses: list[dict | int],
    agent_responses: list[dict | int],
) -> httpx.MockTransport:
    """Build a transport that pops one scripted response per call.

    Each entry in ``*_responses`` is either a dict (returned as 200 JSON) or
    an int (returned as that HTTP status with empty body).
    """

    edge_idx = {"i": 0}
    agent_idx = {"i": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/predict":
            spec = edge_responses[edge_idx["i"]]
            edge_idx["i"] += 1
        elif path == "/recommend":
            spec = agent_responses[agent_idx["i"]]
            agent_idx["i"] += 1
        else:
            return httpx.Response(404, text=f"unexpected path {path}")
        if isinstance(spec, int):
            return httpx.Response(spec, text="error")
        return httpx.Response(200, json=spec)

    return httpx.MockTransport(_handler)


def _make_clients(transport: httpx.MockTransport) -> tuple[EdgeClient, AgentClient, httpx.AsyncClient, httpx.AsyncClient]:
    edge_http = httpx.AsyncClient(transport=transport, base_url="http://edge.test")
    agent_http = httpx.AsyncClient(transport=transport, base_url="http://agent.test")
    return EdgeClient(edge_http), AgentClient(agent_http), edge_http, agent_http


# ---------------------------------------------------------------------------
# JsonlEventWriter
# ---------------------------------------------------------------------------


def _stub_event(node_id: str = "n1") -> OrchestratorEvent:
    return OrchestratorEvent(
        event_id="abc",
        timestamp=datetime(2026, 5, 9, tzinfo=UTC),
        node_id=node_id,
        system_id="sys-1",
        system_type=SystemType.PV,
        step_number=0,
        ground_truth_label="PV_Normal",
        alert=Alert.model_validate(
            _alert_payload(severity="monitor", fault_class="PV_Normal", system_id="sys-1")
        ),
        recommendation=None,
        error=None,
        edge_elapsed_ms=1.5,
        agent_elapsed_ms=None,
    )


def test_writer_appends_jsonl(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    writer = JsonlEventWriter(path)
    writer.append(_stub_event("n1"))
    writer.append(_stub_event("n2"))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["node_id"] == "n1"
    assert json.loads(lines[1])["node_id"] == "n2"


def test_writer_read_all_round_trips(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    writer = JsonlEventWriter(path)
    e = _stub_event()
    writer.append(e)
    out = writer.read_all()
    assert len(out) == 1
    assert out[0].event_id == e.event_id


def test_writer_truncate_clears_file(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    writer = JsonlEventWriter(path)
    writer.append(_stub_event())
    writer.truncate()
    assert path.read_text(encoding="utf-8") == ""
    assert writer.read_all() == []


def test_summarize_counts_severity_and_faults(tmp_path) -> None:
    e_warning = OrchestratorEvent(
        event_id="1",
        timestamp=datetime(2026, 5, 9, tzinfo=UTC),
        node_id="n1",
        system_id="sys",
        system_type=SystemType.PV,
        step_number=0,
        ground_truth_label="Partial_shading",
        alert=Alert.model_validate(
            _alert_payload(
                severity="warning", fault_class="Partial_shading", system_id="sys"
            )
        ),
    )
    e_err = OrchestratorEvent(
        event_id="2",
        timestamp=datetime(2026, 5, 9, tzinfo=UTC),
        node_id="n1",
        system_id="sys",
        system_type=SystemType.PV,
        step_number=1,
        ground_truth_label="PV_Normal",
        error="edge_predict_failed: connection refused",
    )
    s = summarize([e_warning, e_err])
    assert s["n_total"] == 2
    assert s["n_alerts"] == 1
    assert s["n_errors"] == 1
    assert s["by_severity"] == {"warning": 1}


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_edge_client_decodes_alert_on_200() -> None:
    transport = _scripted_transport(
        edge_responses=[
            _alert_payload(
                severity="critical", fault_class="Inverter_fault", system_id="sys-1"
            )
        ],
        agent_responses=[],
    )
    edge, _, edge_http, agent_http = _make_clients(transport)
    try:
        from api.schemas import SensorWindow

        window = SensorWindow.model_validate(
            {
                "timestamp_start": datetime(2026, 5, 9, tzinfo=UTC).isoformat(),
                "system_id": "sys-1",
                "system_type": "PV",
                "sample_rate_hz": 1.0,
                "window_size": 2,
                "feature_names": ["P_dc", "P_ac"],
                "values": [[1.0, 2.0], [3.0, 4.0]],
            }
        )
        out = await edge.predict(window)
    finally:
        await edge_http.aclose()
        await agent_http.aclose()
    assert isinstance(out, Alert)
    assert out.fault_class == "Inverter_fault"


@pytest.mark.asyncio
async def test_edge_client_returns_client_error_on_503() -> None:
    transport = _scripted_transport(
        edge_responses=[503],
        agent_responses=[],
    )
    edge, _, edge_http, agent_http = _make_clients(transport)
    try:
        from api.schemas import SensorWindow

        window = SensorWindow.model_validate(
            {
                "timestamp_start": datetime(2026, 5, 9, tzinfo=UTC).isoformat(),
                "system_id": "sys-1",
                "system_type": "PV",
                "sample_rate_hz": 1.0,
                "window_size": 2,
                "feature_names": ["a", "b"],
                "values": [[1.0, 2.0], [3.0, 4.0]],
            }
        )
        out = await edge.predict(window)
    finally:
        await edge_http.aclose()
        await agent_http.aclose()
    assert isinstance(out, ClientError)
    assert out.status_code == 503


@pytest.mark.asyncio
async def test_agent_client_returns_recommendation() -> None:
    transport = _scripted_transport(
        edge_responses=[],
        agent_responses=[_recommendation_payload()],
    )
    _, agent, edge_http, agent_http = _make_clients(transport)
    try:
        alert = Alert.model_validate(
            _alert_payload(
                severity="critical", fault_class="Inverter_fault", system_id="sys"
            )
        )
        rec, elapsed = await agent.recommend_timed(alert)
    finally:
        await edge_http.aclose()
        await agent_http.aclose()
    assert isinstance(rec, Recommendation)
    assert rec.urgency is Urgency.IMMEDIATE
    assert elapsed >= 0.0


# ---------------------------------------------------------------------------
# NodeRunner.step
# ---------------------------------------------------------------------------


def _node_config(*, system_type: SystemType, fault_probability: float, seed: int) -> NodeConfig:
    return NodeConfig(
        node_id=f"n-{system_type.value.lower()}-{seed}",
        system_id=f"S-{system_type.value}-{seed:03d}",
        system_type=system_type,
        seed=seed,
        fault_probability=fault_probability,
        period_seconds=0.05,  # 单测用很短周期
    )


@pytest.mark.asyncio
async def test_node_step_emits_event_with_alert_for_monitor(tmp_path) -> None:
    """Edge returns severity=monitor → no agent call, alert recorded."""

    transport = _scripted_transport(
        edge_responses=[
            _alert_payload(severity="monitor", fault_class="PV_Normal", system_id="x")
        ],
        agent_responses=[],  # MUST not be called
    )
    edge, agent, edge_http, agent_http = _make_clients(transport)
    writer = JsonlEventWriter(tmp_path / "events.jsonl")
    runner = NodeRunner(
        _node_config(system_type=SystemType.PV, fault_probability=0.0, seed=1),
        edge=edge,
        agent=agent,
        writer=writer,
    )
    try:
        event = await runner.step()
    finally:
        await edge_http.aclose()
        await agent_http.aclose()
    assert event.alert is not None
    assert event.recommendation is None
    assert event.error is None
    assert event.alert.severity is Severity.MONITOR
    assert runner.state.n_alerts == 1
    assert runner.state.n_recommendations == 0


@pytest.mark.asyncio
async def test_node_step_calls_agent_for_warning(tmp_path) -> None:
    transport = _scripted_transport(
        edge_responses=[
            _alert_payload(
                severity="warning", fault_class="Partial_shading", system_id="x"
            )
        ],
        agent_responses=[_recommendation_payload()],
    )
    edge, agent, edge_http, agent_http = _make_clients(transport)
    writer = JsonlEventWriter(tmp_path / "events.jsonl")
    runner = NodeRunner(
        _node_config(system_type=SystemType.PV, fault_probability=1.0, seed=2),
        edge=edge,
        agent=agent,
        writer=writer,
    )
    try:
        event = await runner.step()
    finally:
        await edge_http.aclose()
        await agent_http.aclose()
    assert event.alert is not None
    assert event.recommendation is not None
    assert event.recommendation.urgency is Urgency.IMMEDIATE
    assert runner.state.n_recommendations == 1


@pytest.mark.asyncio
async def test_node_step_edge_only_skips_agent_on_warning(tmp_path) -> None:
    transport = _scripted_transport(
        edge_responses=[
            _alert_payload(
                severity="warning", fault_class="Partial_shading", system_id="x"
            )
        ],
        agent_responses=[],
    )
    edge, agent, edge_http, agent_http = _make_clients(transport)
    writer = JsonlEventWriter(tmp_path / "events.jsonl")
    cfg = _node_config(system_type=SystemType.PV, fault_probability=1.0, seed=21).model_copy(
        update={"integration_mode": "edge_only"}
    )
    runner = NodeRunner(cfg, edge=edge, agent=agent, writer=writer)
    try:
        event = await runner.step()
    finally:
        await edge_http.aclose()
        await agent_http.aclose()
    assert event.alert is not None
    assert event.recommendation is None
    assert event.error is None
    assert runner.state.n_recommendations == 0


@pytest.mark.asyncio
async def test_node_step_cloud_only_uses_synthetic_alert_and_calls_agent(tmp_path) -> None:
    transport = _scripted_transport(
        edge_responses=[],
        agent_responses=[_recommendation_payload()],
    )
    edge, agent, edge_http, agent_http = _make_clients(transport)
    writer = JsonlEventWriter(tmp_path / "events.jsonl")
    cfg = _node_config(system_type=SystemType.PV, fault_probability=1.0, seed=22).model_copy(
        update={"integration_mode": "cloud_only"}
    )
    runner = NodeRunner(cfg, edge=edge, agent=agent, writer=writer)
    try:
        event = await runner.step()
    finally:
        await edge_http.aclose()
        await agent_http.aclose()
    assert event.edge_elapsed_ms == 0.0
    assert event.alert is not None
    assert event.recommendation is not None
    assert runner.state.n_recommendations == 1


@pytest.mark.asyncio
async def test_node_step_records_edge_error_without_recommendation(tmp_path) -> None:
    transport = _scripted_transport(
        edge_responses=[503],
        agent_responses=[],
    )
    edge, agent, edge_http, agent_http = _make_clients(transport)
    writer = JsonlEventWriter(tmp_path / "events.jsonl")
    runner = NodeRunner(
        _node_config(system_type=SystemType.PV, fault_probability=0.5, seed=3),
        edge=edge,
        agent=agent,
        writer=writer,
    )
    try:
        event = await runner.step()
    finally:
        await edge_http.aclose()
        await agent_http.aclose()
    assert event.alert is None
    assert event.recommendation is None
    assert event.error is not None
    assert event.error.startswith("edge_predict_failed")
    assert runner.state.n_errors == 1


@pytest.mark.asyncio
async def test_node_step_handles_agent_error(tmp_path) -> None:
    transport = _scripted_transport(
        edge_responses=[
            _alert_payload(
                severity="critical", fault_class="Inverter_fault", system_id="x"
            )
        ],
        agent_responses=[503],
    )
    edge, agent, edge_http, agent_http = _make_clients(transport)
    writer = JsonlEventWriter(tmp_path / "events.jsonl")
    runner = NodeRunner(
        _node_config(system_type=SystemType.PV, fault_probability=1.0, seed=4),
        edge=edge,
        agent=agent,
        writer=writer,
    )
    try:
        event = await runner.step()
    finally:
        await edge_http.aclose()
        await agent_http.aclose()
    assert event.alert is not None  # 我们仍然保留 alert
    assert event.recommendation is None
    assert event.error is not None and event.error.startswith("agent_recommend_failed")
    assert runner.state.n_errors == 1


@pytest.mark.asyncio
async def test_node_step_deterministic_label_given_seed(tmp_path) -> None:
    """Same seed + same step count → same ground_truth_label sequence."""

    def make_runner(seed: int) -> tuple[NodeRunner, Any, Any]:
        transport = _scripted_transport(
            edge_responses=[
                _alert_payload(severity="monitor", fault_class="PV_Normal", system_id="x")
            ]
            * 5,
            agent_responses=[],
        )
        edge, agent, edge_http, agent_http = _make_clients(transport)
        writer = JsonlEventWriter(tmp_path / f"events-{seed}.jsonl")
        runner = NodeRunner(
            _node_config(system_type=SystemType.PV, fault_probability=0.5, seed=seed),
            edge=edge,
            agent=agent,
            writer=writer,
        )
        return runner, edge_http, agent_http

    runner_a, ea, aa = make_runner(seed=42)
    runner_b, eb, ab = make_runner(seed=42)
    try:
        labels_a = [(await runner_a.step()).ground_truth_label for _ in range(5)]
        labels_b = [(await runner_b.step()).ground_truth_label for _ in range(5)]
    finally:
        for c in (ea, aa, eb, ab):
            await c.aclose()
    assert labels_a == labels_b
    # 同时验证至少出现一次故障标签（保证 fault_probability 真正生效）
    assert any(lbl != "PV_Normal" for lbl in labels_a)


# ---------------------------------------------------------------------------
# Orchestrator (run + summary)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_runs_for_duration_and_summarizes(tmp_path) -> None:
    transport = _scripted_transport(
        edge_responses=[
            _alert_payload(severity="monitor", fault_class="PV_Normal", system_id="x")
        ]
        * 100,
        agent_responses=[],
    )
    edge, agent, edge_http, agent_http = _make_clients(transport)
    writer = JsonlEventWriter(tmp_path / "events.jsonl")
    cfg = OrchestratorConfig(
        nodes=(
            _node_config(system_type=SystemType.PV, fault_probability=0.0, seed=1),
            _node_config(system_type=SystemType.PV, fault_probability=0.0, seed=2),
        ),
        duration_seconds=0.25,  # ~5 steps per node @ 0.05s
    )
    orch = Orchestrator(cfg, edge=edge, agent=agent, writer=writer)
    try:
        await orch.run()
        summary = orch.summary()
    finally:
        await edge_http.aclose()
        await agent_http.aclose()
    assert summary["n_nodes"] == 2
    assert summary["global"]["n_total"] >= 2  # 至少每节点跑过一步
    for per in summary["per_node"]:
        assert per["n_steps"] >= 1


@pytest.mark.asyncio
async def test_orchestrator_rejects_zero_nodes(tmp_path) -> None:
    transport = _scripted_transport(edge_responses=[], agent_responses=[])
    edge, agent, edge_http, agent_http = _make_clients(transport)
    try:
        with pytest.raises(ValueError, match="at least one node"):
            Orchestrator(
                OrchestratorConfig(nodes=(), duration_seconds=0.01),
                edge=edge,
                agent=agent,
                writer=JsonlEventWriter(tmp_path / "x.jsonl"),
            )
    finally:
        await edge_http.aclose()
        await agent_http.aclose()


# ---------------------------------------------------------------------------
# OrchestratorEvent schema
# ---------------------------------------------------------------------------


def test_orchestrator_event_requires_alert_or_error() -> None:
    with pytest.raises(ValueError, match="requires either alert or error"):
        OrchestratorEvent(
            event_id="x",
            timestamp=datetime(2026, 5, 9, tzinfo=UTC),
            node_id="n",
            system_id="sys",
            system_type=SystemType.PV,
            step_number=0,
            ground_truth_label="PV_Normal",
        )


def test_orchestrator_event_recommendation_requires_alert() -> None:
    rec = Recommendation.model_validate(_recommendation_payload())
    with pytest.raises(ValueError, match="recommendation cannot be set"):
        OrchestratorEvent(
            event_id="x",
            timestamp=datetime(2026, 5, 9, tzinfo=UTC),
            node_id="n",
            system_id="sys",
            system_type=SystemType.PV,
            step_number=0,
            ground_truth_label="PV_Normal",
            recommendation=rec,
            error="something went wrong",
        )


def test_orchestrator_event_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="not in known fault classes"):
        OrchestratorEvent(
            event_id="x",
            timestamp=datetime(2026, 5, 9, tzinfo=UTC),
            node_id="n",
            system_id="sys",
            system_type=SystemType.PV,
            step_number=0,
            ground_truth_label="Made_up_fault",
            error="bad label",
        )


# ---------------------------------------------------------------------------
# Built-in node catalogues (Component 6 multi-node fan-out)
# ---------------------------------------------------------------------------


def test_catalog_pv6_bess4_has_at_least_10_nodes() -> None:
    """C6 requires ≥10 simulated nodes — verify the catalogue stays compliant."""

    from orchestrator.__main__ import _catalog

    nodes = _catalog("pv6_bess4")
    assert len(nodes) >= 10
    pv = [n for n in nodes if n.system_type is SystemType.PV]
    bess = [n for n in nodes if n.system_type is SystemType.BESS]
    assert len(pv) == 6
    assert len(bess) == 4
    ids = [n.node_id for n in nodes]
    assert len(set(ids)) == len(ids), "node_id must be unique"
    sys_ids = [n.system_id for n in nodes]
    assert len(set(sys_ids)) == len(sys_ids), "system_id must be unique"
    modes = {n.integration_mode for n in nodes}
    assert modes == {"full", "edge_only", "cloud_only"}, (
        "All three integration modes must be represented for the C6 ablation."
    )


def test_catalog_rejects_unknown_name() -> None:
    from orchestrator.__main__ import _catalog

    with pytest.raises(ValueError, match="Unknown nodes catalogue"):
        _catalog("does_not_exist")
