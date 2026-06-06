"""Unit tests for :mod:`agent.workflows.react` + :mod:`agent.orchestration.llm_client`."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agent.orchestration.llm_client import (
    MockLlmClient,
    ToolCall,
    build_llm_client,
    urgency_for_severity,
)
from agent.workflows.react import ReActAgent, ReActConfig
from api.schemas import (
    AgentConfidence,
    Alert,
    Recommendation,
    Severity,
    SystemType,
    Urgency,
)
from rag.chunking import Chunk
from rag.retrieval import Retriever
from tools.escalate_alert import EscalateAlertTool
from tools.estimate_rul import EstimateRulTool
from tools.retrieve_knowledge import RetrieveKnowledgeTool
from tools.system_history import SystemHistoryTool

# ---------------------------------------------------------------------------
# urgency mapping
# ---------------------------------------------------------------------------


def test_urgency_mapping() -> None:
    assert urgency_for_severity(Severity.CRITICAL) is Urgency.IMMEDIATE
    assert urgency_for_severity(Severity.WARNING) is Urgency.SCHEDULED
    assert urgency_for_severity(Severity.MONITOR) is Urgency.MONITOR


# ---------------------------------------------------------------------------
# MockLlmClient
# ---------------------------------------------------------------------------


def _critical_alert() -> Alert:
    return Alert(
        timestamp=datetime(2026, 5, 9, tzinfo=UTC),
        system_id="PV_001",
        system_type=SystemType.PV,
        fault_class="Inverter_fault",
        severity=Severity.CRITICAL,
        confidence=0.95,
        sensor_snapshot={"P_dc": 250.0, "P_ac": 0.0},
    )


def _warning_alert() -> Alert:
    return Alert(
        timestamp=datetime(2026, 5, 9, tzinfo=UTC),
        system_id="PV_001",
        system_type=SystemType.PV,
        fault_class="Partial_shading",
        severity=Severity.WARNING,
        confidence=0.80,
        sensor_snapshot={"P_dc": 200.0},
    )


def test_mock_llm_plan_includes_retrieve_for_any_alert() -> None:
    plan = MockLlmClient().plan_tools(_warning_alert())
    assert plan[0].tool_name == "retrieve_knowledge"


def test_mock_llm_plan_critical_includes_escalate() -> None:
    plan = MockLlmClient().plan_tools(_critical_alert())
    names = [c.tool_name for c in plan]
    assert "escalate_alert" in names
    assert "estimate_rul" in names


def test_mock_llm_plan_warning_excludes_escalate() -> None:
    plan = MockLlmClient().plan_tools(_warning_alert())
    names = [c.tool_name for c in plan]
    assert "escalate_alert" not in names


def test_mock_llm_synthesize_marks_output_as_mock() -> None:
    text, _ = MockLlmClient().synthesize_recommendation(
        alert=_warning_alert(),
        tool_results={"retrieve_knowledge": {"source_titles": ["Doc A"]}},
    )
    assert text.startswith("[MOCK]")


def test_mock_llm_confidence_low_when_no_sources() -> None:
    _, conf = MockLlmClient().synthesize_recommendation(
        alert=_warning_alert(),
        tool_results={"retrieve_knowledge": {"source_titles": []}},
    )
    assert conf is AgentConfidence.LOW


def test_mock_llm_confidence_high_only_when_critical_with_sources() -> None:
    _, conf = MockLlmClient().synthesize_recommendation(
        alert=_critical_alert(),
        tool_results={"retrieve_knowledge": {"source_titles": ["Doc A"]}},
    )
    assert conf is AgentConfidence.HIGH


def test_build_llm_client_unknown_backend_raises() -> None:
    with pytest.raises(NotImplementedError, match="not implemented"):
        build_llm_client("openai")


# ---------------------------------------------------------------------------
# ReActAgent
# ---------------------------------------------------------------------------


@pytest.fixture
def agent_with_real_tools() -> ReActAgent:
    chunks = [
        Chunk(text="Inverter fault diagnosis steps.", source="inv.md", title="Inverter Playbook", section=None),
        Chunk(text="Partial shading recovery.", source="ps.md", title="Shading Playbook", section=None),
    ]
    retrieve = RetrieveKnowledgeTool(retriever=Retriever(chunks))
    return ReActAgent(
        tools={
            retrieve.name: retrieve,
            "system_history": SystemHistoryTool(),
            "estimate_rul": EstimateRulTool(),
            "escalate_alert": EscalateAlertTool(),
        }
    )


@pytest.mark.asyncio
async def test_react_returns_validated_recommendation(agent_with_real_tools) -> None:
    rec = await agent_with_real_tools.run(_critical_alert())
    assert isinstance(rec, Recommendation)
    assert rec.urgency is Urgency.IMMEDIATE
    assert rec.confidence is AgentConfidence.HIGH
    assert len(rec.knowledge_sources) >= 1
    assert rec.recommended_action.startswith("[MOCK]")


@pytest.mark.asyncio
async def test_react_warning_skips_escalate(agent_with_real_tools) -> None:
    rec = await agent_with_real_tools.run(_warning_alert())
    actions = [s.action for s in rec.reasoning_trace if s.action]
    assert "escalate_alert" not in actions


@pytest.mark.asyncio
async def test_react_trace_phases_in_order(agent_with_real_tools) -> None:
    """Phases must follow observe → reason → act* → reflect → report."""

    rec = await agent_with_real_tools.run(_critical_alert())
    phases = [s.phase for s in rec.reasoning_trace]
    assert phases[0] == "observe"
    assert phases[1] == "reason"
    assert phases[-2] == "reflect"
    assert phases[-1] == "report"
    # 中间至少一个 act
    assert any(p == "act" for p in phases[2:-2])
    # step ids 单调递增 0..N-1
    step_ids = [s.step for s in rec.reasoning_trace]
    assert step_ids == list(range(len(step_ids)))


@pytest.mark.asyncio
async def test_react_unknown_tool_in_plan_handled_gracefully() -> None:
    """If LLM plans a tool that's not registered, ReAct logs and continues."""

    class FakeLlm(MockLlmClient):
        def plan_tools(self, alert):
            return [ToolCall("nonexistent_tool", {"foo": "bar"})]

    chunks = [
        Chunk(
            text="Generic playbook content used only to satisfy the TF-IDF vocabulary fitter.",
            source="s.md",
            title="T",
            section=None,
        )
    ]
    agent = ReActAgent(
        tools={"retrieve_knowledge": RetrieveKnowledgeTool(retriever=Retriever(chunks))},
        llm=FakeLlm(),
    )
    rec = await agent.run(_warning_alert())
    # 至少一个 act 步骤标记 skipped
    assert any("skipped" in (s.result_summary or "") for s in rec.reasoning_trace)
    assert rec.confidence is AgentConfidence.LOW  # 没拿到 retrieve 证据


@pytest.mark.asyncio
async def test_react_empty_tools_rejected() -> None:
    with pytest.raises(ValueError, match="at least one tool"):
        ReActAgent(tools={})


@pytest.mark.asyncio
async def test_react_max_tool_calls_truncates_plan() -> None:
    """A plan exceeding max_tool_calls must be truncated, not all executed."""

    class BigPlanLlm(MockLlmClient):
        def plan_tools(self, alert):
            return [
                ToolCall("retrieve_knowledge", {"query": "x", "top_k": 1})
                for _ in range(20)
            ]

    chunks = [
        Chunk(
            text="Generic playbook content used only to satisfy the TF-IDF vocabulary fitter.",
            source="s.md",
            title="T",
            section=None,
        )
    ]
    agent = ReActAgent(
        tools={"retrieve_knowledge": RetrieveKnowledgeTool(retriever=Retriever(chunks))},
        llm=BigPlanLlm(),
        config=ReActConfig(max_tool_calls=3),
    )
    rec = await agent.run(_warning_alert())
    n_acts = sum(1 for s in rec.reasoning_trace if s.phase == "act")
    assert n_acts == 3


@pytest.mark.asyncio
async def test_react_strip_reasoning_trace_collapses_trace() -> None:
    chunks = [
        Chunk(
            text="Inspect strings for shading patterns.",
            source="a.md",
            title="Playbook",
            section=None,
        )
    ]
    agent = ReActAgent(
        tools={"retrieve_knowledge": RetrieveKnowledgeTool(retriever=Retriever(chunks))},
        llm=MockLlmClient(),
    )
    rec = await agent.run(_warning_alert(), strip_reasoning_trace=True)
    assert len(rec.reasoning_trace) == 1
    assert rec.reasoning_trace[0].phase == "report"
    assert "ABLATION" in rec.reasoning_trace[0].thought
