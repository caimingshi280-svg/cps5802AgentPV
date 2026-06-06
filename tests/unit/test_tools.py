"""Unit tests for the four MVP tools (rule §11)."""
from __future__ import annotations

import pytest

from api.schemas import Severity, SystemType, Urgency
from rag.chunking import Chunk
from rag.retrieval import Retriever
from tools.escalate_alert import EscalateAlertTool
from tools.estimate_rul import EstimateRulTool
from tools.retrieve_knowledge import RetrieveKnowledgeTool
from tools.system_history import SystemHistoryTool


def _is_tool_error(d: dict) -> bool:
    return {"error_code", "tool_name", "trace_id"}.issubset(d.keys())


# ---------------------------------------------------------------------------
# retrieve_knowledge
# ---------------------------------------------------------------------------


@pytest.fixture
def retrieve_tool() -> RetrieveKnowledgeTool:
    chunks = [
        Chunk(text="Inverter fault diagnosis requires P_ac collapse.", source="i.md", title="Inverter", section=None),
        Chunk(text="Partial shading multi-step IV curve.", source="s.md", title="Shading", section=None),
        Chunk(text="Battery thermal runaway preceded by cell temperature rise.", source="t.md", title="Thermal", section=None),
    ]
    return RetrieveKnowledgeTool(retriever=Retriever(chunks))


@pytest.mark.asyncio
async def test_retrieve_knowledge_returns_docs(retrieve_tool) -> None:
    out = await retrieve_tool({"query": "inverter fault P_ac", "top_k": 2})
    assert not _is_tool_error(out)
    assert out["query"] == "inverter fault P_ac"
    assert len(out["docs"]) == 2
    assert out["docs"][0]["title"] == "Inverter"
    assert out["source_titles"][0] == "Inverter"


@pytest.mark.asyncio
async def test_retrieve_knowledge_validation_error(retrieve_tool) -> None:
    out = await retrieve_tool({"query": "", "top_k": 3})  # min_length=1
    assert _is_tool_error(out)
    assert out["error_code"] == "VALIDATION"


@pytest.mark.asyncio
async def test_retrieve_knowledge_top_k_clamped(retrieve_tool) -> None:
    """top_k larger than corpus must still return what's available."""
    out = await retrieve_tool({"query": "anything", "top_k": 5})
    assert not _is_tool_error(out)
    assert len(out["docs"]) <= 5


# ---------------------------------------------------------------------------
# system_history
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_system_history_returns_mock_entries() -> None:
    out = await SystemHistoryTool()(
        {"system_id": "PV_X1", "system_type": "PV", "hours": 24}
    )
    assert not _is_tool_error(out)
    assert out["backend"] == "mock"
    assert isinstance(out["entries"], list)
    for e in out["entries"]:
        assert e["note"].startswith("[MOCK]")


@pytest.mark.asyncio
async def test_system_history_deterministic_on_same_id() -> None:
    tool = SystemHistoryTool()
    out_a = await tool({"system_id": "BESS_42", "system_type": "BESS", "hours": 24})
    out_b = await tool({"system_id": "BESS_42", "system_type": "BESS", "hours": 24})
    # 时间戳会变（now）；故障序列必须相同。
    assert [e["fault_class"] for e in out_a["entries"]] == [
        e["fault_class"] for e in out_b["entries"]
    ]


@pytest.mark.asyncio
async def test_system_history_different_ids_may_differ() -> None:
    tool = SystemHistoryTool()
    seqs = []
    for sid in ("X1", "X2", "X3", "X4", "X5", "X6"):
        out = await tool({"system_id": sid, "system_type": "PV", "hours": 24})
        seqs.append(tuple(e["fault_class"] for e in out["entries"]))
    # 至少有两个不同的序列（避免 hash 全部撞同一桶）。
    assert len(set(seqs)) >= 2


# ---------------------------------------------------------------------------
# estimate_rul
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_estimate_rul_critical_returns_zero_days() -> None:
    out = await EstimateRulTool()(
        {
            "system_id": "PV_007",
            "system_type": "PV",
            "fault_class": "Inverter_fault",
            "severity": "critical",
            "confidence": 0.99,
        }
    )
    assert not _is_tool_error(out)
    assert out["rul_days_estimate"] == 0
    assert out["requires_immediate_action"] is True


@pytest.mark.asyncio
async def test_estimate_rul_normal_returns_long_horizon() -> None:
    out = await EstimateRulTool()(
        {
            "system_id": "PV_007",
            "system_type": "PV",
            "fault_class": "PV_Normal",
            "severity": "monitor",
            "confidence": 0.95,
        }
    )
    assert not _is_tool_error(out)
    assert out["rul_days_estimate"] > 365
    assert out["requires_immediate_action"] is False


@pytest.mark.asyncio
async def test_estimate_rul_unknown_class_returns_internal_error() -> None:
    out = await EstimateRulTool()(
        {
            "system_id": "PV_007",
            "system_type": "PV",
            "fault_class": "Made_up_fault",
            "severity": "warning",
            "confidence": 0.5,
        }
    )
    assert _is_tool_error(out)
    assert out["error_code"] == "INTERNAL"
    assert "Unknown fault_class" in out["message"]


@pytest.mark.asyncio
async def test_estimate_rul_higher_confidence_narrows_band() -> None:
    tool = EstimateRulTool()
    low = await tool(
        {
            "system_id": "B1",
            "system_type": "BESS",
            "fault_class": "Capacity_fade",
            "severity": "monitor",
            "confidence": 0.20,
        }
    )
    high = await tool(
        {
            "system_id": "B1",
            "system_type": "BESS",
            "fault_class": "Capacity_fade",
            "severity": "monitor",
            "confidence": 0.95,
        }
    )
    low_band = low["rul_days_upper"] - low["rul_days_lower"]
    high_band = high["rul_days_upper"] - high["rul_days_lower"]
    assert high_band < low_band


# ---------------------------------------------------------------------------
# escalate_alert
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_escalate_alert_returns_id_and_mock_marker() -> None:
    out = await EscalateAlertTool()(
        {
            "system_id": "PV_007",
            "system_type": SystemType.PV.value,
            "fault_class": "Inverter_fault",
            "severity": Severity.CRITICAL.value,
            "urgency": Urgency.IMMEDIATE.value,
            "summary": "test escalation",
        }
    )
    assert not _is_tool_error(out)
    assert out["escalated"] is True
    assert out["backend"] == "audit_log"
    assert out["note"].startswith("[MOCK]")
    assert len(out["escalation_id"]) == 32  # uuid4 hex


@pytest.mark.asyncio
async def test_escalate_alert_missing_summary_validation_error() -> None:
    out = await EscalateAlertTool()(
        {
            "system_id": "X",
            "system_type": "PV",
            "fault_class": "Inverter_fault",
            "severity": "critical",
            "urgency": "immediate",
            # summary missing
        }
    )
    assert _is_tool_error(out)
    assert out["error_code"] == "VALIDATION"
