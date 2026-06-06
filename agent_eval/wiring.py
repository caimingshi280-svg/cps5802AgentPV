"""Build :class:`agent.workflows.react.ReActAgent` for offline benchmarks.

Mirrors ``api/agent_service.py::_build_react_agent`` but exposes **ablation**
knobs: drop tools from the registry so the mock LLM plan still references them
and the ReAct loop records *skipped* behaviour — this is the standard way to
measure "what if this tool were unavailable on the edge?".
"""
from __future__ import annotations

from pathlib import Path

from agent.orchestration.llm_client import LlmClient, build_llm_client
from agent.workflows.react import ReActAgent
from configs.settings import get_settings
from tools.escalate_alert import EscalateAlertTool
from tools.estimate_rul import EstimateRulTool
from tools.retrieve_knowledge import build_default_tool as build_retrieve_tool
from tools.system_history import SystemHistoryTool

#: Full toolset keys (must match :class:`tools.*.Tool.name`).
ALL_TOOL_NAMES: frozenset[str] = frozenset(
    {"retrieve_knowledge", "system_history", "estimate_rul", "escalate_alert"}
)


def build_benchmark_agent(
    *,
    knowledge_base_dir: Path,
    llm_backend: str = "mock",
    llm: LlmClient | None = None,
    disabled_tools: frozenset[str] | None = None,
) -> ReActAgent:
    """Construct a :class:`ReActAgent` wired like production ``agent_service``.

    Parameters
    ----------
    knowledge_base_dir
        Directory of markdown playbooks (``rag/knowledge_base/documents``).
    disabled_tools
        Tool names **omitted** from the registry entirely (ablation study).
    """

    disabled = disabled_tools or frozenset()
    unknown = disabled - ALL_TOOL_NAMES
    if unknown:
        raise ValueError(f"unknown disabled_tools entries: {sorted(unknown)}")

    retrieve = build_retrieve_tool(knowledge_base_dir, settings=get_settings())
    tools: dict = {}
    if "retrieve_knowledge" not in disabled:
        tools[retrieve.name] = retrieve
    if "system_history" not in disabled:
        hist = SystemHistoryTool()
        tools[hist.name] = hist
    if "estimate_rul" not in disabled:
        rul = EstimateRulTool()
        tools[rul.name] = rul
    if "escalate_alert" not in disabled:
        esc = EscalateAlertTool()
        tools[esc.name] = esc

    if not tools:
        raise ValueError("disabled_tools removed every tool — pick a smaller ablation set")

    llm_client = (
        llm
        if llm is not None
        else build_llm_client(llm_backend, settings=get_settings())
    )
    return ReActAgent(tools=tools, llm=llm_client)
