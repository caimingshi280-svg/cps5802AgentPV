"""Simplified ReAct workflow for Component 4 MVP.

Loop (rule §27 minimal viable):

1. **Observe** — record the incoming alert.
2. **Reason** — ask the LLM client for a tool plan.
3. **Act** — execute each tool in order, recording the result summary.
4. **Reflect / Report** — synthesize the final recommendation.

Each phase emits a :class:`api.schemas.ReasoningStep` so the auditor /
operator can replay exactly what the agent saw and decided. The trace is
returned alongside the final :class:`api.schemas.Recommendation`.

The MVP is **deterministic** because the bound LLM client is the
:class:`agent.orchestration.llm_client.MockLlmClient`. The same workflow
runs unchanged when a real LLM backend is wired in (polish phase).

中文说明
--------
本文件实现作业要求的 **ReAct 闭环**：观察告警 → LLM 规划工具调用 → 顺序执行
``retrieve_knowledge`` 等工具 → 合成最终 ``Recommendation``。更换为 Ollama
等真实后端时仍走同一状态机，仅 ``LlmClient`` 实现不同。
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from agent.orchestration.llm_client import (
    LlmClient,
    MockLlmClient,
    ToolCall,
    urgency_for_severity,
)
from api.schemas import (
    Alert,
    ReasoningStep,
    Recommendation,
    ToolError,
    Urgency,
)
from tools.base import Tool
from utils.logging_config import get_logger

log = get_logger(__name__)


# Maximum tool calls in one ReAct run — guards against infinite plans.
MAX_TOOL_CALLS = 6


@dataclass
class ReActConfig:
    """Knobs that polish-phase tuning will care about."""

    max_tool_calls: int = MAX_TOOL_CALLS


class ReActAgent:
    """Stateless ReAct executor.

    Construct once with the tool registry + LLM client; each :meth:`run`
    call processes one alert in isolation (no shared state between calls).
    """

    def __init__(
        self,
        tools: dict[str, Tool],
        llm: LlmClient | None = None,
        config: ReActConfig | None = None,
    ) -> None:
        if not tools:
            raise ValueError("ReActAgent requires at least one tool")
        self.tools = dict(tools)
        self.llm: LlmClient = llm or MockLlmClient()
        self.config = config or ReActConfig()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self, alert: Alert, *, strip_reasoning_trace: bool = False) -> Recommendation:
        steps: list[ReasoningStep] = []
        next_step = 0

        # 1) Observe ------------------------------------------------------
        steps.append(
            ReasoningStep(
                step=next_step,
                phase="observe",
                thought=(
                    f"Received {alert.severity.value} alert for "
                    f"{alert.system_id} ({alert.system_type.value}): "
                    f"{alert.fault_class}, confidence={alert.confidence:.2f}"
                ),
            )
        )
        next_step += 1

        # 2) Reason -------------------------------------------------------
        plan = self.llm.plan_tools(alert)
        if len(plan) > self.config.max_tool_calls:
            plan = plan[: self.config.max_tool_calls]
        plan_summary = ", ".join(call.tool_name for call in plan) or "no tools"
        steps.append(
            ReasoningStep(
                step=next_step,
                phase="reason",
                thought=f"Tool plan ({len(plan)}): {plan_summary}",
            )
        )
        next_step += 1

        # 3) Act ----------------------------------------------------------
        tool_results: dict[str, dict] = {}
        for call in plan:
            tool = self.tools.get(call.tool_name)
            if tool is None:
                steps.append(
                    ReasoningStep(
                        step=next_step,
                        phase="act",
                        thought=f"Tool {call.tool_name!r} not registered; skipping.",
                        action=call.tool_name,
                        args=_args_to_jsonable(call.args),
                        result_summary="skipped — unknown tool",
                    )
                )
                next_step += 1
                continue

            result_dict = await tool(call.args)  # type: ignore[arg-type]
            tool_results[call.tool_name] = result_dict
            steps.append(
                ReasoningStep(
                    step=next_step,
                    phase="act",
                    thought=f"Executing {call.tool_name}.",
                    action=call.tool_name,
                    args=_args_to_jsonable(call.args),
                    result_summary=_summarize_tool_result(call.tool_name, result_dict),
                )
            )
            next_step += 1

        # 4) Reflect ------------------------------------------------------
        steps.append(
            ReasoningStep(
                step=next_step,
                phase="reflect",
                thought=(
                    f"Gathered {len(tool_results)} tool result(s); "
                    f"errors: {sum(_is_tool_error(r) for r in tool_results.values())}."
                ),
            )
        )
        next_step += 1

        # 5) Report --------------------------------------------------------
        action, confidence = self.llm.synthesize_recommendation(
            alert=alert, tool_results=tool_results
        )
        urgency: Urgency = urgency_for_severity(alert.severity)
        knowledge_sources: list[str] = (
            (tool_results.get("retrieve_knowledge") or {}).get("source_titles") or []
        )
        steps.append(
            ReasoningStep(
                step=next_step,
                phase="report",
                thought=(
                    f"Final recommendation drafted ({confidence.value} confidence, "
                    f"{urgency.value} urgency, {len(knowledge_sources)} citation(s))."
                ),
            )
        )

        recommendation = Recommendation(
            recommended_action=action,
            urgency=urgency,
            reasoning_trace=steps,
            knowledge_sources=knowledge_sources,
            confidence=confidence,
        )
        if strip_reasoning_trace:
            recommendation = recommendation.model_copy(
                update={
                    "reasoning_trace": [
                        ReasoningStep(
                            step=0,
                            phase="report",
                            thought=(
                                "[ABLATION] reasoning_trace withheld "
                                "(C5 No-Reasoning-Trace interpretability study)."
                            ),
                        )
                    ]
                }
            )
        log.info(
            "react_completed",
            extra={
                "system_id": alert.system_id,
                "fault_class": alert.fault_class,
                "n_steps": len(steps),
                "n_tool_calls": len(tool_results),
                "confidence": confidence.value,
                "trace_stripped": strip_reasoning_trace,
            },
        )
        return recommendation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_tool_error(result: dict) -> bool:
    """A ToolError envelope is a dict with ``error_code`` and ``tool_name``."""

    return isinstance(result, dict) and {"error_code", "tool_name", "trace_id"}.issubset(
        result.keys()
    )


def _summarize_tool_result(tool_name: str, result: dict) -> str:
    """Produce a short human-readable summary for the reasoning trace."""

    if _is_tool_error(result):
        return (
            f"ERROR[{result.get('error_code')}]: "
            f"{(result.get('message') or '')[:120]}"
        )
    if tool_name == "retrieve_knowledge":
        n = len(result.get("docs") or [])
        sources = result.get("source_titles") or []
        return f"retrieved {n} doc(s) from {len(sources)} unique title(s)"
    if tool_name == "system_history":
        return (
            f"backend={result.get('backend')}, "
            f"warnings={result.get('n_warning')}, "
            f"critical={result.get('n_critical')}"
        )
    if tool_name == "estimate_rul":
        return (
            f"rul_days={result.get('rul_days_estimate')} "
            f"({result.get('rul_days_lower')}–{result.get('rul_days_upper')}); "
            f"immediate={result.get('requires_immediate_action')}"
        )
    if tool_name == "escalate_alert":
        return f"escalated={result.get('escalated')}, id={result.get('escalation_id')}"
    return _truncate(json.dumps(result, default=str), 160)


def _args_to_jsonable(args: dict) -> dict:
    """Coerce arbitrary args dict to JSON-serializable for ReasoningStep."""

    out: dict = {}
    for k, v in args.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


__all__ = [
    "ReActAgent",
    "ReActConfig",
    "ToolError",
    "ToolCall",
    "MAX_TOOL_CALLS",
]
