"""LLM client abstraction + Mock backend for AgentPV (Component 4 MVP).

Per project rule §12, every placeholder backend must be **clearly marked**.
:class:`MockLlmClient` outputs are guaranteed to begin with ``[MOCK]`` so a
report or dashboard can always tell synthetic from real LLM output.

Polish-phase backend (**Ollama** local) implements the same
:class:`LlmClient` interface; the orchestrator reads the active backend
from :mod:`configs.settings`.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

from api.schemas import (
    AgentConfidence,
    Alert,
    Severity,
    SystemType,
    Urgency,
)
from configs.settings import Settings, get_settings

# ---------------------------------------------------------------------------
# Plan format used internally by ReAct.
# A plan is an ordered list of (tool_name, args_dict).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """One step in a tool-execution plan."""

    tool_name: str
    args: dict[str, Any]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class LlmClient(abc.ABC):
    """Backend-agnostic LLM contract.

    The Mock backend implements the methods deterministically; the polish-
    phase backends will issue actual HTTP calls.
    """

    backend_name: str

    @abc.abstractmethod
    def plan_tools(self, alert: Alert) -> list[ToolCall]:
        """Decide which tools to call (and in what order) for this alert."""

    @abc.abstractmethod
    def synthesize_recommendation(
        self,
        *,
        alert: Alert,
        tool_results: dict[str, Any],
    ) -> tuple[str, AgentConfidence]:
        """Compose the final recommended_action string + confidence."""


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------


_URGENCY_FOR_SEVERITY: dict[Severity, Urgency] = {
    Severity.CRITICAL: Urgency.IMMEDIATE,
    Severity.WARNING: Urgency.SCHEDULED,
    Severity.MONITOR: Urgency.MONITOR,
}


def urgency_for_severity(severity: Severity) -> Urgency:
    """Map :class:`Severity` to :class:`Urgency` (canonical mapping)."""

    return _URGENCY_FOR_SEVERITY[severity]


def _eta_for_severity(severity: Severity) -> str:
    if severity is Severity.CRITICAL:
        return "the next 4 hours"
    if severity is Severity.WARNING:
        return "1 week"
    return "the next monthly review"


class MockLlmClient(LlmClient):
    """Deterministic, prompt-free LLM mock.

    Decision rules:

    * Always retrieve documentation for the fault class.
    * If severity is ``critical`` or fault is in the always-escalate set
      (Inverter_fault / String_disconnection / Thermal_anomaly), call
      :class:`tools.escalate_alert.EscalateAlertTool` and run RUL estimation.
    * Always pull a 24-hour history snapshot.

    All recommendations begin with ``[MOCK]`` (rule §12).
    """

    backend_name = "mock"

    _CRITICAL_FAULTS = {"Inverter_fault", "String_disconnection", "Thermal_anomaly"}

    def plan_tools(self, alert: Alert) -> list[ToolCall]:
        plan: list[ToolCall] = [
            ToolCall(
                tool_name="retrieve_knowledge",
                args={
                    "query": f"{alert.fault_class} mitigation procedure {alert.system_type.value}",
                    "top_k": 3,
                },
            ),
            ToolCall(
                tool_name="system_history",
                args={
                    "system_id": alert.system_id,
                    "system_type": alert.system_type.value,
                    "hours": 24,
                },
            ),
        ]
        is_critical = (
            alert.severity is Severity.CRITICAL
            or alert.fault_class in self._CRITICAL_FAULTS
        )
        if is_critical:
            plan.append(
                ToolCall(
                    tool_name="estimate_rul",
                    args={
                        "system_id": alert.system_id,
                        "system_type": alert.system_type.value,
                        "fault_class": alert.fault_class,
                        "severity": alert.severity.value,
                        "confidence": alert.confidence,
                    },
                )
            )
            plan.append(
                ToolCall(
                    tool_name="escalate_alert",
                    args={
                        "system_id": alert.system_id,
                        "system_type": alert.system_type.value,
                        "fault_class": alert.fault_class,
                        "severity": alert.severity.value,
                        "urgency": Urgency.IMMEDIATE.value,
                        "summary": (
                            f"{alert.fault_class} ({alert.severity.value}) on "
                            f"{alert.system_id}, confidence={alert.confidence:.2f}"
                        ),
                    },
                )
            )
        return plan

    def synthesize_recommendation(
        self,
        *,
        alert: Alert,
        tool_results: dict[str, Any],
    ) -> tuple[str, AgentConfidence]:
        retrieved = tool_results.get("retrieve_knowledge") or {}
        source_titles: list[str] = retrieved.get("source_titles") or []
        history = tool_results.get("system_history") or {}
        n_recent_warnings = int(history.get("n_warning", 0)) + int(
            history.get("n_critical", 0)
        )

        repeat_clause = (
            f" {n_recent_warnings} prior alert(s) in last 24 h escalate this case."
            if n_recent_warnings > 0
            else ""
        )

        sources_clause = (
            f"Cite: {'; '.join(source_titles[:2])}." if source_titles else ""
        )

        eta = _eta_for_severity(alert.severity)
        action = (
            f"[MOCK] Inspect {alert.system_id} ({alert.system_type.value}) for "
            f"'{alert.fault_class}' ({alert.severity.value}, "
            f"conf={alert.confidence:.2f}) within {eta}.{repeat_clause} {sources_clause}"
        ).strip()

        # Confidence policy under the mock backend:
        # * MEDIUM by default — this is a deterministic stub, not a real LLM.
        # * LOW if no knowledge retrieved (we have no evidence).
        # * HIGH only when severity is CRITICAL AND we have at least one cited
        #   knowledge source — meets Recommendation validator requirement.
        if not source_titles:
            confidence = AgentConfidence.LOW
        elif alert.severity is Severity.CRITICAL:
            confidence = AgentConfidence.HIGH
        else:
            confidence = AgentConfidence.MEDIUM
        return action, confidence


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_llm_client(
    backend: str | None = None,
    *,
    settings: Settings | None = None,
) -> LlmClient:
    """Return an :class:`LlmClient` for the given backend name.

    When ``backend`` is omitted, reads ``llm_backend`` from settings.
    Supported: ``mock``, ``ollama``.
    """

    s = settings or get_settings()
    name = (backend or s.llm_backend).strip().lower()
    if name == "mock":
        return MockLlmClient()
    if name == "ollama":
        from agent.orchestration.remote_llm import OllamaChatLlmClient

        return OllamaChatLlmClient(s)
    raise NotImplementedError(
        f"LLM backend {name!r} not implemented; supported: mock, ollama"
    )


# Re-export for convenience.
def system_type_from_value(value: str) -> SystemType:
    return SystemType(value)
