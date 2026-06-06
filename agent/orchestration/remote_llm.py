"""HTTP-backed LLM client for **local Ollama** (``/api/chat``).

Used when ``configs.settings.Settings.llm_backend`` is ``ollama``.
Planning and synthesis request **JSON-only** replies; malformed JSON falls
back to :class:`agent.orchestration.llm_client.MockLlmClient` behaviour for
tool plans only — synthesis raises on hard failures so callers can surface
5xx.
"""
from __future__ import annotations

import json
import re
from typing import Any

import httpx

from agent.orchestration.llm_client import LlmClient, MockLlmClient, ToolCall
from api.schemas import AgentConfidence, Alert, Severity
from configs.settings import Settings
from utils.logging_config import get_logger

log = get_logger(__name__)

_PLANNER_SYSTEM = """You are AgentPV, a PV/BESS operations planner.
Given an alert JSON, output ONLY valid JSON (no markdown fences) with this shape:
{"tool_calls":[{"tool_name":string,"args":object},...]}
Allowed tool_name values exactly:
retrieve_knowledge, system_history, estimate_rul, escalate_alert
Schema for args:
- retrieve_knowledge: {"query": string, "top_k": integer 1-5}
- system_history: {"system_id": string, "system_type": "PV"|"BESS", "hours": integer}
- estimate_rul: {"system_id", "system_type", "fault_class", "severity", "confidence"}
- escalate_alert: {"system_id", "system_type", "fault_class", "severity", "urgency", "summary"}
Use the alert's system_id, system_type, fault_class, severity, confidence where applicable.
Order tools logically: knowledge first, then history, then rul/escalate if severity warrants."""

_SYNTH_SYSTEM = """You are AgentPV. Output ONLY valid JSON (no markdown fences):
{"recommended_action": string, "agent_confidence": "low"|"medium"|"high"}
Rules:
- recommended_action must be concise operational guidance (2-4 sentences), plain text inside JSON string.
- Use "high" only if retrieve_knowledge returned at least one source_titles entry AND severity is critical.
- Otherwise use medium or low. Never claim high confidence without cited knowledge."""


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def _parse_json_object(text: str) -> dict[str, Any]:
    t = _strip_code_fence(text)
    if not t.startswith("{"):
        m = re.search(r"\{[\s\S]*\}", t)
        if m:
            t = m.group(0)
    return json.loads(t)


def _mock_plan(alert: Alert) -> list[ToolCall]:
    return MockLlmClient().plan_tools(alert)


class OllamaChatLlmClient(LlmClient):
    """Local Ollama ``/api/chat`` with JSON formatted replies when supported."""

    backend_name = "ollama"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base = settings.ollama_base_url.rstrip("/")
        self._model = settings.ollama_model
        self._timeout = float(settings.llm_timeout_s)

    def _chat_json(self, system: str, user: str) -> dict[str, Any]:
        url = f"{self._base}/api/chat"
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        }
        with httpx.Client(timeout=self._timeout) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
        content = data.get("message", {}).get("content") or ""
        return _parse_json_object(content)

    def plan_tools(self, alert: Alert) -> list[ToolCall]:
        user = json.dumps(alert.model_dump(mode="json"), default=str)
        try:
            obj = self._chat_json(_PLANNER_SYSTEM, user)
            raw_calls = obj.get("tool_calls") or []
            plan: list[ToolCall] = []
            for item in raw_calls:
                if not isinstance(item, dict):
                    continue
                name = item.get("tool_name")
                args = item.get("args")
                if isinstance(name, str) and isinstance(args, dict):
                    plan.append(ToolCall(tool_name=name, args=args))
            if plan:
                return plan
        except Exception as exc:  # noqa: BLE001
            log.warning("ollama_plan_fallback_mock", extra={"error": str(exc)})
        return _mock_plan(alert)

    def synthesize_recommendation(
        self,
        *,
        alert: Alert,
        tool_results: dict[str, Any],
    ) -> tuple[str, AgentConfidence]:
        user = json.dumps(
            {"alert": alert.model_dump(mode="json"), "tool_results": tool_results},
            default=str,
        )
        obj = self._chat_json(_SYNTH_SYSTEM, user)
        action = str(obj.get("recommended_action") or "").strip()
        conf_raw = str(obj.get("agent_confidence") or "medium").strip().lower()
        if not action:
            raise RuntimeError("LLM returned empty recommended_action")
        sources = (tool_results.get("retrieve_knowledge") or {}).get("source_titles") or []
        confidence = _coerce_confidence(conf_raw, alert, bool(sources))
        return action, confidence


def _coerce_confidence(
    raw: str,
    alert: Alert,
    has_knowledge: bool,
) -> AgentConfidence:
    try:
        c = AgentConfidence(raw)
    except ValueError:
        c = AgentConfidence.MEDIUM
    if c is AgentConfidence.HIGH and (not has_knowledge or alert.severity is not Severity.CRITICAL):
        return AgentConfidence.MEDIUM
    return c
