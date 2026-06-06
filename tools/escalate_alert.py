"""Tool: escalate_alert — record an escalation in the structured audit log.

The MVP backend writes a structured log line via :func:`utils.logging_config.get_logger`.
The polish-phase replacement can post to PagerDuty / Slack / a ticketing
system without altering the contract on this module.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from api.schemas import Severity, SystemType, Urgency
from tools.base import Tool
from utils.logging_config import get_logger

log = get_logger("agentpv.escalation")


class EscalateAlertInput(BaseModel):
    system_id: str = Field(min_length=1, max_length=64)
    system_type: SystemType
    fault_class: str = Field(min_length=1)
    severity: Severity
    urgency: Urgency
    summary: str = Field(min_length=1, max_length=500)


class EscalateAlertOutput(BaseModel):
    escalated: bool
    escalation_id: str
    backend: str = Field(description="One of: 'audit_log' | 'pagerduty' | 'slack'")
    timestamp: datetime
    channel: str
    note: str  # 始终以 "[MOCK]" 前缀开头


class EscalateAlertTool(Tool[EscalateAlertInput, EscalateAlertOutput]):
    """Append an escalation record to the structured log.

    MVP backend: ``audit_log`` — emits a single structured log entry that
    operations can grep for later. No external service is touched.
    """

    name = "escalate_alert"
    description = (
        "Record a fault escalation. MVP backend writes to the audit log only; "
        "polish-phase backends can fan out to PagerDuty/Slack."
    )
    input_model = EscalateAlertInput
    output_model = EscalateAlertOutput
    timeout_s = 3.0

    async def _run(self, inp: EscalateAlertInput) -> EscalateAlertOutput:
        escalation_id = uuid.uuid4().hex
        ts = datetime.now(UTC)
        log.info(
            "alert_escalated",
            extra={
                "escalation_id": escalation_id,
                "system_id": inp.system_id,
                "system_type": inp.system_type.value,
                "fault_class": inp.fault_class,
                "severity": inp.severity.value,
                "urgency": inp.urgency.value,
                "summary": inp.summary,
            },
        )
        return EscalateAlertOutput(
            escalated=True,
            escalation_id=escalation_id,
            backend="audit_log",
            timestamp=ts,
            channel="agentpv.escalation",
            note=(
                "[MOCK] escalation recorded to local structured log; "
                "polish phase will fan out to operations channels."
            ),
        )
