"""Tool: system_history — return recent alerts for an asset (MOCK backend).

Per project rule §12 every placeholder backend must be **clearly marked**.
The output's ``backend`` field is set to ``"mock"`` and history entries are
prefixed with ``[MOCK]`` so downstream consumers (and the academic report)
can distinguish synthetic from real data.

Polish-phase upgrade: swap to a real database client. The contract on this
file is stable so call sites won't change.
"""
from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, Field

from api.schemas import Severity, SystemType
from tools.base import Tool


class SystemHistoryInput(BaseModel):
    system_id: str = Field(min_length=1, max_length=64)
    system_type: SystemType
    hours: int = Field(default=24, ge=1, le=720)


class HistoryEntry(BaseModel):
    timestamp: datetime
    fault_class: str
    severity: Severity
    note: str  # 始终以 "[MOCK]" 前缀开头


class SystemHistoryOutput(BaseModel):
    system_id: str
    backend: str = Field(description="One of: 'mock' | 'database'")
    window_hours: int
    entries: list[HistoryEntry]
    n_critical: int
    n_warning: int
    n_monitor: int


# 故障序列字典——给定 (system_type, hash) 决定一条 deterministic mock 历史。
_PV_HISTORY_TEMPLATES: list[list[tuple[str, Severity]]] = [
    [("PV_Normal", Severity.MONITOR)],
    [
        ("PV_Normal", Severity.MONITOR),
        ("Soiling", Severity.MONITOR),
        ("Partial_shading", Severity.WARNING),
    ],
    [
        ("Soiling", Severity.MONITOR),
        ("Partial_shading", Severity.WARNING),
        ("Bypass_diode_fault", Severity.WARNING),
    ],
]
_BESS_HISTORY_TEMPLATES: list[list[tuple[str, Severity]]] = [
    [("BESS_Normal", Severity.MONITOR)],
    [
        ("BESS_Normal", Severity.MONITOR),
        ("Capacity_fade", Severity.MONITOR),
    ],
    [
        ("Capacity_fade", Severity.MONITOR),
        ("Internal_resistance_increase", Severity.WARNING),
        ("Cell_imbalance", Severity.WARNING),
    ],
]


def _select_template(system_id: str, system_type: SystemType) -> list[tuple[str, Severity]]:
    """Deterministic template selection so tests are stable."""

    digest = hashlib.sha256(system_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:4], 16)
    pool = (
        _PV_HISTORY_TEMPLATES if system_type is SystemType.PV else _BESS_HISTORY_TEMPLATES
    )
    return pool[bucket % len(pool)]


class SystemHistoryTool(Tool[SystemHistoryInput, SystemHistoryOutput]):
    """Mock implementation of system history.

    Returns a deterministic short history derived from the ``system_id`` so
    integration tests are reproducible without a database.
    """

    name = "system_history"
    description = (
        "Return recent fault alerts for a given system_id over the last N hours. "
        "MVP backend: mock — entries are deterministic stubs prefixed [MOCK]."
    )
    input_model = SystemHistoryInput
    output_model = SystemHistoryOutput
    timeout_s = 3.0

    async def _run(self, inp: SystemHistoryInput) -> SystemHistoryOutput:
        template = _select_template(inp.system_id, inp.system_type)
        # 最近 N 小时内均匀撒点。
        now = datetime.now(UTC)
        n = len(template)
        entries: list[HistoryEntry] = []
        for i, (fault_class, severity) in enumerate(template):
            ts = now - timedelta(hours=(inp.hours / max(n, 1)) * (n - i))
            entries.append(
                HistoryEntry(
                    timestamp=ts,
                    fault_class=fault_class,
                    severity=severity,
                    note=f"[MOCK] historical entry {i + 1}/{n}",
                )
            )
        n_critical = sum(1 for e in entries if e.severity is Severity.CRITICAL)
        n_warning = sum(1 for e in entries if e.severity is Severity.WARNING)
        n_monitor = sum(1 for e in entries if e.severity is Severity.MONITOR)
        return SystemHistoryOutput(
            system_id=inp.system_id,
            backend="mock",
            window_hours=inp.hours,
            entries=entries,
            n_critical=n_critical,
            n_warning=n_warning,
            n_monitor=n_monitor,
        )
