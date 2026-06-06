"""Tool: estimate_rul — rule-based remaining-useful-life estimate.

This is **not** a learned model in MVP — it's a deterministic policy table
keyed on fault class and severity. The output exposes ``backend="rule_based"``
and the ``confidence_band`` field deliberately reports a wide band so
operators are not misled (rule §12 — placeholder must be honest).

Polish phase: replace with a per-asset survival-analysis model. Contract
on this module is stable.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from api.schemas import (
    BESS_FAULT_CLASSES,
    PV_FAULT_CLASSES,
    Severity,
    SystemType,
)
from tools.base import Tool


class EstimateRulInput(BaseModel):
    system_id: str = Field(min_length=1, max_length=64)
    system_type: SystemType
    fault_class: str = Field(min_length=1)
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)


class EstimateRulOutput(BaseModel):
    system_id: str
    backend: str = Field(description="rule_based | survival_model")
    rul_days_estimate: int
    rul_days_lower: int
    rul_days_upper: int
    rationale: str  # human-readable explanation
    requires_immediate_action: bool


# 故障级 → (中位 RUL 天, 区间宽度)；critical 故障默认 0 天，需立即行动。
_RUL_TABLE: dict[str, tuple[int, int]] = {
    "PV_Normal": (3650, 365),
    "BESS_Normal": (3650, 365),
    "Soiling": (180, 60),
    "Partial_shading": (90, 45),
    "Bypass_diode_fault": (45, 30),
    "Degradation": (1095, 365),
    "Capacity_fade": (730, 365),
    "Internal_resistance_increase": (180, 90),
    "Cell_imbalance": (60, 30),
}
_CRITICAL_FAULTS = {"Inverter_fault", "String_disconnection", "Thermal_anomaly"}


class EstimateRulTool(Tool[EstimateRulInput, EstimateRulOutput]):
    """Rule-based RUL estimator."""

    name = "estimate_rul"
    description = (
        "Return a coarse remaining-useful-life estimate (days) based on the "
        "fault class and severity. Rule-based — wide confidence band by design."
    )
    input_model = EstimateRulInput
    output_model = EstimateRulOutput
    timeout_s = 2.0

    async def _run(self, inp: EstimateRulInput) -> EstimateRulOutput:
        # 校验 fault_class 属于已知 taxonomy（rule §3）。
        known = set(PV_FAULT_CLASSES) | set(BESS_FAULT_CLASSES)
        if inp.fault_class not in known:
            raise ValueError(
                f"Unknown fault_class={inp.fault_class!r}; "
                f"see api.schemas.ALL_FAULT_CLASSES"
            )

        if inp.fault_class in _CRITICAL_FAULTS or inp.severity is Severity.CRITICAL:
            return EstimateRulOutput(
                system_id=inp.system_id,
                backend="rule_based",
                rul_days_estimate=0,
                rul_days_lower=0,
                rul_days_upper=1,
                rationale=(
                    f"Critical fault {inp.fault_class} requires immediate "
                    "intervention; RUL set to 0 days regardless of confidence."
                ),
                requires_immediate_action=True,
            )

        median, half_width = _RUL_TABLE.get(inp.fault_class, (365, 180))
        # Confidence shrinks the lower bound: 高置信度 → 下限更接近 median。
        adj_half = max(int(half_width * (1.0 - inp.confidence)), 7)
        lower = max(median - adj_half, 0)
        upper = median + adj_half
        rationale = (
            f"Rule-based estimate for {inp.fault_class} ({inp.severity.value}, "
            f"confidence={inp.confidence:.2f}): median {median} days, "
            f"band ±{adj_half} days."
        )
        return EstimateRulOutput(
            system_id=inp.system_id,
            backend="rule_based",
            rul_days_estimate=median,
            rul_days_lower=lower,
            rul_days_upper=upper,
            rationale=rationale,
            requires_immediate_action=False,
        )
