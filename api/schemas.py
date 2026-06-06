"""Typed API contracts for AgentPV.

This module defines the stable data contracts shared by simulation, edge
inference, the cloud agent, and the dashboard. Keep these models small,
strict, and backwards compatible.

中文说明
--------
全项目的「单一事实来源」数据模型：告警、推荐、传感器窗口等均在此定义；
修改字段须同步 ``docs/alert_schema.json`` 与对端服务契约。
"""

from __future__ import annotations

import math
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

JsonScalar = str | int | float | bool | None
SensorSnapshot = dict[str, JsonScalar]


class StrictBaseModel(BaseModel):
    """Base model that rejects undeclared fields."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class SystemType(str, Enum):
    """Supported monitored system types."""

    PV = "PV"
    BESS = "BESS"


class Severity(str, Enum):
    """Edge alert severity levels required by the assignment."""

    MONITOR = "monitor"
    WARNING = "warning"
    CRITICAL = "critical"


class Urgency(str, Enum):
    """Agent recommendation urgency levels required by the assignment."""

    IMMEDIATE = "immediate"
    SCHEDULED = "scheduled"
    MONITOR = "monitor"


class AgentConfidence(str, Enum):
    """Agent self-assessed confidence levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class OperatingCondition(str, Enum):
    """Minimum required operating conditions for generated samples."""

    HIGH_IRRADIANCE = "high_irradiance"
    LOW_IRRADIANCE = "low_irradiance"
    HIGH_TEMPERATURE = "high_temperature"


PV_FAULT_CLASSES: tuple[str, ...] = (
    "PV_Normal",
    "Partial_shading",
    "Soiling",
    "Bypass_diode_fault",
    "String_disconnection",
    "Inverter_fault",
    "Degradation",
)

BESS_FAULT_CLASSES: tuple[str, ...] = (
    "BESS_Normal",
    "Capacity_fade",
    "Internal_resistance_increase",
    "Thermal_anomaly",
    "Cell_imbalance",
)

ALL_FAULT_CLASSES: tuple[str, ...] = PV_FAULT_CLASSES + BESS_FAULT_CLASSES


def _is_finite_number(value: Any) -> bool:
    """Return True if value is a finite int or float."""

    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class SensorWindow(StrictBaseModel):
    """A fixed-size time-series input window for edge inference."""

    timestamp_start: datetime = Field(description="ISO 8601 start timestamp.")
    system_id: str = Field(min_length=1, max_length=64)
    system_type: SystemType
    sample_rate_hz: float = Field(gt=0)
    window_size: int = Field(gt=0, description="Number of time steps in the window.")
    feature_names: list[str] = Field(min_length=1)
    values: list[list[float]] = Field(description="Shape: (window_size, num_features).")
    operating_condition: OperatingCondition | None = None
    injected_fault: str | None = Field(default=None, description="Optional demo/test fault label.")

    @field_validator("feature_names")
    @classmethod
    def feature_names_must_be_unique(cls, value: list[str]) -> list[str]:
        """Ensure features have stable unique names."""

        if len(value) != len(set(value)):
            raise ValueError("feature_names must be unique")
        if any(not name.strip() for name in value):
            raise ValueError("feature_names must not contain empty strings")
        return value

    @model_validator(mode="after")
    def values_must_match_declared_shape(self) -> SensorWindow:
        """Validate the time-series matrix shape and numeric values."""

        if len(self.values) != self.window_size:
            raise ValueError("values row count must equal window_size")

        expected_features = len(self.feature_names)
        for row_index, row in enumerate(self.values):
            if len(row) != expected_features:
                raise ValueError(
                    f"values[{row_index}] has {len(row)} features; expected {expected_features}"
                )
            if not all(_is_finite_number(v) for v in row):
                raise ValueError(f"values[{row_index}] contains non-finite numeric values")
        return self


class Alert(StrictBaseModel):
    """Fixed edge-to-cloud alert contract from the assignment PDF."""

    timestamp: datetime = Field(description="ISO 8601 timestamp.")
    system_id: str = Field(min_length=1, max_length=64)
    system_type: SystemType
    fault_class: str = Field(min_length=1, description="Predicted class label.")
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    sensor_snapshot: SensorSnapshot = Field(default_factory=dict)

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_finite(cls, value: float) -> float:
        """Reject NaN and infinite confidence values."""

        if not math.isfinite(value):
            raise ValueError("confidence must be finite")
        return value

    @field_validator("sensor_snapshot")
    @classmethod
    def snapshot_values_must_be_json_scalars(cls, value: SensorSnapshot) -> SensorSnapshot:
        """Keep alert snapshots JSON-serializable and simple."""

        for key, item in value.items():
            if not key.strip():
                raise ValueError("sensor_snapshot keys must not be empty")
            if isinstance(item, float) and not math.isfinite(item):
                raise ValueError(f"sensor_snapshot[{key}] must be finite")
        return value


class ReasoningStep(StrictBaseModel):
    """One auditable step in the agent reasoning trace."""

    step: int = Field(ge=0)
    phase: Literal["observe", "reason", "act", "reflect", "report"]
    thought: str = Field(min_length=1)
    action: str | None = None
    args: dict[str, JsonScalar] | None = None
    result_summary: str | None = None


class Recommendation(StrictBaseModel):
    """Structured LLM agent output required by the assignment."""

    recommended_action: str = Field(min_length=1)
    urgency: Urgency
    reasoning_trace: list[ReasoningStep] = Field(min_length=1)
    knowledge_sources: list[str] = Field(default_factory=list)
    confidence: AgentConfidence

    @model_validator(mode="after")
    def critical_logic_must_be_explainable(self) -> Recommendation:
        """Require citations when the agent is highly confident."""

        if self.confidence is AgentConfidence.HIGH and not self.knowledge_sources:
            raise ValueError("high confidence recommendations must include knowledge_sources")
        return self


class ErrorResponse(StrictBaseModel):
    """Structured API error response shared by all services."""

    error_code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)
    retry_after_s: int | None = Field(default=None, ge=0)


class ToolError(StrictBaseModel):
    """Structured error envelope returned by every agent tool failure.

    Tools never raise; failures must be surfaced via this contract so the
    agent's reflect step can decide how to recover.
    """

    error_code: Literal["VALIDATION", "TIMEOUT", "INTERNAL", "NOT_FOUND"]
    message: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    trace_id: str = Field(min_length=1)


class HealthResponse(StrictBaseModel):
    """Service health response."""

    status: Literal["ok", "degraded"]
    service: str = Field(min_length=1)
    version: str = Field(default="0.1.0")


# ---------------------------------------------------------------------------
# Simulation-layer contracts (Component 1 output → consumed by training,
# evaluation, and the dashboard's playback view).
# ---------------------------------------------------------------------------


class RawSample(StrictBaseModel):
    """Labeled training sample produced by the simulation layer.

    Wraps a :class:`SensorWindow` with a ground-truth fault class label and
    requires :attr:`SensorWindow.operating_condition` to be set so downstream
    stratification by condition is possible.
    """

    window: SensorWindow
    label: str = Field(min_length=1, description="Ground-truth fault class label.")

    @field_validator("label")
    @classmethod
    def label_must_be_known(cls, value: str) -> str:
        """Constrain labels to the documented PV/BESS taxonomy."""

        if value not in ALL_FAULT_CLASSES:
            raise ValueError(
                f"label={value!r} not in known fault classes; "
                f"see api.schemas.ALL_FAULT_CLASSES"
            )
        return value

    @model_validator(mode="after")
    def label_must_match_system_type(self) -> RawSample:
        """Forbid labelling a PV system with a BESS fault and vice versa."""

        if self.window.system_type is SystemType.PV and self.label not in PV_FAULT_CLASSES:
            raise ValueError(f"PV system cannot carry BESS label {self.label!r}")
        if self.window.system_type is SystemType.BESS and self.label not in BESS_FAULT_CLASSES:
            raise ValueError(f"BESS system cannot carry PV label {self.label!r}")
        if self.window.operating_condition is None:
            raise ValueError("RawSample requires window.operating_condition to be set")
        return self


class SplitName(str, Enum):
    """Dataset split names."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class DatasetMetadata(StrictBaseModel):
    """Reproducibility metadata persisted alongside a generated dataset."""

    schema_version: str = "0.1.0"
    generated_at: datetime
    seed: int = Field(ge=0)
    sample_rate_hz: float = Field(gt=0)
    window_size: int = Field(gt=0)
    pv_feature_names: list[str] = Field(min_length=1)
    bess_feature_names: list[str] = Field(min_length=1)
    n_samples: int = Field(ge=0)
    splits: dict[SplitName, int]
    class_distribution: dict[str, int]
    operating_condition_distribution: dict[OperatingCondition, int]
    notes: str = ""

    @model_validator(mode="after")
    def totals_must_be_consistent(self) -> DatasetMetadata:
        """Cross-check that splits and class counts agree with n_samples."""

        if sum(self.splits.values()) != self.n_samples:
            raise ValueError("sum(splits) must equal n_samples")
        if sum(self.class_distribution.values()) != self.n_samples:
            raise ValueError("sum(class_distribution) must equal n_samples")
        if sum(self.operating_condition_distribution.values()) != self.n_samples:
            raise ValueError("sum(operating_condition_distribution) must equal n_samples")
        return self


# ---------------------------------------------------------------------------
# Orchestrator-layer contracts (Component 7 output → consumed by the
# dashboard and Component 5 agent benchmark replay).
# ---------------------------------------------------------------------------


class OrchestratorEvent(StrictBaseModel):
    """One end-to-end event emitted by an orchestrator node.

    Captures the full chain ``simulator window → edge alert → optional agent
    recommendation`` so that the dashboard can render one row per event and
    the agent benchmark can replay scenarios deterministically.

    Either :attr:`alert` or :attr:`error` MUST be present so a consumer can
    distinguish "edge classified as monitor" from "edge call failed".
    """

    event_id: str = Field(min_length=1, description="UUID4 hex; one per event.")
    timestamp: datetime = Field(description="ISO 8601 — when the event was emitted.")
    node_id: str = Field(
        min_length=1,
        max_length=64,
        description="Stable id of the simulator node (1:1 with system_id in MVP).",
    )
    system_id: str = Field(min_length=1, max_length=64)
    system_type: SystemType
    step_number: int = Field(ge=0, description="Monotonic per-node step counter.")
    ground_truth_label: str = Field(
        min_length=1,
        description="Injected fault class (or *_Normal for clean windows).",
    )
    alert: Alert | None = None
    recommendation: Recommendation | None = None
    error: str | None = Field(
        default=None, description="Set if the chain failed at any step."
    )
    edge_elapsed_ms: float | None = Field(default=None, ge=0.0)
    agent_elapsed_ms: float | None = Field(default=None, ge=0.0)

    @field_validator("ground_truth_label")
    @classmethod
    def label_must_be_known(cls, value: str) -> str:
        """Constrain ground-truth labels to the documented taxonomy."""

        if value not in ALL_FAULT_CLASSES:
            raise ValueError(
                f"ground_truth_label={value!r} not in known fault classes; "
                f"see api.schemas.ALL_FAULT_CLASSES"
            )
        return value

    @model_validator(mode="after")
    def must_have_alert_or_error(self) -> OrchestratorEvent:
        """At least one of alert/error must be present (else the event is empty)."""

        if self.alert is None and self.error is None:
            raise ValueError("OrchestratorEvent requires either alert or error")
        if self.recommendation is not None and self.alert is None:
            raise ValueError("recommendation cannot be set without an alert")
        return self
