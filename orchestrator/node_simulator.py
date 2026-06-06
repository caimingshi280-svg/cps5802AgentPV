"""Per-node simulator + classification loop (Component 7 MVP).

Each :class:`NodeRunner` represents one physical PV/BESS asset:

* It holds a deterministic ``numpy.random.Generator`` keyed on
  ``NodeConfig.seed`` so a given (seed, step_number) pair always produces
  the same window.
* On every tick it (1) decides clean vs faulty by Bernoulli draw, (2)
  generates the window via :mod:`simulation`, (3) calls edge ``/predict``
  unless ``integration_mode`` is ``cloud_only``, (4) optionally calls agent
  ``/recommend`` when severity is non-trivial (skipped in ``edge_only``),
  (5) emits a single :class:`api.schemas.OrchestratorEvent` to the writer.

A single transport failure does not crash the loop — the failure is
captured in :attr:`OrchestratorEvent.error` and the loop continues.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.schemas import (
    ALL_FAULT_CLASSES,
    BESS_FAULT_CLASSES,
    PV_FAULT_CLASSES,
    Alert,
    OperatingCondition,
    OrchestratorEvent,
    SensorWindow,
    Severity,
    SystemType,
)
from orchestrator.clients import AgentClient, ClientError, EdgeClient
from orchestrator.event_log import JsonlEventWriter
from simulation.battery_simulator import BESS_FEATURE_NAMES, BatterySimulator
from simulation.fault_injector import inject_fault
from simulation.pv_simulator import PV_FEATURE_NAMES, PVSimulator
from utils.logging_config import get_logger

log = get_logger(__name__)


_PV_NORMAL = "PV_Normal"
_BESS_NORMAL = "BESS_Normal"
_PV_FAULT_CHOICES: tuple[str, ...] = tuple(c for c in PV_FAULT_CLASSES if c != _PV_NORMAL)
_BESS_FAULT_CHOICES: tuple[str, ...] = tuple(
    c for c in BESS_FAULT_CLASSES if c != _BESS_NORMAL
)


def _severity_from_ground_truth_label(label: str) -> Severity:
    if label in (_PV_NORMAL, _BESS_NORMAL):
        return Severity.MONITOR
    if label in ("Inverter_fault", "String_disconnection", "Thermal_anomaly"):
        return Severity.CRITICAL
    return Severity.WARNING


def _sensor_snapshot_from_window(window: SensorWindow) -> dict[str, float | str]:
    if not window.values or not window.feature_names:
        return {}
    last = window.values[-1]
    snap: dict[str, float | str] = {}
    for i, name in enumerate(window.feature_names):
        if i < len(last):
            snap[str(name)] = float(last[i])
    return snap


def _alert_from_window_label(window: SensorWindow, label: str) -> Alert:
    return Alert(
        timestamp=window.timestamp_start,
        system_id=window.system_id,
        system_type=window.system_type,
        fault_class=label,
        severity=_severity_from_ground_truth_label(label),
        confidence=0.9,
        sensor_snapshot=_sensor_snapshot_from_window(window),
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class NodeConfig(BaseModel):
    """Static configuration of one orchestrator node.

    Lives outside :mod:`api.schemas` because it is a *runtime* config rather
    than an interface contract shared across services (rule §3 carve-out).
    """

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1, max_length=64)
    system_id: str = Field(min_length=1, max_length=64)
    system_type: SystemType
    seed: int = Field(ge=0)
    fault_probability: float = Field(ge=0.0, le=1.0)
    fault_classes: tuple[str, ...] = Field(
        default=(),
        description=(
            "Fault classes to sample from when a faulty window is drawn. "
            "If empty, defaults to all faults of the system_type."
        ),
    )
    period_seconds: float = Field(gt=0.0, default=1.0)
    sample_rate_hz: float = Field(gt=0.0, default=1.0)
    window_size: int = Field(gt=0, default=60)
    operating_condition: OperatingCondition = OperatingCondition.HIGH_IRRADIANCE
    integration_mode: Literal["full", "edge_only", "cloud_only"] = Field(
        default="full",
        description=(
            "full: call edge /predict then agent when severity triggers; "
            "edge_only: call edge but never agent; "
            "cloud_only: skip edge, build Alert from simulator label + window snapshot."
        ),
    )

    @field_validator("fault_classes")
    @classmethod
    def fault_classes_known(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for fc in value:
            if fc not in ALL_FAULT_CLASSES:
                raise ValueError(
                    f"unknown fault_class={fc!r}; see api.schemas.ALL_FAULT_CLASSES"
                )
        return value


# ---------------------------------------------------------------------------
# Per-node state (running counters)
# ---------------------------------------------------------------------------


@dataclass
class NodeState:
    step_number: int = 0
    n_alerts: int = 0
    n_recommendations: int = 0
    n_errors: int = 0
    last_event_id: str | None = None
    last_event_at: datetime | None = field(default=None)


# ---------------------------------------------------------------------------
# NodeRunner
# ---------------------------------------------------------------------------


class NodeRunner:
    """A single async task that drives one node end-to-end."""

    AGENT_TRIGGER_SEVERITIES: frozenset[Severity] = frozenset(
        {Severity.WARNING, Severity.CRITICAL}
    )

    def __init__(
        self,
        config: NodeConfig,
        edge: EdgeClient,
        agent: AgentClient,
        writer: JsonlEventWriter,
    ) -> None:
        self.config = config
        self.edge = edge
        self.agent = agent
        self.writer = writer
        self.state = NodeState()
        self._rng = np.random.default_rng(config.seed)

        # Decide which fault catalogue this node samples from.
        if config.fault_classes:
            self._fault_choices = config.fault_classes
        elif config.system_type is SystemType.PV:
            self._fault_choices = _PV_FAULT_CHOICES
        else:
            self._fault_choices = _BESS_FAULT_CHOICES

        # One simulator per node — cheap, stateless apart from its rng.
        if config.system_type is SystemType.PV:
            self._pv_sim: PVSimulator | None = PVSimulator(seed=int(config.seed))
            self._bess_sim: BatterySimulator | None = None
            self._feature_names = PV_FEATURE_NAMES
            self._normal_label = _PV_NORMAL
        else:
            self._pv_sim = None
            self._bess_sim = BatterySimulator(seed=int(config.seed))
            self._feature_names = BESS_FEATURE_NAMES
            self._normal_label = _BESS_NORMAL

    # ------------------------------------------------------------------
    # Synthetic window generation
    # ------------------------------------------------------------------

    def _decide_label(self) -> str:
        if self._rng.random() < self.config.fault_probability:
            idx = int(self._rng.integers(0, len(self._fault_choices)))
            return self._fault_choices[idx]
        return self._normal_label

    def _generate_window(self) -> tuple[SensorWindow, str]:
        """Generate one labeled window. Returns (window, ground_truth_label)."""

        label = self._decide_label()
        if self.config.system_type is SystemType.PV:
            assert self._pv_sim is not None
            clean = self._pv_sim.simulate(
                condition=self.config.operating_condition,
                window_size=self.config.window_size,
            )
        else:
            assert self._bess_sim is not None
            clean = self._bess_sim.simulate(
                condition=self.config.operating_condition,
                window_size=self.config.window_size,
            )
        if label != self._normal_label:
            arr = inject_fault(clean, label=label, rng=self._rng)
        else:
            arr = clean

        window = SensorWindow(
            timestamp_start=datetime.now(UTC),
            system_id=self.config.system_id,
            system_type=self.config.system_type,
            sample_rate_hz=self.config.sample_rate_hz,
            window_size=self.config.window_size,
            feature_names=list(self._feature_names),
            values=arr.astype(float).tolist(),
            operating_condition=self.config.operating_condition,
            injected_fault=None if label == self._normal_label else label,
        )
        return window, label

    # ------------------------------------------------------------------
    # One end-to-end step
    # ------------------------------------------------------------------

    async def step(self) -> OrchestratorEvent:
        """Generate one window, classify, optionally recommend, emit one event."""

        step_idx = self.state.step_number
        self.state.step_number += 1

        window, label = self._generate_window()

        alert: Alert | None = None
        recommendation = None
        error: str | None = None
        agent_elapsed_ms: float | None = None
        edge_elapsed_ms: float = 0.0
        mode = self.config.integration_mode

        if mode == "cloud_only":
            alert = _alert_from_window_label(window, label)
            self.state.n_alerts += 1
        else:
            edge_t0 = time.perf_counter()
            edge_result = await self.edge.predict(window)
            edge_elapsed_ms = (time.perf_counter() - edge_t0) * 1000.0

            if isinstance(edge_result, ClientError):
                error = f"edge_predict_failed: {edge_result.message}"
                self.state.n_errors += 1
            else:
                alert = edge_result
                self.state.n_alerts += 1

        if error is None and alert is not None:
            agent_allowed = (
                mode != "edge_only" and alert.severity in self.AGENT_TRIGGER_SEVERITIES
            )
            if agent_allowed:
                agent_result, agent_elapsed_ms = await self.agent.recommend_timed(alert)
                if isinstance(agent_result, ClientError):
                    error = f"agent_recommend_failed: {agent_result.message}"
                    self.state.n_errors += 1
                else:
                    recommendation = agent_result
                    self.state.n_recommendations += 1

        event = OrchestratorEvent(
            event_id=uuid.uuid4().hex,
            timestamp=datetime.now(UTC),
            node_id=self.config.node_id,
            system_id=self.config.system_id,
            system_type=self.config.system_type,
            step_number=step_idx,
            ground_truth_label=label,
            alert=alert,
            recommendation=recommendation,
            error=error,
            edge_elapsed_ms=edge_elapsed_ms,
            agent_elapsed_ms=agent_elapsed_ms,
        )
        self.writer.append(event)
        self.state.last_event_id = event.event_id
        self.state.last_event_at = event.timestamp
        log.info(
            "node_step_emitted",
            extra={
                "node_id": self.config.node_id,
                "step": step_idx,
                "label": label,
                "alert": alert.fault_class if alert else None,
                "severity": alert.severity.value if alert else None,
                "has_recommendation": recommendation is not None,
                "has_error": error is not None,
                "edge_ms": round(edge_elapsed_ms, 2),
                "agent_ms": round(agent_elapsed_ms, 2) if agent_elapsed_ms else None,
            },
        )
        return event

    # ------------------------------------------------------------------
    # Forever loop
    # ------------------------------------------------------------------

    async def run_forever(self, stop_event: asyncio.Event) -> None:
        """Loop until ``stop_event`` is set, sleeping ``period_seconds`` between steps."""

        log.info(
            "node_runner_started",
            extra={"node_id": self.config.node_id, "period_s": self.config.period_seconds},
        )
        try:
            while not stop_event.is_set():
                try:
                    await self.step()
                except Exception:  # noqa: BLE001 — guard the whole loop
                    log.exception(
                        "node_step_unexpected_error",
                        extra={"node_id": self.config.node_id},
                    )
                    self.state.n_errors += 1
                # 即使被取消，也允许 step 完成并 emit event。
                try:
                    await asyncio.wait_for(
                        stop_event.wait(), timeout=self.config.period_seconds
                    )
                except TimeoutError:
                    pass
        finally:
            log.info(
                "node_runner_stopped",
                extra={
                    "node_id": self.config.node_id,
                    "n_steps": self.state.step_number,
                    "n_alerts": self.state.n_alerts,
                    "n_recommendations": self.state.n_recommendations,
                    "n_errors": self.state.n_errors,
                },
            )
