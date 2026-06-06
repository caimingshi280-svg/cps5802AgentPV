"""Logits → :class:`api.schemas.Alert` post-processing.

Maps a model's softmax output to:

* ``fault_class``  — the predicted label string (taxonomy from
  :mod:`api.schemas`),
* ``severity``     — one of monitor / warning / critical, derived from the
  fault taxonomy and the model's confidence,
* ``confidence``   — the softmax probability of the predicted class.

Severity policy (defended in the academic report):

* **critical** is reserved for faults that can cause immediate safety
  damage if ignored: ``Inverter_fault``, ``String_disconnection``,
  ``Thermal_anomaly``.
* **warning** covers degraded operation that requires near-term
  intervention: ``Partial_shading``, ``Bypass_diode_fault``,
  ``Cell_imbalance``, ``Internal_resistance_increase``, plus any
  ``Soiling`` prediction that the model is highly confident about.
* **monitor** is the default for healthy operation, soft / slow
  degradations (``Degradation``, ``Capacity_fade``), and low-confidence
  soiling predictions.

Confidence threshold ``high_conf_threshold`` (default 0.85) is exposed
so it can be tuned during the polish phase.
"""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from api.schemas import (
    BESS_FAULT_CLASSES,
    PV_FAULT_CLASSES,
    Alert,
    SensorSnapshot,
    Severity,
    SystemType,
)

# Hard-coded severity buckets per fault class. Anything not listed is treated
# as MONITOR (this fail-safe matters: a future fault class added without
# updating this map will at least not be silently elevated to CRITICAL).
_CRITICAL_FAULTS: frozenset[str] = frozenset(
    {"Inverter_fault", "String_disconnection", "Thermal_anomaly"}
)
_WARNING_FAULTS: frozenset[str] = frozenset(
    {
        "Partial_shading",
        "Bypass_diode_fault",
        "Cell_imbalance",
        "Internal_resistance_increase",
    }
)
_MONITOR_FAULTS: frozenset[str] = frozenset(
    {"PV_Normal", "BESS_Normal", "Degradation", "Capacity_fade"}
)
# Soiling escalates to warning only when the model is confident.
_CONFIDENCE_GATED_FAULTS: frozenset[str] = frozenset({"Soiling"})


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""

    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def severity_for(fault_class: str, confidence: float, *, high_conf_threshold: float = 0.85) -> Severity:
    """Map (fault_class, confidence) to a :class:`Severity` level."""

    if fault_class in _CRITICAL_FAULTS:
        return Severity.CRITICAL
    if fault_class in _WARNING_FAULTS:
        return Severity.WARNING
    if fault_class in _MONITOR_FAULTS:
        return Severity.MONITOR
    if fault_class in _CONFIDENCE_GATED_FAULTS:
        return Severity.WARNING if confidence >= high_conf_threshold else Severity.MONITOR
    # Unknown class — fail safe to MONITOR rather than CRITICAL.
    return Severity.MONITOR


def labels_for(system_type: SystemType) -> tuple[str, ...]:
    """Return the label tuple matching ``system_type``."""

    return PV_FAULT_CLASSES if system_type is SystemType.PV else BESS_FAULT_CLASSES


def logits_to_alert(
    *,
    logits: np.ndarray,
    system_id: str,
    system_type: SystemType,
    sensor_snapshot: SensorSnapshot | None = None,
    timestamp: datetime | None = None,
    high_conf_threshold: float = 0.85,
) -> Alert:
    """Convert a single 1-D logits vector into a validated :class:`Alert`.

    Parameters
    ----------
    logits : np.ndarray
        1-D float vector with one entry per fault class. Length must match
        the fault taxonomy length for ``system_type``.
    system_id : str
        Identifier of the monitored asset.
    system_type : SystemType
        Selects which fault taxonomy applies.
    sensor_snapshot : SensorSnapshot, optional
        Recent sensor scalars to attach for the cloud agent. Defaults to ``{}``.
    timestamp : datetime, optional
        Alert wall-clock timestamp. Defaults to ``datetime.now(timezone.utc)``.
    high_conf_threshold : float
        Confidence threshold above which gated faults escalate to warning.
    """

    if logits.ndim != 1:
        raise ValueError(f"logits must be 1-D, got shape {logits.shape}")

    labels = labels_for(system_type)
    if logits.shape[0] != len(labels):
        raise ValueError(
            f"logits length {logits.shape[0]} != taxonomy length {len(labels)} "
            f"for system_type={system_type.value}"
        )

    probs = _softmax(logits.astype(np.float64))
    pred_idx = int(np.argmax(probs))
    fault_class = labels[pred_idx]
    confidence = float(probs[pred_idx])

    severity = severity_for(
        fault_class, confidence, high_conf_threshold=high_conf_threshold
    )

    return Alert(
        timestamp=timestamp or datetime.now(UTC),
        system_id=system_id,
        system_type=system_type,
        fault_class=fault_class,
        severity=severity,
        confidence=confidence,
        sensor_snapshot=sensor_snapshot or {},
    )
