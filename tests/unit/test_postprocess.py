"""Unit tests for :mod:`inference.postprocess`."""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from api.schemas import (
    BESS_FAULT_CLASSES,
    PV_FAULT_CLASSES,
    Alert,
    Severity,
    SystemType,
)
from inference.postprocess import (
    labels_for,
    logits_to_alert,
    severity_for,
)

# ---------------------------------------------------------------------------
# Severity policy
# ---------------------------------------------------------------------------


def test_severity_critical_for_inverter_fault() -> None:
    assert severity_for("Inverter_fault", confidence=0.5) is Severity.CRITICAL


def test_severity_critical_for_string_disconnection() -> None:
    assert severity_for("String_disconnection", confidence=0.99) is Severity.CRITICAL


def test_severity_critical_for_thermal_anomaly() -> None:
    assert severity_for("Thermal_anomaly", confidence=0.6) is Severity.CRITICAL


def test_severity_warning_for_partial_shading() -> None:
    assert severity_for("Partial_shading", confidence=0.7) is Severity.WARNING


def test_severity_warning_for_cell_imbalance() -> None:
    assert severity_for("Cell_imbalance", confidence=0.4) is Severity.WARNING


def test_severity_monitor_for_normal() -> None:
    assert severity_for("PV_Normal", confidence=0.99) is Severity.MONITOR
    assert severity_for("BESS_Normal", confidence=0.99) is Severity.MONITOR


def test_severity_monitor_for_slow_degradation() -> None:
    assert severity_for("Degradation", confidence=0.99) is Severity.MONITOR
    assert severity_for("Capacity_fade", confidence=0.99) is Severity.MONITOR


def test_severity_soiling_gated_by_confidence() -> None:
    assert severity_for("Soiling", confidence=0.5) is Severity.MONITOR
    assert severity_for("Soiling", confidence=0.9) is Severity.WARNING


def test_severity_soiling_threshold_can_be_tuned() -> None:
    assert severity_for("Soiling", confidence=0.9, high_conf_threshold=0.95) is Severity.MONITOR
    assert severity_for("Soiling", confidence=0.96, high_conf_threshold=0.95) is Severity.WARNING


def test_severity_unknown_class_falls_back_to_monitor() -> None:
    """Forward-compat: unknown classes must NOT silently elevate to critical."""
    assert severity_for("Made_up_fault", confidence=0.99) is Severity.MONITOR


# ---------------------------------------------------------------------------
# Logits → Alert
# ---------------------------------------------------------------------------


def test_labels_for_returns_correct_taxonomy() -> None:
    assert labels_for(SystemType.PV) == PV_FAULT_CLASSES
    assert labels_for(SystemType.BESS) == BESS_FAULT_CLASSES


def test_logits_to_alert_returns_validated_alert() -> None:
    logits = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # argmax -> PV_Normal
    alert = logits_to_alert(
        logits=logits,
        system_id="PV_test_001",
        system_type=SystemType.PV,
    )
    assert isinstance(alert, Alert)
    assert alert.fault_class == "PV_Normal"
    assert alert.severity is Severity.MONITOR
    assert 0.99 < alert.confidence <= 1.0


def test_logits_to_alert_critical_path() -> None:
    """Argmax inverter -> CRITICAL severity."""
    inverter_idx = PV_FAULT_CLASSES.index("Inverter_fault")
    logits = np.full(len(PV_FAULT_CLASSES), -5.0)
    logits[inverter_idx] = 5.0
    alert = logits_to_alert(
        logits=logits,
        system_id="PV_007",
        system_type=SystemType.PV,
    )
    assert alert.fault_class == "Inverter_fault"
    assert alert.severity is Severity.CRITICAL


def test_logits_to_alert_uses_provided_timestamp_and_snapshot() -> None:
    ts = datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC)
    snapshot = {"P_dc": 12.5, "T_mod": 45.0}
    logits = np.zeros(len(BESS_FAULT_CLASSES))
    logits[0] = 1.0
    alert = logits_to_alert(
        logits=logits,
        system_id="BESS_03",
        system_type=SystemType.BESS,
        sensor_snapshot=snapshot,
        timestamp=ts,
    )
    assert alert.timestamp == ts
    assert alert.sensor_snapshot == snapshot


def test_logits_to_alert_rejects_wrong_dim() -> None:
    with pytest.raises(ValueError, match="1-D"):
        logits_to_alert(
            logits=np.zeros((2, 7)),
            system_id="x",
            system_type=SystemType.PV,
        )


def test_logits_to_alert_rejects_wrong_taxonomy_length() -> None:
    """PV taxonomy has 7 classes; passing 5 must fail loudly."""
    with pytest.raises(ValueError, match="taxonomy length"):
        logits_to_alert(
            logits=np.zeros(5),
            system_id="x",
            system_type=SystemType.PV,
        )
