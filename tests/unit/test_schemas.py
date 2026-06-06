"""Smoke tests for the AgentPV core contracts in :mod:`api.schemas`."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.schemas import (
    ALL_FAULT_CLASSES,
    BESS_FAULT_CLASSES,
    PV_FAULT_CLASSES,
    AgentConfidence,
    Alert,
    DatasetMetadata,
    OperatingCondition,
    RawSample,
    Recommendation,
    SensorWindow,
    Severity,
    SplitName,
    SystemType,
    ToolError,
    Urgency,
)


def _good_alert_payload() -> dict:
    return {
        "timestamp": "2026-03-15T14:23:00+00:00",
        "system_id": "PV_003",
        "system_type": "PV",
        "fault_class": "Partial_shading",
        "severity": "warning",
        "confidence": 0.91,
        "sensor_snapshot": {"V_dc": 520.1, "I_dc": 4.2},
    }


def _good_window_payload(window_size: int = 3, n_features: int = 2) -> dict:
    return {
        "timestamp_start": "2026-03-15T14:00:00+00:00",
        "system_id": "PV_001",
        "system_type": "PV",
        "sample_rate_hz": 1.0,
        "window_size": window_size,
        "feature_names": [f"f{i}" for i in range(n_features)],
        "values": [[float(i + j) for j in range(n_features)] for i in range(window_size)],
    }


# ----------------------------- Alert ---------------------------------------


def test_alert_accepts_valid_payload():
    alert = Alert.model_validate(_good_alert_payload())
    assert alert.system_type is SystemType.PV
    assert alert.severity is Severity.WARNING
    assert 0.0 <= alert.confidence <= 1.0


def test_alert_rejects_extra_field():
    payload = _good_alert_payload()
    payload["nonsense_field"] = True
    with pytest.raises(ValidationError):
        Alert.model_validate(payload)


def test_alert_rejects_unknown_severity():
    payload = _good_alert_payload()
    payload["severity"] = "banana"
    with pytest.raises(ValidationError):
        Alert.model_validate(payload)


def test_alert_rejects_out_of_range_confidence():
    payload = _good_alert_payload()
    payload["confidence"] = 1.5
    with pytest.raises(ValidationError):
        Alert.model_validate(payload)


def test_alert_rejects_unknown_system_type():
    payload = _good_alert_payload()
    payload["system_type"] = "WIND"
    with pytest.raises(ValidationError):
        Alert.model_validate(payload)


# --------------------------- SensorWindow ----------------------------------


def test_sensor_window_validates_shape():
    window = SensorWindow.model_validate(_good_window_payload())
    assert window.window_size == 3
    assert len(window.feature_names) == 2


def test_sensor_window_rejects_shape_mismatch():
    payload = _good_window_payload()
    payload["values"] = payload["values"][:-1]  # one row short
    with pytest.raises(ValidationError):
        SensorWindow.model_validate(payload)


def test_sensor_window_rejects_duplicate_feature_names():
    payload = _good_window_payload(n_features=2)
    payload["feature_names"] = ["dup", "dup"]
    with pytest.raises(ValidationError):
        SensorWindow.model_validate(payload)


def test_sensor_window_rejects_non_finite_values():
    payload = _good_window_payload()
    payload["values"][0][0] = float("nan")
    with pytest.raises(ValidationError):
        SensorWindow.model_validate(payload)


# --------------------------- Recommendation --------------------------------


def test_recommendation_high_confidence_requires_sources():
    payload = {
        "recommended_action": "Inspect string 3 for shading.",
        "urgency": "scheduled",
        "reasoning_trace": [
            {"step": 0, "phase": "observe", "thought": "Received warning alert"},
        ],
        "knowledge_sources": [],
        "confidence": "high",
    }
    with pytest.raises(ValidationError):
        Recommendation.model_validate(payload)


def test_recommendation_accepts_full_trace():
    payload = {
        "recommended_action": "Isolate string 3 if I_dc remains <50% for 30 min.",
        "urgency": "scheduled",
        "reasoning_trace": [
            {"step": 0, "phase": "observe", "thought": "Warning alert received"},
            {
                "step": 1,
                "phase": "act",
                "thought": "Need fault remediation knowledge",
                "action": "retrieve_knowledge",
                "args": {"query": "partial shading remediation"},
                "result_summary": "Found 2 docs",
            },
        ],
        "knowledge_sources": ["pv_partial_shading", "action_isolate_string"],
        "confidence": "high",
    }
    rec = Recommendation.model_validate(payload)
    assert rec.urgency is Urgency.SCHEDULED
    assert rec.confidence is AgentConfidence.HIGH
    assert len(rec.reasoning_trace) == 2


def test_recommendation_requires_non_empty_trace():
    payload = {
        "recommended_action": "Monitor.",
        "urgency": "monitor",
        "reasoning_trace": [],
        "knowledge_sources": [],
        "confidence": "low",
    }
    with pytest.raises(ValidationError):
        Recommendation.model_validate(payload)


# --------------------------- Class catalog ---------------------------------


def test_fault_class_catalog_meets_assignment_minimums():
    # Assignment requires >= 7 PV faults and >= 5 BESS faults.
    assert len(PV_FAULT_CLASSES) >= 7
    assert len(BESS_FAULT_CLASSES) >= 5
    assert set(ALL_FAULT_CLASSES) == set(PV_FAULT_CLASSES) | set(BESS_FAULT_CLASSES)


# ----------------------------- ToolError -----------------------------------


def test_tool_error_rejects_unknown_code():
    with pytest.raises(ValidationError):
        ToolError.model_validate(
            {
                "error_code": "BANANA",
                "message": "x",
                "tool_name": "retrieve_knowledge",
                "trace_id": "abc",
            }
        )


def test_tool_error_accepts_documented_codes():
    for code in ("VALIDATION", "TIMEOUT", "INTERNAL", "NOT_FOUND"):
        err = ToolError.model_validate(
            {
                "error_code": code,
                "message": "x",
                "tool_name": "retrieve_knowledge",
                "trace_id": "abc",
            }
        )
        assert err.error_code == code


# ----------------------------- RawSample -----------------------------------


def _good_raw_sample_payload() -> dict:
    window = _good_window_payload()
    window["operating_condition"] = OperatingCondition.HIGH_IRRADIANCE.value
    return {"window": window, "label": "Partial_shading"}


def test_raw_sample_accepts_valid_pv_payload():
    sample = RawSample.model_validate(_good_raw_sample_payload())
    assert sample.label == "Partial_shading"
    assert sample.window.operating_condition is OperatingCondition.HIGH_IRRADIANCE


def test_raw_sample_rejects_unknown_label():
    payload = _good_raw_sample_payload()
    payload["label"] = "Aliens"
    with pytest.raises(ValidationError):
        RawSample.model_validate(payload)


def test_raw_sample_rejects_pv_label_on_bess_system():
    payload = _good_raw_sample_payload()
    payload["window"]["system_type"] = SystemType.BESS.value
    payload["label"] = "Partial_shading"  # PV-only fault
    with pytest.raises(ValidationError):
        RawSample.model_validate(payload)


def test_raw_sample_requires_operating_condition():
    payload = _good_raw_sample_payload()
    payload["window"].pop("operating_condition", None)
    with pytest.raises(ValidationError):
        RawSample.model_validate(payload)


# --------------------------- DatasetMetadata -------------------------------


def _good_dataset_meta() -> dict:
    return {
        "schema_version": "0.1.0",
        "generated_at": "2026-03-15T12:00:00+00:00",
        "seed": 42,
        "sample_rate_hz": 1.0,
        "window_size": 60,
        "pv_feature_names": ["V_dc", "I_dc"],
        "bess_feature_names": ["V_term", "I"],
        "n_samples": 4,
        "splits": {"train": 2, "val": 1, "test": 1},
        "class_distribution": {"PV_Normal": 2, "BESS_Normal": 2},
        "operating_condition_distribution": {
            "high_irradiance": 2,
            "low_irradiance": 1,
            "high_temperature": 1,
        },
        "notes": "smoke",
    }


def test_dataset_metadata_consistent_totals():
    meta = DatasetMetadata.model_validate(_good_dataset_meta())
    assert meta.splits[SplitName.TRAIN] == 2


def test_dataset_metadata_rejects_inconsistent_totals():
    payload = _good_dataset_meta()
    payload["splits"]["train"] = 99  # breaks the sum invariant
    with pytest.raises(ValidationError):
        DatasetMetadata.model_validate(payload)
