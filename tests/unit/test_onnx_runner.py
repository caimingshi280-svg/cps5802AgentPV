"""Unit tests for :mod:`inference.onnx_runner`.

These tests build their own tiny ONNX model in tmp_path so they do **not**
depend on a trained checkpoint being present.
"""
from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

from api.schemas import (  # noqa: E402
    BESS_FAULT_CLASSES,
    PV_FAULT_CLASSES,
    Alert,
    SensorWindow,
    Severity,
    SystemType,
)
from inference.onnx_runner import LatencyStats, OnnxClassifier  # noqa: E402
from models.cnn1d import CNN1D  # noqa: E402
from quantization.onnx_export import export_checkpoint  # noqa: E402
from training.data import FeatureStats  # noqa: E402

PV_FEATURE_NAMES = (
    "V_dc",
    "I_dc",
    "P",
    "T_module",
    "T_amb",
    "G",
    "P_ac",
    "eta",
)


def _save_checkpoint_and_export(tmp_path, system_type: SystemType):
    """Helper: create a tiny .pt → export to .onnx in tmp_path."""

    classes = (
        list(PV_FAULT_CLASSES) if system_type is SystemType.PV else list(BESS_FAULT_CLASSES)
    )
    n_classes = len(classes)

    torch.manual_seed(0)
    model = CNN1D(in_channels=8, n_classes=n_classes, dropout=0.0)
    stats = FeatureStats(
        mean=np.zeros(8, dtype=np.float32),
        std=np.ones(8, dtype=np.float32),
    )
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": 1,
        "val_macro_f1": 0.95,
        "val_accuracy": 0.95,
        "system_type": system_type.value,
        "n_classes": n_classes,
        "label_classes": classes,
        "feature_stats": stats.to_dict(),
        "model_arch": "CNN1D",
        "in_channels": 8,
        "dropout": 0.0,
    }
    ckpt_path = tmp_path / f"{system_type.value.lower()}.pt"
    torch.save(payload, ckpt_path)
    out_path = tmp_path / f"{system_type.value.lower()}.onnx"
    export_checkpoint(checkpoint_path=ckpt_path, output_path=out_path)
    return out_path


def _make_window(system_type: SystemType, system_id: str = "test_001") -> SensorWindow:
    rng = np.random.default_rng(seed=0)
    values = rng.standard_normal((60, 8)).tolist()
    return SensorWindow(
        timestamp_start=datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC),
        system_id=system_id,
        system_type=system_type,
        sample_rate_hz=1.0,
        window_size=60,
        feature_names=list(PV_FEATURE_NAMES),
        values=values,
        operating_condition="high_irradiance",
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_classifier_loads_metadata_correctly(tmp_path) -> None:
    onnx_path = _save_checkpoint_and_export(tmp_path, SystemType.PV)
    clf = OnnxClassifier(onnx_path)
    assert clf.system_type is SystemType.PV
    assert clf.label_classes == PV_FAULT_CLASSES
    assert clf.in_channels == 8


def test_classifier_loads_bess_metadata(tmp_path) -> None:
    onnx_path = _save_checkpoint_and_export(tmp_path, SystemType.BESS)
    clf = OnnxClassifier(onnx_path)
    assert clf.system_type is SystemType.BESS
    assert clf.label_classes == BESS_FAULT_CLASSES


def test_classifier_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="ONNX model missing"):
        OnnxClassifier(tmp_path / "nonexistent.onnx")


# ---------------------------------------------------------------------------
# predict_window
# ---------------------------------------------------------------------------


def test_predict_window_returns_validated_alert(tmp_path) -> None:
    onnx_path = _save_checkpoint_and_export(tmp_path, SystemType.PV)
    clf = OnnxClassifier(onnx_path)
    alert = clf.predict_window(_make_window(SystemType.PV))
    assert isinstance(alert, Alert)
    assert alert.system_type is SystemType.PV
    assert alert.fault_class in PV_FAULT_CLASSES
    assert isinstance(alert.severity, Severity)
    assert 0.0 <= alert.confidence <= 1.0


def test_predict_window_snapshot_is_last_row(tmp_path) -> None:
    """The Alert.sensor_snapshot must reflect the most recent sensor reading."""

    onnx_path = _save_checkpoint_and_export(tmp_path, SystemType.PV)
    clf = OnnxClassifier(onnx_path)
    window = _make_window(SystemType.PV)
    alert = clf.predict_window(window)
    last_row = window.values[-1]
    for i, name in enumerate(window.feature_names):
        assert alert.sensor_snapshot[name] == pytest.approx(last_row[i], rel=1e-6)


def test_predict_window_rejects_system_mismatch(tmp_path) -> None:
    """Sending a BESS window to a PV-loaded classifier must fail loudly."""

    onnx_path = _save_checkpoint_and_export(tmp_path, SystemType.PV)
    clf = OnnxClassifier(onnx_path)
    bess_window = _make_window(SystemType.BESS, system_id="BESS_001")
    with pytest.raises(ValueError, match="does not match"):
        clf.predict_window(bess_window)


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


def test_benchmark_reports_finite_stats(tmp_path) -> None:
    onnx_path = _save_checkpoint_and_export(tmp_path, SystemType.PV)
    clf = OnnxClassifier(onnx_path)
    stats = clf.benchmark(n=20)
    assert isinstance(stats, LatencyStats)
    assert stats.n == 20
    assert stats.mean_ms > 0
    assert stats.p99_ms >= stats.p95_ms >= stats.p50_ms
    # Edge inference budget — even on the slowest CPU this should be << 100 ms.
    assert stats.p99_ms < 100.0
