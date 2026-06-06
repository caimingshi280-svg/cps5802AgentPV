"""Unit tests for :mod:`quantization.onnx_export`."""
from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from models.cnn1d import CNN1D  # noqa: E402
from quantization.onnx_export import (  # noqa: E402
    StandardizingClassifier,
    export_checkpoint,
)
from training.data import FeatureStats  # noqa: E402


def _make_dummy_checkpoint(tmp_path, n_classes: int = 7):
    """Create a tiny .pt checkpoint matching the schema saved by training/train.py."""

    torch.manual_seed(0)
    model = CNN1D(in_channels=8, n_classes=n_classes, dropout=0.0)
    feature_stats = FeatureStats(
        mean=np.zeros(8, dtype=np.float32),
        std=np.ones(8, dtype=np.float32),
    )
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": 1,
        "val_macro_f1": 0.95,
        "val_accuracy": 0.95,
        "system_type": "PV" if n_classes == 7 else "BESS",
        "n_classes": n_classes,
        "label_classes": [
            "PV_Normal",
            "Partial_shading",
            "Soiling",
            "Bypass_diode_fault",
            "String_disconnection",
            "Inverter_fault",
            "Degradation",
        ][:n_classes]
        if n_classes == 7
        else [
            "BESS_Normal",
            "Capacity_fade",
            "Internal_resistance_increase",
            "Thermal_anomaly",
            "Cell_imbalance",
        ],
        "feature_stats": feature_stats.to_dict(),
        "model_arch": "CNN1D",
        "in_channels": 8,
        "dropout": 0.0,
    }
    ckpt_path = tmp_path / "fake.pt"
    torch.save(payload, ckpt_path)
    return ckpt_path, model, feature_stats


# ---------------------------------------------------------------------------
# StandardizingClassifier
# ---------------------------------------------------------------------------


def test_standardizing_classifier_applies_normalization() -> None:
    torch.manual_seed(0)
    base = CNN1D(in_channels=8, n_classes=7, dropout=0.0).eval()
    stats = FeatureStats(
        mean=np.full(8, 5.0, dtype=np.float32),
        std=np.full(8, 2.0, dtype=np.float32),
    )
    wrapped = StandardizingClassifier(base, stats).eval()

    raw = torch.full((1, 60, 8), 7.0)  # raw input
    standardized = (raw - 5.0) / 2.0   # = 1.0 everywhere

    with torch.no_grad():
        wrapped_out = wrapped(raw)
        manual_out = base(standardized)

    assert torch.allclose(wrapped_out, manual_out, atol=1e-6)


# ---------------------------------------------------------------------------
# export_checkpoint
# ---------------------------------------------------------------------------


def test_export_creates_valid_onnx(tmp_path) -> None:
    ckpt_path, _, _ = _make_dummy_checkpoint(tmp_path)
    out_path = tmp_path / "fake.onnx"
    export_checkpoint(checkpoint_path=ckpt_path, output_path=out_path)
    assert out_path.exists()
    onnx.checker.check_model(str(out_path))


def test_export_attaches_metadata(tmp_path) -> None:
    ckpt_path, _, _ = _make_dummy_checkpoint(tmp_path)
    out_path = tmp_path / "fake.onnx"
    export_checkpoint(checkpoint_path=ckpt_path, output_path=out_path)

    proto = onnx.load(str(out_path))
    meta = {p.key: p.value for p in proto.metadata_props}
    assert meta["agentpv.system_type"] == "PV"
    assert json.loads(meta["agentpv.label_classes"])[0] == "PV_Normal"
    assert int(meta["agentpv.in_channels"]) == 8


def test_export_numerical_parity_with_pytorch(tmp_path) -> None:
    """ONNX inference must match PyTorch (with stats applied) within FP32 tolerance."""

    ckpt_path, model, stats = _make_dummy_checkpoint(tmp_path)
    out_path = tmp_path / "fake.onnx"
    export_checkpoint(checkpoint_path=ckpt_path, output_path=out_path)

    rng = np.random.default_rng(seed=0)
    raw = rng.standard_normal((3, 60, 8)).astype(np.float32) * 4 + 2

    # Critical: BatchNorm uses batch statistics in train() mode and running
    # statistics in eval() mode — only eval() matches the ONNX graph.
    model.eval()
    with torch.no_grad():
        torch_logits = model(torch.from_numpy(stats.apply(raw))).numpy()

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    onnx_logits = sess.run(None, {"sensor_window": raw})[0]

    assert np.max(np.abs(torch_logits - onnx_logits)) < 1e-3


def test_export_rejects_checkpoint_without_feature_stats(tmp_path) -> None:
    """Old checkpoints without stats must fail loudly, not silently produce broken ONNX."""

    torch.manual_seed(0)
    model = CNN1D(in_channels=8, n_classes=7, dropout=0.0)
    payload = {
        "model_state_dict": model.state_dict(),
        "system_type": "PV",
        "n_classes": 7,
        "label_classes": ["x"] * 7,
        "model_arch": "CNN1D",
        "in_channels": 8,
        "dropout": 0.0,
        # NO feature_stats
    }
    ckpt_path = tmp_path / "old.pt"
    torch.save(payload, ckpt_path)

    with pytest.raises(KeyError, match="feature_stats"):
        export_checkpoint(checkpoint_path=ckpt_path, output_path=tmp_path / "old.onnx")


def test_export_supports_dynamic_batch(tmp_path) -> None:
    """The exported graph must accept varying batch sizes."""

    ckpt_path, _, _ = _make_dummy_checkpoint(tmp_path)
    out_path = tmp_path / "fake.onnx"
    export_checkpoint(checkpoint_path=ckpt_path, output_path=out_path)
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])

    for batch in (1, 5, 32):
        x = np.zeros((batch, 60, 8), dtype=np.float32)
        out = sess.run(None, {"sensor_window": x})[0]
        assert out.shape == (batch, 7)
