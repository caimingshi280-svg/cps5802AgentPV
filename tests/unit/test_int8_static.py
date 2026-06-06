"""Unit tests for :mod:`quantization.int8_static`.

Strategy
--------
* Pure helpers (calibration index selection, NumPy reader, metadata copy)
  are tested in isolation.
* The end-to-end :func:`quantize_to_int8_static` pipeline is exercised
  on a *tiny* synthetic CNN (1 Conv1d + 1 Linear) we build on the fly,
  so the test stays under ~3 s and doesn't depend on the real PV / BESS
  artefacts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from torch import nn

from api.schemas import SystemType
from quantization.int8_static import (
    NumpyCalibrationDataReader,
    _copy_agentpv_metadata,
    _select_calibration_indices,
    quantize_to_int8_static,
)
from utils.paths import ensure_dir

# ---------------------------------------------------------------------------
# Calibration index selection
# ---------------------------------------------------------------------------


def test_select_indices_balances_per_class() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, size=400)
    indices = _select_calibration_indices(y, samples_per_class=20, seed=42)
    selected = y[indices]
    counts = {int(c): int((selected == c).sum()) for c in np.unique(y)}
    assert all(v == 20 for v in counts.values())
    assert (np.diff(indices) > 0).all(), "indices must be sorted"


def test_select_indices_handles_classes_smaller_than_quota() -> None:
    y = np.array([0, 0, 1, 2, 2, 2])
    indices = _select_calibration_indices(y, samples_per_class=5, seed=0)
    sel = y[indices]
    assert (sel == 0).sum() == 2  # only 2 available
    assert (sel == 1).sum() == 1
    assert (sel == 2).sum() == 3


def test_select_indices_is_deterministic() -> None:
    rng = np.random.default_rng(7)
    y = rng.integers(0, 5, size=200)
    a = _select_calibration_indices(y, samples_per_class=10, seed=42)
    b = _select_calibration_indices(y, samples_per_class=10, seed=42)
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# NumpyCalibrationDataReader
# ---------------------------------------------------------------------------


def test_numpy_reader_iterates_and_terminates() -> None:
    x = np.arange(60, dtype=np.float32).reshape(5, 4, 3)  # (B,T,F)
    reader = NumpyCalibrationDataReader(x, input_name="sensor_window", batch_size=2)
    batches: list[np.ndarray] = []
    while True:
        nxt = reader.get_next()
        if nxt is None:
            break
        assert "sensor_window" in nxt
        batches.append(nxt["sensor_window"])
    sizes = [b.shape[0] for b in batches]
    assert sizes == [2, 2, 1]
    np.testing.assert_array_equal(np.concatenate(batches, axis=0), x)


def test_numpy_reader_rejects_non_3d_input() -> None:
    with pytest.raises(ValueError, match="3-D"):
        NumpyCalibrationDataReader(
            np.zeros((4, 4)), input_name="x", batch_size=1
        )


def test_numpy_reader_rewind_restarts_iteration() -> None:
    x = np.zeros((2, 3, 4), dtype=np.float32)
    reader = NumpyCalibrationDataReader(x, input_name="x", batch_size=1)
    assert reader.get_next() is not None
    assert reader.get_next() is not None
    assert reader.get_next() is None
    reader.rewind()
    assert reader.get_next() is not None


# ---------------------------------------------------------------------------
# Metadata copy
# ---------------------------------------------------------------------------


def _build_minimal_onnx(tmp_path: Path, *, attach_meta: bool) -> Path:
    """Export a 1-conv toy model and optionally attach AgentPV metadata."""

    ensure_dir(tmp_path)

    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
            self.fc = nn.Linear(4 * 6, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.transpose(1, 2)
            x = self.conv(x)
            x = torch.relu(x)
            x = x.flatten(1)
            return self.fc(x)

    model = Toy().eval()
    dummy = torch.randn(1, 6, 2)
    onnx_path = tmp_path / "toy.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["sensor_window"],
        output_names=["logits"],
        dynamic_axes={"sensor_window": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    if attach_meta:
        proto = onnx.load(str(onnx_path))
        for k, v in [
            ("agentpv.system_type", "PV"),
            ("agentpv.label_classes", '["a","b","c"]'),
            ("agentpv.in_channels", "2"),
        ]:
            prop = proto.metadata_props.add()
            prop.key, prop.value = k, v
        onnx.save(proto, str(onnx_path))
    return onnx_path


def test_copy_metadata_carries_agentpv_keys_and_marks_int8(tmp_path: Path) -> None:
    src = _build_minimal_onnx(tmp_path, attach_meta=True)
    dst = _build_minimal_onnx(tmp_path / "dst", attach_meta=False)

    copied = _copy_agentpv_metadata(src, dst)

    proto = onnx.load(str(dst))
    meta = {p.key: p.value for p in proto.metadata_props}
    assert meta["agentpv.system_type"] == "PV"
    assert meta["agentpv.label_classes"] == '["a","b","c"]'
    assert meta["agentpv.precision"] == "int8"
    assert "agentpv.precision" in copied


def test_copy_metadata_handles_missing_source_gracefully(tmp_path: Path) -> None:
    src = _build_minimal_onnx(tmp_path, attach_meta=False)
    dst = _build_minimal_onnx(tmp_path / "dst", attach_meta=False)
    out = _copy_agentpv_metadata(src, dst)
    assert out == {}  # no source meta means nothing copied


# ---------------------------------------------------------------------------
# End-to-end (toy model + synthetic calibration data)
# ---------------------------------------------------------------------------


def _write_synthetic_split(
    tmp_path: Path, *, n_per_class: int = 8, n_classes: int = 3
) -> tuple[Path, Path]:
    """Stand in for the real processed/ + splits/ directory layout.

    Matches the schema in :func:`training.data._load_split_arrays`:
    flat ``X_pv.npz`` / ``y_pv.npz`` / ``meta_pv.csv`` under
    ``processed_dir``, and ``train.csv`` under ``splits_dir``.
    """

    processed = tmp_path / "processed"
    splits = tmp_path / "splits"
    ensure_dir(processed)
    ensure_dir(splits)

    rng = np.random.default_rng(0)
    n = n_per_class * n_classes
    x = rng.standard_normal((n, 6, 2)).astype(np.float32)
    # PV synthetic class names — they don't have to match real PV_FAULT_CLASSES,
    # only be unique strings since calibration only uses np.unique grouping.
    class_names = [f"synthetic_class_{i}" for i in range(n_classes)]
    y = np.array(
        [class_names[i // n_per_class] for i in range(n)], dtype="<U32"
    )

    np.savez(processed / "X_pv.npz", X=x)
    np.savez(processed / "y_pv.npz", y=y)

    import pandas as pd

    sample_idx = np.arange(n)
    system_id = np.array([f"PV_{i:03d}" for i in sample_idx])
    meta_df = pd.DataFrame(
        {
            "local_idx": sample_idx,
            "sample_idx": sample_idx,
            "system_id": system_id,
            "system_type": "PV",
            "label": y,
            "operating_condition": "default",
        }
    )
    meta_df.to_csv(processed / "meta_pv.csv", index=False)

    split_df = meta_df[
        ["sample_idx", "system_id", "system_type", "label", "operating_condition"]
    ].copy()
    split_df.to_csv(splits / "train.csv", index=False)
    return processed, splits


def test_quantize_to_int8_static_end_to_end(tmp_path: Path) -> None:
    fp32 = _build_minimal_onnx(tmp_path, attach_meta=True)
    int8 = tmp_path / "toy_int8.onnx"
    processed_root, splits_root = _write_synthetic_split(tmp_path)

    result = quantize_to_int8_static(
        fp32_onnx_path=fp32,
        int8_onnx_path=int8,
        system_type=SystemType.PV,
        processed_dir=processed_root,
        splits_dir=splits_root,
        samples_per_class=4,
        calibration_seed=42,
    )

    assert int8.exists()
    assert result.calibration_samples == 4 * 3
    assert result.fp32_bytes > 0
    assert result.int8_bytes > 0
    # The toy model is so small that quantization metadata overhead can
    # exceed the weight savings — only assert files differ (otherwise
    # we know quantization didn't actually run).
    assert result.int8_bytes != result.fp32_bytes
    proto = onnx.load(str(int8))
    meta = {p.key: p.value for p in proto.metadata_props}
    assert meta.get("agentpv.precision") == "int8"


def test_quantize_to_int8_static_rejects_missing_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        quantize_to_int8_static(
            fp32_onnx_path=tmp_path / "nope.onnx",
            int8_onnx_path=tmp_path / "out.onnx",
            system_type=SystemType.PV,
        )
