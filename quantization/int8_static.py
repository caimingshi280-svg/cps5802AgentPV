"""INT8 static post-training quantization for AgentPV ONNX classifiers.

Implements assignment §4.2 ("Apply at least one compression technique:
structured pruning, INT8 quantization, or knowledge distillation") via
ONNX Runtime's static PTQ pipeline (rule §17 — CPU-only edge target).

Pipeline
--------
1. **Pre-process**: ``onnxruntime.quantization.shape_inference.quant_pre_process``
   does shape inference, fuses constants, and prepares the FP32 graph
   for quantization (otherwise certain ops fail to quantize cleanly).
2. **Calibrate**: a small batch of *representative* inputs from the
   training split is run through the FP32 graph to gather activation
   distributions per node. We sample ``calibration_size`` rows per
   class to ensure no class is unrepresented (rule §6 reproducibility).
3. **Quantize**: ``quantize_static`` writes an INT8 graph that uses
   8-bit weights + activations with per-tensor (default) or per-channel
   scales. We pick QDQ format (insert Quantize / DeQuantize nodes
   around FP32 ops) which is the most portable and widely-supported
   ONNX quantization format.
4. **Re-attach metadata**: ORT's quantizer drops our ``agentpv.*``
   metadata props; we re-load + re-add them so downstream
   :class:`inference.onnx_runner.OnnxClassifier` keeps working
   without code changes.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from api.schemas import SystemType
from training.data import _load_split_arrays
from utils.logging_config import get_logger
from utils.paths import ARTIFACTS_DIR, PROCESSED_DIR, SPLITS_DIR, ensure_dir

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Calibration data reader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CalibrationConfig:
    """Knobs that control how calibration samples are drawn."""

    samples_per_class: int = 30
    seed: int = 42


def _select_calibration_indices(
    y: np.ndarray, samples_per_class: int, seed: int
) -> np.ndarray:
    """Pick ``samples_per_class`` row indices from each class deterministically.

    Returning a flat sorted index array keeps the calibration order
    stable across runs (rule §6 reproducibility).
    """

    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        if idx.size == 0:
            continue
        if idx.size <= samples_per_class:
            picked = idx
        else:
            picked = rng.choice(idx, size=samples_per_class, replace=False)
        selected.extend(int(i) for i in picked)
    selected.sort()
    return np.asarray(selected, dtype=np.int64)


class NumpyCalibrationDataReader(CalibrationDataReader):
    """Yield ``{input_name: array}`` batches drawn from a NumPy split.

    ORT's calibrator iterates this once. We pre-materialise the indices
    so the iterator is deterministic and resumable in tests.
    """

    def __init__(self, x: np.ndarray, input_name: str, batch_size: int = 1) -> None:
        if x.ndim != 3:
            raise ValueError(f"calibration X must be 3-D, got {x.shape}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be ≥ 1, got {batch_size}")
        self._x = x.astype(np.float32, copy=False)
        self._input_name = input_name
        self._batch_size = batch_size
        self._iter: Iterator[dict[str, np.ndarray]] | None = None

    def _make_iter(self) -> Iterator[dict[str, np.ndarray]]:
        n = self._x.shape[0]
        for start in range(0, n, self._batch_size):
            batch = self._x[start : start + self._batch_size]
            yield {self._input_name: batch}

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._iter is None:
            self._iter = self._make_iter()
        return next(self._iter, None)

    def rewind(self) -> None:
        """Reset so the calibrator can iterate again if it wants to."""
        self._iter = None


# ---------------------------------------------------------------------------
# Metadata preservation
# ---------------------------------------------------------------------------


def _copy_agentpv_metadata(src_path: Path, dst_path: Path) -> dict[str, str]:
    """Copy every ``agentpv.*`` metadata prop from ``src_path`` to ``dst_path``.

    The ONNX Runtime quantizer rewrites the model and drops our
    metadata; we re-attach it so :class:`OnnxClassifier` (which reads
    ``agentpv.system_type`` / ``agentpv.label_classes``) still works
    against the INT8 model with zero code changes.
    """

    src = onnx.load(str(src_path))
    dst = onnx.load(str(dst_path))

    src_meta = {p.key: p.value for p in src.metadata_props}
    if not src_meta:
        log.warning(
            "int8_no_source_metadata",
            extra={"src": str(src_path), "dst": str(dst_path)},
        )
        return {}

    # Mark this artefact so consumers can tell it apart from FP32.
    src_meta["agentpv.precision"] = "int8"

    # Drop existing entries with the same key, then add ours.
    keys_to_set = set(src_meta.keys())
    keep = [p for p in dst.metadata_props if p.key not in keys_to_set]
    del dst.metadata_props[:]
    for prop in keep:
        dst.metadata_props.append(prop)
    for k, v in src_meta.items():
        prop = dst.metadata_props.add()
        prop.key = k
        prop.value = v

    onnx.save(dst, str(dst_path))
    log.info(
        "int8_metadata_copied",
        extra={"src": str(src_path), "dst": str(dst_path), "n_keys": len(src_meta)},
    )
    return src_meta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Int8QuantizationResult:
    """Summary of one INT8 quantization run."""

    fp32_path: Path
    int8_path: Path
    calibration_samples: int
    fp32_bytes: int
    int8_bytes: int

    @property
    def compression_ratio(self) -> float:
        """``fp32 / int8`` size ratio (higher is better)."""
        return float(self.fp32_bytes) / float(self.int8_bytes) if self.int8_bytes > 0 else 0.0


def quantize_to_int8_static(
    *,
    fp32_onnx_path: Path,
    int8_onnx_path: Path,
    system_type: SystemType,
    processed_dir: Path = PROCESSED_DIR,
    splits_dir: Path = SPLITS_DIR,
    samples_per_class: int = 30,
    calibration_seed: int = 42,
    quant_format: QuantFormat = QuantFormat.QDQ,
    activation_type: QuantType = QuantType.QInt8,
    weight_type: QuantType = QuantType.QInt8,
    calibration_method: CalibrationMethod = CalibrationMethod.MinMax,
) -> Int8QuantizationResult:
    """Quantize an FP32 ONNX classifier to INT8 using static PTQ.

    Parameters
    ----------
    fp32_onnx_path
        FP32 ONNX file produced by :mod:`quantization.onnx_export`.
    int8_onnx_path
        Where to write the INT8 ONNX file. Parent directory is created.
    system_type
        Drives which split arrays / calibration data are loaded.
    samples_per_class
        Number of training-split samples to draw per class for the
        calibration pass. Total calibration size is
        ``samples_per_class × n_classes``.
    quant_format
        QDQ inserts Quantize / DeQuantize nodes (more portable);
        QOperator uses fused INT8 ops (faster on supporting backends).
    activation_type / weight_type
        ``QInt8`` (signed) is the default and works for the asymmetric
        sensor distributions we have. ``QUInt8`` would be slightly
        faster on ARM but loses range for negative-going signals.
    calibration_method
        ``MinMax`` is the simplest and reproducible-by-default choice
        for our balanced training data; ``Entropy`` (KL) is the polish
        upgrade if INT8 ever loses meaningful accuracy.
    """

    if not fp32_onnx_path.exists():
        raise FileNotFoundError(f"FP32 ONNX not found: {fp32_onnx_path}")
    ensure_dir(int8_onnx_path.parent)

    # 1) Load the training split for calibration.
    from api.schemas import SplitName
    x_train, y_train = _load_split_arrays(
        processed_dir, splits_dir, system_type, SplitName.TRAIN
    )
    indices = _select_calibration_indices(y_train, samples_per_class, calibration_seed)
    x_calib = x_train[indices]
    log.info(
        "int8_calibration_data_selected",
        extra={
            "system_type": system_type.value,
            "n_train_total": int(x_train.shape[0]),
            "n_calibration": int(x_calib.shape[0]),
            "samples_per_class": samples_per_class,
        },
    )

    # 2) Pre-process: shape inference + constant folding.
    pre_path = int8_onnx_path.with_suffix(".pre.onnx")
    quant_pre_process(
        input_model_path=str(fp32_onnx_path),
        output_model_path=str(pre_path),
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=False,
    )

    # 3) Calibrate + quantize. We have to find the input name of the
    # *pre-processed* model — pre_process can rename inputs in some
    # edge cases, though for our straight-through Conv1d graph it
    # doesn't. Read the pre-processed model to be safe.
    pre_proto = onnx.load(str(pre_path))
    if not pre_proto.graph.input:
        raise RuntimeError(f"pre-processed ONNX has no inputs: {pre_path}")
    input_name = pre_proto.graph.input[0].name

    reader = NumpyCalibrationDataReader(x_calib, input_name=input_name, batch_size=1)
    quantize_static(
        model_input=str(pre_path),
        model_output=str(int8_onnx_path),
        calibration_data_reader=reader,
        quant_format=quant_format,
        activation_type=activation_type,
        weight_type=weight_type,
        calibrate_method=calibration_method,
        per_channel=False,  # CPUExecutionProvider supports both; per-tensor is simpler & deterministic
        reduce_range=False,
    )

    # 4) Re-attach AgentPV metadata (precision flag included).
    _copy_agentpv_metadata(fp32_onnx_path, int8_onnx_path)

    # 5) Cleanup the intermediate file.
    try:
        pre_path.unlink()
    except OSError:
        log.warning("int8_pre_path_cleanup_failed", extra={"path": str(pre_path)})

    fp32_bytes = fp32_onnx_path.stat().st_size
    int8_bytes = int8_onnx_path.stat().st_size
    log.info(
        "int8_done",
        extra={
            "fp32_bytes": fp32_bytes,
            "int8_bytes": int8_bytes,
            "ratio": round(fp32_bytes / max(int8_bytes, 1), 3),
        },
    )

    onnx.checker.check_model(str(int8_onnx_path))
    return Int8QuantizationResult(
        fp32_path=fp32_onnx_path,
        int8_path=int8_onnx_path,
        calibration_samples=int(x_calib.shape[0]),
        fp32_bytes=fp32_bytes,
        int8_bytes=int8_bytes,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="agentpv-int8-static")
    parser.add_argument(
        "--system",
        choices=["pv", "bess"],
        required=True,
        help="Which system's classifier to quantize.",
    )
    parser.add_argument(
        "--fp32-input",
        type=Path,
        default=None,
        help="FP32 ONNX path. Default: ARTIFACTS_DIR/cnn1d_<system>.onnx",
    )
    parser.add_argument(
        "--int8-output",
        type=Path,
        default=None,
        help="INT8 ONNX path. Default: ARTIFACTS_DIR/cnn1d_<system>_int8.onnx",
    )
    parser.add_argument("--samples-per-class", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    system_type = SystemType.PV if args.system == "pv" else SystemType.BESS
    fp32_path = args.fp32_input or ARTIFACTS_DIR / f"cnn1d_{args.system}.onnx"
    int8_path = args.int8_output or ARTIFACTS_DIR / f"cnn1d_{args.system}_int8.onnx"

    result = quantize_to_int8_static(
        fp32_onnx_path=fp32_path,
        int8_onnx_path=int8_path,
        system_type=system_type,
        samples_per_class=args.samples_per_class,
        calibration_seed=args.seed,
    )
    print(
        json.dumps(
            {
                "fp32_path": str(result.fp32_path),
                "int8_path": str(result.int8_path),
                "calibration_samples": result.calibration_samples,
                "fp32_bytes": result.fp32_bytes,
                "int8_bytes": result.int8_bytes,
                "compression_ratio": round(result.compression_ratio, 3),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
