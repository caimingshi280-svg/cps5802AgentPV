"""ONNX Runtime classifier wrapper for AgentPV edge inference.

Responsibilities (rule §2 — single responsibility):

* Load a self-contained ONNX model exported by :mod:`quantization.onnx_export`
  (the graph already includes per-channel standardization).
* Validate that the request's ``system_type`` matches the model's metadata.
* Run inference with batch size 1 and produce a validated :class:`Alert`.
* Provide a ``benchmark`` helper that reports p50 / p95 / p99 / mean / max
  latency over N synthetic windows for the perf-budget gate (rule §17).

Anything *outside* this scope (HTTP routing, batching across requests,
asynchronous IO, model-version negotiation) belongs to ``api/edge_service.py``.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from api.schemas import (
    BESS_FAULT_CLASSES,
    PV_FAULT_CLASSES,
    Alert,
    SensorWindow,
    SystemType,
)
from inference.postprocess import logits_to_alert
from utils.logging_config import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class LatencyStats:
    """Single-sample latency benchmark (milliseconds)."""

    n: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "mean_ms": round(self.mean_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "max_ms": round(self.max_ms, 3),
        }


def _read_onnx_metadata(onnx_path: Path) -> dict[str, str]:
    """Read AgentPV-specific metadata embedded by ``quantization.onnx_export``."""

    proto = onnx.load(str(onnx_path))
    return {prop.key: prop.value for prop in proto.metadata_props}


class OnnxClassifier:
    """ONNX Runtime-backed classifier for one system type."""

    def __init__(self, onnx_path: Path) -> None:
        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model missing: {onnx_path}")

        # 显式 CPU provider — rule §17：edge 推理必须 CPU-only。
        self.session = ort.InferenceSession(
            str(self.onnx_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        meta = _read_onnx_metadata(self.onnx_path)
        if "agentpv.system_type" not in meta:
            raise ValueError(
                f"ONNX model {onnx_path} is missing 'agentpv.system_type' metadata; "
                "re-export with `quantization.onnx_export`."
            )
        self.system_type = SystemType(meta["agentpv.system_type"])
        self.label_classes: tuple[str, ...] = tuple(
            json.loads(meta["agentpv.label_classes"])
        )
        self.in_channels = int(meta.get("agentpv.in_channels", "8"))

        # 双重保险：ONNX metadata 的 labels 必须与 api.schemas 的 taxonomy 一致。
        # 任何分歧都说明有人改了 schema 但没重新导出 ONNX，必须立即报错（§3）。
        expected = (
            PV_FAULT_CLASSES
            if self.system_type is SystemType.PV
            else BESS_FAULT_CLASSES
        )
        if self.label_classes != expected:
            raise ValueError(
                f"Label taxonomy mismatch in {onnx_path}: "
                f"ONNX={self.label_classes} but api.schemas={expected}. "
                "Re-train and re-export the model."
            )

        log.info(
            "onnx_classifier_loaded",
            extra={
                "path": str(self.onnx_path),
                "system_type": self.system_type.value,
                "n_classes": len(self.label_classes),
                "input_name": self.input_name,
            },
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run_logits(self, x: np.ndarray) -> np.ndarray:
        """Run ONNX inference and return raw logits.

        Parameters
        ----------
        x
            Input tensor shaped ``(B, T, F)`` of raw sensor values. The
            ONNX graph applies per-channel standardisation internally
            (rule §17 self-contained model), so callers feed unprocessed
            window values directly.

        Returns
        -------
        np.ndarray
            ``(B, n_classes)`` float32 logits.
        """

        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        if x.ndim != 3:
            raise ValueError(f"expected (B, T, F), got {x.shape}")
        if x.shape[-1] != self.in_channels:
            raise ValueError(
                f"expected F={self.in_channels} feature channels, got {x.shape[-1]}"
            )
        return self.session.run([self.output_name], {self.input_name: x})[0]

    # Back-compat: keep the old private name for callers inside this
    # module (e.g. ``benchmark`` below) but route them through the public
    # method to avoid drift.
    _run_logits = run_logits

    def predict_window(self, window: SensorWindow) -> Alert:
        """Run inference on a single :class:`SensorWindow` and return an :class:`Alert`."""

        if window.system_type is not self.system_type:
            raise ValueError(
                f"Window system_type={window.system_type.value} does not match "
                f"this classifier's system_type={self.system_type.value}"
            )

        x = np.asarray(window.values, dtype=np.float32)
        if x.shape != (window.window_size, len(window.feature_names)):
            raise ValueError(
                f"window.values shape {x.shape} does not match declared "
                f"(window_size={window.window_size}, "
                f"n_features={len(window.feature_names)})"
            )
        logits = self._run_logits(x[np.newaxis, ...])[0]

        # Snapshot 取窗口最后一拍——给 cloud agent 一个最新视图。
        last_row = window.values[-1]
        snapshot = {name: float(last_row[i]) for i, name in enumerate(window.feature_names)}

        return logits_to_alert(
            logits=logits,
            system_id=window.system_id,
            system_type=window.system_type,
            sensor_snapshot=snapshot,
            timestamp=window.timestamp_start,
        )

    # ------------------------------------------------------------------
    # Benchmarking — perf budget verification (rule §17)
    # ------------------------------------------------------------------

    def benchmark(self, n: int = 200, window_size: int = 60) -> LatencyStats:
        """Run ``n`` synthetic single-sample inferences and report latency stats."""

        rng = np.random.default_rng(seed=0)
        latencies_ms: list[float] = []

        # 5 次 warmup —— 排除 first-run 编译开销。
        warmup = rng.standard_normal((1, window_size, self.in_channels)).astype(np.float32)
        for _ in range(5):
            self._run_logits(warmup)

        for _ in range(n):
            x = rng.standard_normal((1, window_size, self.in_channels)).astype(np.float32)
            t0 = time.perf_counter()
            self._run_logits(x)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        arr = np.asarray(latencies_ms)
        return LatencyStats(
            n=n,
            mean_ms=float(arr.mean()),
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            max_ms=float(arr.max()),
        )
