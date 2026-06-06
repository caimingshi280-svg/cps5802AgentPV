"""CPU latency benchmark for ONNX classifiers (assignment §4.3 + §8.3).

Spec requirements:

* Inference latency mean **and** 95th percentile over 1000 runs on CPU
* CPU-only, ``onnxruntime.InferenceSession`` (rule §17, §22)

Implementation notes:

* We exclude the first ``warmup_runs`` calls from statistics so JIT
  warm-up / OS page faults / thread-pool spin-up do not dominate.
* Each run uses the **same** synthetic input — we are measuring *graph*
  latency, not data-loading or simulator overhead.
* Returned timings are in **milliseconds**. The JSON payload includes
  configuration so the report is reproducible.

The benchmark is decoupled from any specific runner via a callable
``predict_fn(np.ndarray) -> np.ndarray`` interface; that lets us reuse
the same kernel for ONNX, ONNX-INT8, or any future PyTorch baseline.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from utils.logging_config import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class LatencyResult:
    """Latency statistics over a benchmark run.

    All durations are in **milliseconds**.
    """

    n_runs: int  # measured runs (excluding warm-up)
    n_warmup: int
    batch_size: int
    window_size: int
    in_channels: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Plain-dict for JSON dump; floats rounded to 4 decimals."""

        out = {
            "n_runs": self.n_runs,
            "n_warmup": self.n_warmup,
            "batch_size": self.batch_size,
            "window_size": self.window_size,
            "in_channels": self.in_channels,
            "mean_ms": round(self.mean_ms, 4),
            "p50_ms": round(self.p50_ms, 4),
            "p95_ms": round(self.p95_ms, 4),
            "p99_ms": round(self.p99_ms, 4),
            "min_ms": round(self.min_ms, 4),
            "max_ms": round(self.max_ms, 4),
            "std_ms": round(self.std_ms, 4),
        }
        if self.extra:
            out["extra"] = self.extra
        return out


def _percentile_ms(timings: np.ndarray, q: float) -> float:
    """Return ``q``-th percentile in ms; ``q`` in [0, 100]."""

    if timings.size == 0:
        return 0.0
    return float(np.percentile(timings, q))


def benchmark_latency(
    predict_fn: Callable[[np.ndarray], Any],
    *,
    window_size: int,
    in_channels: int,
    n_runs: int = 1000,
    n_warmup: int = 50,
    batch_size: int = 1,
    seed: int = 42,
    extra: dict[str, Any] | None = None,
) -> LatencyResult:
    """Run ``predict_fn`` ``n_runs + n_warmup`` times and report stats.

    Parameters
    ----------
    predict_fn
        Callable taking an ``np.ndarray`` of shape ``(batch_size,
        window_size, in_channels)`` and returning anything (we only time
        the call). The callable is responsible for any internal type
        conversion.
    window_size, in_channels
        Used to construct the synthetic input. The values are drawn from
        a fixed-seed RNG so the timings are deterministic across runs of
        the same Python process.
    n_runs
        Number of timed iterations. Assignment requires ≥ 1000.
    n_warmup
        Number of un-timed iterations to dampen JIT / page-fault effects.
        Choose ≥ 10; 50 is a generous default.
    batch_size
        Inference batch size. The assignment focuses on single-sample
        latency on the edge, but the API supports larger batches for
        Component 6 throughput experiments.
    seed
        RNG seed for the synthetic input.
    extra
        Free-form additional metadata to embed in the result (e.g. ONNX
        path, system_type, providers).
    """

    if n_runs < 1:
        raise ValueError(f"n_runs must be ≥ 1, got {n_runs}")
    if n_warmup < 0:
        raise ValueError(f"n_warmup must be ≥ 0, got {n_warmup}")
    if window_size <= 0 or in_channels <= 0 or batch_size <= 0:
        raise ValueError(
            f"window_size / in_channels / batch_size must all be positive; "
            f"got {window_size}, {in_channels}, {batch_size}"
        )

    rng = np.random.default_rng(seed)
    sample = rng.standard_normal((batch_size, window_size, in_channels)).astype(np.float32)

    for _ in range(n_warmup):
        predict_fn(sample)

    timings = np.empty(n_runs, dtype=np.float64)
    for i in range(n_runs):
        t0 = time.perf_counter()
        predict_fn(sample)
        timings[i] = (time.perf_counter() - t0) * 1000.0

    mean_ms = float(timings.mean())
    p50 = _percentile_ms(timings, 50.0)
    p95 = _percentile_ms(timings, 95.0)
    p99 = _percentile_ms(timings, 99.0)
    log.info(
        "latency_benchmark_done",
        extra={
            "n_runs": n_runs,
            "n_warmup": n_warmup,
            "mean_ms": round(mean_ms, 4),
            "p95_ms": round(p95, 4),
        },
    )
    return LatencyResult(
        n_runs=n_runs,
        n_warmup=n_warmup,
        batch_size=batch_size,
        window_size=window_size,
        in_channels=in_channels,
        mean_ms=mean_ms,
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        min_ms=float(timings.min()),
        max_ms=float(timings.max()),
        std_ms=float(timings.std(ddof=0)),
        extra=dict(extra or {}),
    )
