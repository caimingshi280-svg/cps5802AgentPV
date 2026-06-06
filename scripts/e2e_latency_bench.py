"""Measure edge + agent HTTP latency for Component 6 integration evaluation.

Three integration modes (assignment §4.6 ablation):

* ``--mode full``        — edge ``/predict`` → agent ``/recommend`` (default)
* ``--mode edge_only``   — edge ``/predict`` only (graceful degradation)
* ``--mode cloud_only``  — agent ``/recommend`` only, building the Alert in
                          Python from the simulator-style payload (raw sensor
                          to agent path, no edge classifier)

Outputs a JSON file with per-mode P50 / P95 / mean / max latencies plus
the raw per-iteration timing array so downstream renderers can produce
violin / CDF plots.

Examples::

    # Full pipeline, 50× (default)
    python scripts/e2e_latency_bench.py --iterations 50 \\
        --out-json reports/integration/e2e_latency_full.json

    # Edge-only ablation
    python scripts/e2e_latency_bench.py --mode edge_only --iterations 50

    # Cloud-only ablation
    python scripts/e2e_latency_bench.py --mode cloud_only --iterations 50
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import httpx  # noqa: E402

from simulation.pv_simulator import PV_FEATURE_NAMES  # noqa: E402

Mode = Literal["full", "edge_only", "cloud_only"]


# ---------------------------------------------------------------------------
# Synthetic payload builders
# ---------------------------------------------------------------------------


def _pv_window_payload(*, window_size: int = 60) -> dict:
    """Build a deterministic ``SensorWindow`` payload for edge ``/predict``."""

    row = [28.0, 8.0, 240.0, 52.0, 25.0, 900.0, 190.0, 0.15]
    values = [list(row) for _ in range(window_size)]
    return {
        "timestamp_start": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "system_id": "BENCH-PV-001",
        "system_type": "PV",
        "sample_rate_hz": 1.0,
        "window_size": window_size,
        "feature_names": list(PV_FEATURE_NAMES),
        "values": values,
        "operating_condition": "high_irradiance",
    }


def _synthesised_pv_alert() -> dict:
    """Build an Alert directly from sensor snapshot (cloud_only path).

    The cloud-only ablation skips the edge classifier entirely: the agent
    receives a coarse rule-based Alert (severity ``warning`` is a stand-in
    for the real edge severity head). This is the "raw sensor data sent
    directly to agent" path defined by assignment §4.6.
    """

    return {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "system_id": "BENCH-PV-001",
        "system_type": "PV",
        "fault_class": "Inverter_fault",
        "severity": "warning",
        "confidence": 0.55,
        "sensor_snapshot": {
            "V_dc": 28.0,
            "I_dc": 8.0,
            "P": 240.0,
            "T_module": 52.0,
            "T_amb": 25.0,
            "G": 900.0,
            "P_ac": 190.0,
            "eta": 0.15,
        },
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return float(sorted_vals[idx])


def _summarise(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "max": 0.0, "min": 0.0, "n": 0}
    s = sorted(values)
    return {
        "p50": _percentile(s, 0.50),
        "p95": _percentile(s, 0.95),
        "p99": _percentile(s, 0.99),
        "mean": float(statistics.fmean(values)),
        "max": float(s[-1]),
        "min": float(s[0]),
        "n": len(values),
    }


# ---------------------------------------------------------------------------
# Bench loop
# ---------------------------------------------------------------------------


def _run_one_iter(
    *,
    client: httpx.Client,
    mode: Mode,
    edge_base: str,
    agent_base: str,
    window: dict,
    alert_template: dict,
) -> tuple[float, float | None, float | None]:
    """Run a single bench iteration. Returns ``(total_ms, edge_ms, agent_ms)``."""

    edge_ms: float | None = None
    agent_ms: float | None = None
    t_start = time.perf_counter()

    if mode == "cloud_only":
        # Skip edge entirely; mutate the alert template's timestamp so the
        # agent doesn't see clock-frozen requests.
        alert = dict(alert_template)
        alert["timestamp"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        ta0 = time.perf_counter()
        r_agent = client.post(f"{agent_base}/recommend", json=alert)
        agent_ms = (time.perf_counter() - ta0) * 1000.0
        if r_agent.status_code != 200:
            raise SystemExit(
                f"agent /recommend failed mode={mode} status={r_agent.status_code} body={r_agent.text[:300]}"
            )
        return (time.perf_counter() - t_start) * 1000.0, None, agent_ms

    # full or edge_only — always start with /predict
    te0 = time.perf_counter()
    r_edge = client.post(f"{edge_base}/predict", json=window)
    edge_ms = (time.perf_counter() - te0) * 1000.0
    if r_edge.status_code != 200:
        raise SystemExit(
            f"edge /predict failed mode={mode} status={r_edge.status_code} body={r_edge.text[:300]}"
        )
    if mode == "edge_only":
        return (time.perf_counter() - t_start) * 1000.0, edge_ms, None

    alert = r_edge.json()
    ta0 = time.perf_counter()
    r_agent = client.post(f"{agent_base}/recommend", json=alert)
    agent_ms = (time.perf_counter() - ta0) * 1000.0
    if r_agent.status_code != 200:
        raise SystemExit(
            f"agent /recommend failed mode={mode} status={r_agent.status_code} body={r_agent.text[:300]}"
        )
    return (time.perf_counter() - t_start) * 1000.0, edge_ms, agent_ms


def main() -> None:
    p = argparse.ArgumentParser(description="E2E latency bench (predict + recommend).")
    p.add_argument("--edge-url", default="http://127.0.0.1:8000", help="edge_service base URL")
    p.add_argument("--agent-url", default="http://127.0.0.1:8001", help="agent_service base URL")
    p.add_argument("--iterations", type=int, default=50,
                   help="Number of iterations (1–5000).")
    p.add_argument(
        "--mode",
        choices=("full", "edge_only", "cloud_only"),
        default="full",
        help="Integration-mode ablation (assignment §4.6).",
    )
    p.add_argument(
        "--skip-agent",
        action="store_true",
        help="DEPRECATED — equivalent to --mode edge_only.",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Discard the first N iterations to avoid TLS / cold-cache outliers.",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=_ROOT / "reports" / "e2e_latency_last.json",
        help="Where to write timing JSON.",
    )
    args = p.parse_args()

    if args.skip_agent and args.mode == "full":
        args.mode = "edge_only"
    if not 1 <= args.iterations <= 5000:
        raise SystemExit(f"--iterations must be 1..5000, got {args.iterations}")
    if args.warmup < 0:
        raise SystemExit(f"--warmup must be ≥0, got {args.warmup}")

    edge_base = args.edge_url.rstrip("/")
    agent_base = args.agent_url.rstrip("/")
    window = _pv_window_payload()
    alert_template = _synthesised_pv_alert()

    totals: list[float] = []
    edge_ms: list[float] = []
    agent_ms: list[float] = []
    warmup_totals: list[float] = []

    t_wall_start = time.perf_counter()
    with httpx.Client(timeout=120.0) as client:
        for i in range(args.iterations + args.warmup):
            t_ms, e_ms, a_ms = _run_one_iter(
                client=client,
                mode=args.mode,
                edge_base=edge_base,
                agent_base=agent_base,
                window=window,
                alert_template=alert_template,
            )
            if i < args.warmup:
                warmup_totals.append(t_ms)
                continue
            totals.append(t_ms)
            if e_ms is not None:
                edge_ms.append(e_ms)
            if a_ms is not None:
                agent_ms.append(a_ms)
    wall_seconds = time.perf_counter() - t_wall_start

    out: dict = {
        "mode": args.mode,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "wall_seconds": round(wall_seconds, 3),
        "edge_url": edge_base,
        "agent_url": agent_base,
        "total_ms": _summarise(totals),
        "edge_ms": _summarise(edge_ms),
        "agent_ms": _summarise(agent_ms),
        "warmup_total_ms": _summarise(warmup_totals),
        "raw": {
            "total_ms": [round(v, 3) for v in totals],
            "edge_ms": [round(v, 3) for v in edge_ms],
            "agent_ms": [round(v, 3) for v in agent_ms],
        },
    }

    out["meets_p95_budget_10s"] = bool(out["total_ms"]["p95"] <= 10_000.0)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    headline = {k: out[k] for k in ("mode", "iterations", "wall_seconds")}
    headline["total_p50_ms"] = out["total_ms"]["p50"]
    headline["total_p95_ms"] = out["total_ms"]["p95"]
    headline["meets_p95_budget_10s"] = out["meets_p95_budget_10s"]
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
