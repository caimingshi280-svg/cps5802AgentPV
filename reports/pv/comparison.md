# AgentPV Component 3 — `PV` variant comparison

- **Split**: `test`
- **Baseline**: `pytorch_fp32`
- **N samples (test)**: 4211

| Variant | Macro-F1 | Δ vs base | Acc. | p50 (ms) | p95 (ms) | p99 (ms) | Size (KiB) | Size (MiB) | Δ size |
|---|---|---|---|---|---|---|---|---|---|
| `pytorch_fp32` | 0.9993 | — | 0.9993 | 0.559 | 1.014 | 3.289 | 188.45 | 0.1840 | — |
| `onnx_fp32` | 0.9993 | +0.0000 | 0.9993 | 0.110 | 0.132 | 0.145 | 179.85 | 0.1756 | ×1.05 |
| `onnx_int8` | 0.9995 | +0.0002 | 0.9995 | 0.058 | 0.082 | 0.100 | 59.66 | 0.0583 | ×3.16 |

**Notes**
- *Δ vs base* is `macro_f1(variant) - macro_f1(baseline)` (positive ⇒ variant is more accurate).
- *Δ size* is `baseline_kib / variant_kib` (higher ⇒ variant is more compressed).
- Latency is single-sample p50/p95/p99 in ms over ≥ 1000 timed CPU runs (Component 2 hard target: p95 ≤ 100 ms).
- Size budget per Component 2: ≤ 50 MiB on disk.
