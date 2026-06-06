# AgentPV Component 3 — `BESS` variant comparison

- **Split**: `test`
- **Baseline**: `pytorch_fp32`
- **N samples (test)**: 3395

| Variant | Macro-F1 | Δ vs base | Acc. | p50 (ms) | p95 (ms) | p99 (ms) | Size (KiB) | Size (MiB) | Δ size |
|---|---|---|---|---|---|---|---|---|---|
| `pytorch_fp32` | 0.9986 | — | 0.9985 | 0.543 | 0.820 | 1.641 | 188.07 | 0.1837 | — |
| `onnx_fp32` | 0.9986 | +0.0000 | 0.9985 | 0.087 | 0.112 | 0.160 | 179.33 | 0.1751 | ×1.05 |
| `onnx_int8` | 0.7016 | -0.2970 | 0.7016 | 0.056 | 0.078 | 0.111 | 59.50 | 0.0581 | ×3.16 |

**Notes**
- *Δ vs base* is `macro_f1(variant) - macro_f1(baseline)` (positive ⇒ variant is more accurate).
- *Δ size* is `baseline_kib / variant_kib` (higher ⇒ variant is more compressed).
- Latency is single-sample p50/p95/p99 in ms over ≥ 1000 timed CPU runs (Component 2 hard target: p95 ≤ 100 ms).
- Size budget per Component 2: ≤ 50 MiB on disk.
