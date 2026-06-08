# `evaluation/` — Model Evaluation (Component 3)

Project rule §24, 14-section template. This module is the test-bench for
the trained edge classifier: it answers "did Component 2 actually meet
the assignment's numerical requirements?".

---

## 1. Module purpose

Implement assignment §4.3 — rigorously evaluate the multi-class fault
classifier across:

- per-class precision / recall / F1,
- macro-average F1 (target ≥ 0.90),
- confusion matrix (heatmap),
- CPU latency over 1 000 runs (target ≤ 100 ms p95),
- on-disk model size (target ≤ 50 MiB).

All against the held-out **test** split with the **ONNX** artefact (the
deployment binary, not the PyTorch checkpoint) so the numbers reflect
edge reality rather than research-cluster reality.

---

## 2. Why it exists

Component 2 produces a model. Component 3 has to prove that model is
*good enough* — and that the trade-offs it makes are documented. A
classifier with 95 % accuracy that fails entirely on the rare critical
classes is dangerous in safety-critical PV/BESS operations
(assignment §4.3.0). Per-class evaluation is the difference between
"shipping" and "guessing".

---

## 3. Architecture

```text
test split (npz/csv) ──► OnnxClassifier.run_logits ─► EvaluationPredictions
                                                          │
                ┌─────────────────────────────────────────┼─────────────────────────────┐
                ▼                                         ▼                             ▼
   classification_report.py                     confusion_matrix.py             latency_benchmark.py
        per-class P/R/F1                         compute + render PNG               1 000 timed runs
        macro / weighted F1
                                                                                       │
                                                                                       ▼
                                                                                model_size.py
                                                                              file → bytes/KiB/MiB

                              ▼

                     evaluation/runner.py  ──►  reports/<system>/{summary.json, summary.md, confusion_matrix.png}
                              │
                              ▼
                     evaluation/__main__.py   (CLI)
```

The package is split into pure-numeric kernels (testable in <1 s without
matplotlib) and a runner that stitches them with file I/O (rule §27
minimal-viable + testable layer separation).

---

## 4. Key files

| File | Role |
|---|---|
| `metrics.py` | `EvaluationPredictions`, `accuracy`, `macro_f1`, `per_class_metrics` (sklearn-backed pure functions). |
| `classification_report.py` | `ClassificationReport` dataclass + JSON / Markdown writers. |
| `confusion_matrix.py` | `compute_confusion_matrix` (NumPy) + `render_confusion_matrix_png` (matplotlib, lazy import). |
| `latency_benchmark.py` | `benchmark_latency` — generic; takes any `predict_fn`. |
| `model_size.py` | `measure_model_size` — bytes / KiB / MiB + budget gate. |
| `predictor.py` | `Predictor` Protocol — implemented by ONNX & PyTorch backends so `evaluate_predictor` is backend-agnostic. |
| `pytorch_runner.py` | `PyTorchClassifier` (FP32 baseline) + `measure_pytorch_state_dict_bytes`. **(S12)** |
| `runner.py` | `evaluate_predictor(...)` core + `evaluate_onnx` / `evaluate_pytorch` thin wrappers. |
| `compare_variants.py` | `VariantRow`, `VariantComparison`, `build_comparison_rows`, `comparison_to_markdown`, `render_tradeoff_png`, `compare_variants`. **(S12)** |
| `__main__.py` | CLI: `python -m evaluation [--systems pv bess] [--variants ...] [--compare]`. |

---

## 5. Inputs & outputs

### Inputs

- ONNX model file at `quantization/artifacts/cnn1d_{pv,bess}.onnx`
  (must include `agentpv.system_type`, `agentpv.label_classes`, and
  `agentpv.feature_stats` metadata — this is the contract from
  `quantization/onnx_export.py`).
- Dataset: `data/processed/X_{pv,bess}.npz` + `meta_{pv,bess}.csv` +
  `data/splits/{test,val,train}.csv`.

### Outputs (per system, per variant — S12 layout)

- `reports/<system>/<variant>/summary.json` — fully-typed metrics payload:
  classification report, full confusion matrix, latency stats, size.
- `reports/<system>/<variant>/summary.md` — human-readable Markdown with pass /
  fail badges (✅ / ❌) against assignment thresholds.
- `reports/<system>/<variant>/confusion_matrix.png` — row-normalised heatmap with
  raw integer counts annotated.
- **(S12 cross-variant)** `reports/<system>/comparison.{md,json}` and
  `reports/<system>/comparison_tradeoff.png` — produced when running
  `python -m evaluation --compare` with ≥ 2 variants.

`<variant>` is one of `pytorch_fp32` / `onnx_fp32` / `onnx_int8`.

---

## 6. Design decisions

| Decision | Alternatives | Why |
|---|---|---|
| Numeric kernels separate from runner | One monolithic file | Each kernel is a 30-line pure function, trivially unit-testable. |
| Backed by `sklearn.metrics` | Hand-rolled NumPy | sklearn is already pinned (rule §23); standard implementation passes academic scrutiny. |
| Lazy `import matplotlib` inside `render_confusion_matrix_png` | Top-level import | Math kernels stay importable without GUI backend; tests of the math run in <1 s. |
| `OnnxClassifier.run_logits` (public, batch input) | Re-implement ORT call here | Single source of truth for the input contract (`(B, T, F)` raw float32 — graph applies standardisation). |
| Markdown summary is **per-system** | One global report | Each summary is self-contained for the report's appendix; a global merger is polish-phase. |
| Model size budget = **50 MiB** | 50 MB (decimal) | Assignment text says "50 MB"; we treat MiB as the slightly stricter interpretation. |
| Latency: synthetic input, fixed seed, 50 warm-up | Use real test windows | Measures **graph** latency, not data-loading; assignment requires CPU inference benchmarking. |
| Latency reports p50 / p95 / p99 | Only mean | p95 is the assignment metric; p50 / p99 add operational signal at zero cost. |
| Confusion matrix annotates raw counts on a row-normalised colour map | Either-or | Counts are needed for academic interpretation; row-normalised colour exposes per-class recall visually. |
| **(S12)** `Predictor` Protocol abstracts backend | Subclassing | Each backend (ONNX, PyTorch, future TFLite) only has to expose 3 attrs + 1 method; no inheritance ceremony. |
| **(S12)** `evaluate_predictor` core; `evaluate_onnx` / `evaluate_pytorch` thin wrappers | One function per backend | Single source of truth for metric definitions; output schema stays consistent across variants. |
| **(S12)** `compare_variants` is per-system, not global | One mega comparison | PV (7 classes) and BESS (5 classes) class taxonomies don't align; per-system tables are honest. |
| **(S12)** Default baseline = `pytorch_fp32` | First listed variant | PT is the closest thing to "ground truth FP32 reference"; deltas vs PT directly answer "did ONNX export / INT8 lose accuracy?". |
| **(S12)** Tradeoff plot uses `matplotlib` | Plotly / static seaborn | Already pinned (S11); zero new deps; PNG embeds in PDF cleanly. |

---

## 7. What does NOT live here

- Training / hyper-parameter tuning → `training/`.
- Model architectures → `models/`.
- ONNX export / quantization → `quantization/`.
- ONNX inference plumbing → `inference/`.
- Long-form ablation reports & figures → `reports/` (this module just
  writes the *numbers*; narrative analysis goes elsewhere).

---

## 8. Teaching notes

- **Why per-class is non-negotiable**: assignment §4.3.0 explicitly
  warns against "95 % accuracy by correctly classifying the majority
  Normal class while failing on rare fault classes". Macro-F1 + the
  per-class table are the safety net.
- **Why latency uses synthetic input**: the test split has only ~7 600
  rows; we need ≥ 1 000 timed runs and we should *not* mix data-loading
  time into model-latency numbers. Synthetic random input gives the
  same per-call cost without I/O bias.
- **Why we re-run with `n_warmup=50`**: ORT lazily initialises threads
  + Conv kernels; the first 5–10 calls can be 5× slower. Warm-up isn't
  cheating — the real edge service is also warm by the time it serves
  alerts.

---

## 9. Interfaces with other modules

| Upstream | Downstream |
|---|---|
| `inference.onnx_runner.OnnxClassifier` | — (this is a leaf evaluation tool) |
| `training.data._load_split_arrays` | — |
| `api.schemas.{SystemType, SplitName}` | — |
| `quantization/artifacts/*.onnx` | `reports/<system>/*` |

We do **not** depend on FastAPI, Streamlit, or the agent layer — pure
Python + numpy + sklearn + matplotlib + onnxruntime.

---

## 10. Tests

`tests/unit/test_evaluation.py` (25 cases — S11 kernels):

| Group | Coverage |
|---|---|
| `EvaluationPredictions` | shape / length / label-range / non-empty validation |
| `accuracy` / `macro_f1` | perfect predictions / empty / missing class penalty |
| `per_class_metrics` | one row per class / unseen class produces 0-row |
| `to_dict` | floats are stably rounded |
| `ClassificationReport` | aggregate values / JSON round-trip / Markdown structure / empty-support handling |
| `compute_confusion_matrix` | counts / shape / empty matrix / shape-mismatch raises |
| `render_confusion_matrix_png` | normalised + non-normalised file write |
| `benchmark_latency` | call-count, output ordering, extra propagation, input validation |
| `measure_model_size` | within / over budget / missing file / bad budget |

`tests/unit/test_pytorch_runner.py` (10 cases — S12):

| Group | Coverage |
|---|---|
| `PyTorchClassifier` | satisfies `Predictor` protocol / output shape / determinism / shape & channel validation / missing checkpoint |
| Checkpoint compatibility | unknown `model_arch` rejected / required fields enforced |
| `measure_pytorch_state_dict_bytes` | positive value / close to full checkpoint / missing path raises |

`tests/unit/test_compare_variants.py` (13 cases — S12):

| Group | Coverage |
|---|---|
| `load_variant_summary` | round-trip / missing file / missing-keys validation |
| `build_comparison_rows` | default-baseline pick / fallback / mixed-system rejection / mixed-split rejection / duplicate names / unknown baseline / empty input |
| `comparison_to_markdown` | column structure / baseline em-dash / compression ratio formatting |
| `VariantRow` | freeze / `to_json` |
| `compare_variants` | end-to-end MD + JSON + PNG written |

Runner-level integration is exercised live via
`python -m evaluation --variants pytorch_fp32 onnx_fp32 onnx_int8 --compare`
against the real PV + BESS artefacts (headline numbers in
`reports/model_eval.md` and `reports/pv/comparison.md`).

---

## 11. Performance budget

| Step | Budget | Actual |
|---|---|---|
| Per-class metrics on 7 600 samples | < 100 ms | ~30 ms |
| Confusion matrix PNG (7×7) | < 1 s | ~250 ms |
| 1 000-run latency benchmark | < 5 s | ~200 ms |
| Total `evaluate_onnx` per system | < 10 s | ~3 s |

(`time python -m evaluation` end-to-end on this dev box: ~6 s wall.)

---

## 12. Future work (Polish phase)

- **t-SNE visualisation** of feature-extractor embeddings per class
  (assignment §4.3.0 "error analysis").
- **SHAP feature attribution** for misclassified samples.
- **Cross-condition slicing**: per-`operating_condition` Macro-F1 to
  detect distribution shift.
- **Calibration plots**: confidence vs accuracy bins (ECE) — relevant
  to the alert severity mapping in `inference/postprocess.py`.
- **Pruning / KD variants** added to the comparison table once
  `quantization/prune.py` and `kd.py` land.
- **Per-condition × per-variant matrix**: BESS INT8 may regress more
  on `Cell_imbalance` under high SOC; current per-class table catches
  per-class regression but not per-condition ones.
- **Latency under contention** — current p95 is single-thread / single
  client; multi-tenant edge nodes need contention modelling.

---

## 13. Usage example

```powershell
# Single-variant ONNX evaluation (S11 era):
python -m evaluation --variants onnx_fp32

# Three-variant comparison (S12 — the assignment-grade run):
python -m evaluation \
    --systems pv bess \
    --variants pytorch_fp32 onnx_fp32 onnx_int8 \
    --compare

# Sanity-check against val split:
python -m evaluation --split val --variants onnx_fp32

# Custom latency budget:
python -m evaluation --n-latency-runs 5000 --latency-seed 7
```

Programmatic API:

```python
from pathlib import Path

from api.schemas import SystemType
from evaluation.runner import evaluate_onnx, evaluate_pytorch
from evaluation.compare_variants import compare_variants

# 1) Per variant
evaluate_pytorch(
    checkpoint_path=Path("quantization/artifacts/cnn1d_pv_best.pt"),
    system_type=SystemType.PV,
    variant_name="pytorch_fp32",
    out_dir=Path("reports/pv/pytorch_fp32"),
)
evaluate_onnx(
    onnx_path=Path("quantization/artifacts/cnn1d_pv.onnx"),
    system_type=SystemType.PV,
    variant_name="onnx_fp32",
    out_dir=Path("reports/pv/onnx_fp32"),
)
evaluate_onnx(
    onnx_path=Path("quantization/artifacts/cnn1d_pv_int8.onnx"),
    system_type=SystemType.PV,
    variant_name="onnx_int8",
    out_dir=Path("reports/pv/onnx_int8"),
)

# 2) Cross-variant comparison
md, js, png = compare_variants(
    summary_paths=[
        Path("reports/pv/pytorch_fp32/summary.json"),
        Path("reports/pv/onnx_fp32/summary.json"),
        Path("reports/pv/onnx_int8/summary.json"),
    ],
    out_dir=Path("reports/pv"),
)
```

---

## 14. Known limitations

- **3 variants currently** (PT FP32 / ONNX FP32 / ONNX INT8). Pruning &
  KD variants land in polish phase via `quantization/prune.py` &
  `quantization/kd.py`; the comparison code is already
  variant-count-agnostic.
- **Latency uses a single thread of ORT** — the assignment is about
  edge inference (single-sample, low-context); multi-thread benchmarks
  are out of scope.
- **No streaming / online evaluation** — we evaluate over the static
  test split only. Operator-time evaluation lives in
  `agent_eval/` (Component 5) and the orchestrator (Component 6/7).
- **Matplotlib is used for the heatmap and tradeoff plots** — large
  categorical tick labels can overlap; polish phase may switch to
  seaborn or a custom layout if the report demands it.
- **`reports/<system>/<variant>/summary.md` references PNG via relative
  path** — if a reader copies the Markdown without the PNG the link
  breaks; the polish phase will produce a single bundled HTML report
  instead.
- **PyTorch baseline runs ~10× slower than ONNX** at p95 (0.96 ms vs
  0.17 ms for PV); this is expected (Python overhead) and **not a
  concern for the deployed binary** — only the ONNX variants ship to
  the edge.
- **BESS INT8 loses 0.7% Macro-F1** vs FP32; still ≫ assignment 0.85
  threshold, but a polish opportunity (try `Entropy` calibration or
  per-channel weights to recover).
