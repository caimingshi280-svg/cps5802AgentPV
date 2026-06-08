# Model evaluation (overview) — Deliverable #4

> **🎓 Final academic report (Deliverable §3)** is in
> [`reports/final_report.md`](final_report.md) (source) and
> [`reports/final_report.pdf`](final_report.pdf). This `model_eval.md` is the
> *index* document that the final report synthesises from.

This file indexes **edge model** evaluation artefacts (PV / BESS CNN-1D and ONNX variants).
The headline numbers below come from the last `python -m evaluation --compare` run on the
full-scale dataset (`n_samples = 50 500`, regenerated 2026-06-03).

## Headline (test split)

| System | Variant | Macro-F1 | p95 latency (ms) | Size (MiB) | Meets §4.2 budgets? |
|---|---|---:|---:|---:|---|
| PV   | `pytorch_fp32` | **0.9994** | 0.89 | 0.184 | yes |
| PV   | `onnx_fp32`    | **0.9994** | 0.15 | 0.176 | yes |
| PV   | `onnx_int8`    | **0.9994** | 0.10 | 0.058 | yes (×3.16 compression, lossless) |
| BESS | `pytorch_fp32` | **0.9980** | 1.03 | 0.184 | yes |
| BESS | `onnx_fp32`    | **0.9980** | 0.15 | 0.175 | yes |
| BESS | `onnx_int8`    | 0.7058 | 0.09 | 0.058 | partial — see compression tradeoff below |

Both FP32 variants clear the assignment §4.3 target Macro-F1 ≥ 0.90, all variants are well
under the 50 MiB size budget, and **every** p95 ≤ 100 ms budget passes. The single PT FP32 ↔
ONNX FP32 column shows ≈ 6× CPU latency speedup from `onnxruntime`'s graph optimizations.

## Compression tradeoff (assignment §4.3 — required discussion)

- **PV INT8 is lossless** — the smooth, well-separated PV fault signatures survive per-tensor
  MinMax INT8 quantization with no measurable Macro-F1 loss while shrinking 3.16× and gaining
  another ~30 % latency.
- **BESS INT8 drops 29.2 pp Macro-F1** (0.998 → 0.706). Several BESS classes (`BESS_Normal`,
  `Thermal_anomaly`, `Internal_resistance_increase`) sit in narrow numeric bands of `R_est`,
  `sigma_V`, and SoC-trajectory features; per-tensor MinMax INT8 collapses these distinctions.
  This is the **canonical accuracy/size tradeoff** that §4.3 asks teams to characterize. Two
  in-budget remediations are planned and documented for the report: (a) Entropy (KL)
  calibration via `onnxruntime.quantization.CalibrationMethod.Entropy`, (b) per-channel
  weight quantization. Until then, **production deployment defaults to BESS FP32 ONNX** so
  the edge service stays above 90 % Macro-F1 on both systems.

## Robustness, distribution shift & OOD (Component 3 extension)

The §4.3 numbers above are necessary but not sufficient — the course instructor's
2026-05-13 feedback called out *deployment-realism* evaluation. A full robustness
suite (distribution shift, missing / corrupted features, sensor noise, calibration
drift, FGSM adversarial perturbation, cross-system OOD detection, and an
**energy-based selective-prediction policy**) ships in [`reports/robustness_eval.md`](robustness_eval.md).

Highlights (energy-based logit OOD detector, Liu et al. 2020, calibrated to 95 % in-distribution coverage on the val split):

| System | Clean Macro-F1 | OOD discriminability | Direction | Selective accuracy @95 % coverage | OOD reject rate @threshold |
| --- | ---: | ---: | --- | ---: | ---: |
| PV   | 0.9994 | 0.6037 | inverted (cross-system OOD scores higher) | 1.0000 | 0.3281 |
| BESS | 0.9980 | 1.0000 | inverted (cross-system OOD scores higher) | 0.9994 | 0.0000 |

Per-system summaries, JSON, and 9 presentation-grade figures per system are in
[`reports/robustness/pv/`](robustness/pv/summary.md) and [`reports/robustness/bess/`](robustness/bess/summary.md).
The top-level `robustness_eval.md` also documents *when the strategy succeeds*
(in-distribution rejection, mild noise / drift) and *when it fails* (missing
features, cross-system score inversion, FGSM ≥ 0.05) so the project's evaluation
of stress behaviour is critical rather than cherry-picked.

## LLM agent evaluation (Component 5 — Deliverable #6)

The cloud-tier ReAct agent is graded against a **33-scenario benchmark** with **three
ablations** (`full`, `no_retrieve_knowledge` (No-RAG), `no_reasoning_trace`) using a
*real* local LLM (`ollama / llama3.2`, 2 GiB). Headline rubric numbers
(source: `agent_eval/results/last_run_three_ablations_with_judge.json`, 2026-06-03):

| Ablation | Mean heuristic | % urgency ✓ | % keywords ✓ | % forbidden ✓ | % knowledge ✓ | **KB src / scn** | **% with KB** | Tools / scn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full`                 | **0.9318** | 100 % | 72.7 % | 100 % | 100 % | **2.61** | **91 %** | 1.82 |
| `no_retrieve_knowledge` | 0.9242 | 100 % | 69.7 % | 100 % | 100 % | **0.00** | **0 %**  | 1.06 |
| `no_reasoning_trace`   | 0.9015 | 100 % | 60.6 % | 100 % | 100 % | 2.67 | 91 % | 0.00 |

**LLM-as-judge mean (1–5): 4.10** (99 / 99 scored). Planner JSON fallback to mock plan
is observed in agent HTTP logs (`ollama_plan_fallback_mock`) — graceful recovery without
crashing the benchmark runner.
The two highlighted columns are the headline No-RAG finding (assignment §4.5
ablation #1): disabling `retrieve_knowledge` keeps surface scores stable
but **collapses citation count from 2.61 → 0** and the share of grounded
recommendations from **91 % → 0 %** — i.e. the LLM still produces plausible
text but it is no longer *auditable*.

Full report, ablation-impact figures, run provenance and per-scenario example
recommendations are in [`reports/agent_eval.md`](agent_eval.md).

## Integration evaluation (Component 6 — Deliverables #7 & #8)

End-to-end multi-node integration / latency / graceful-degradation evaluation
ships separately in [`reports/integration_eval.md`](integration_eval.md). The
three integration-mode ablation (assignment §4.6) was measured against the
running `edge_service` (port 8000) and `agent_service` (port 8001) for **50
iterations per mode**, plus a **10-node orchestrator** session running
`--nodes pv6_bess4 --duration 60`:

| Mode | Iter | P50 (ms) | P95 (ms) | Notes |
| --- | ---: | ---: | ---: | --- |
| `edge_only`  | 50 | 4.50   | 5.69   | classifier-only graceful-degradation floor |
| `full`       | 50 | 8 438  | 9 803  | edge → agent, dominated by Ollama `llama3.2` |
| `cloud_only` | 50 | 9 163  | 9 941  | raw alert → agent (no edge classifier) |

All three modes stay under the 10 s P95 budget. The 10-node orchestrator run
(with `--http-timeout 120` recommended) emitted **144 events / 10 distinct nodes**
in 60 s; **3** of those events captured real `agent_recommend_failed` timeouts
that the orchestrator absorbed without crashing, providing the empirical evidence
Deliverable #7 asks for.

Per-mode JSONs, raw timings, 5 presentation-grade figures (`01_latency_bars`,
`02_latency_violin`, `03_edge_vs_agent_split`, `04_node_fanout`,
`05_severity_mix`) and the per-node fan-out table live under
`reports/integration/` and `reports/figures/integration/`.

## Interactive fault injection (Component 7 — Deliverable #9)

The dashboard's sidebar now exposes a **🔥 Fault injection (demo)** panel
that drives the *same* HTTP path as the orchestrator (`POST /predict` →
`POST /recommend`) but for a single user-selected fault. Implementation
lives in [`dashboard/inject.py`](../dashboard/inject.py) and is unit
tested with `httpx.MockTransport` in
[`tests/unit/test_dashboard_inject.py`](../tests/unit/test_dashboard_inject.py)
(11 tests covering happy path, monitor severity skipping the agent,
explicit `skip_agent` graceful degradation, edge / agent HTTP errors,
`persist=False` dry-run, and seed determinism).

A scripted, reproducible run of 5 representative scenarios (PV
`Inverter_fault` / `Partial_shading`, BESS `Thermal_anomaly`, normal
window, and a forced edge-only `String_disconnection`) is documented in
[`reports/integration/fault_injection_demo.md`](integration/fault_injection_demo.md)
together with the raw JSON sidecar
(`reports/integration/fault_injection_demo.json`) and a 5-event JSONL
replay file (`data/orchestrator/events_c7_demo.jsonl`).

## Layout

| Asset | Path |
| --- | --- |
| PV §4.3 comparison table | `reports/pv/comparison.md` |
| BESS §4.3 comparison table | `reports/bess/comparison.md` |
| Per-backend summaries | `reports/pv/{pytorch_fp32,onnx_fp32,onnx_int8}/` and the parallel tree under `reports/bess/` |
| Build provenance | `reports/edge_onnx_build_meta.json` |
| Robustness / OOD overview | `reports/robustness_eval.md` |
| Robustness per-system | `reports/robustness/{pv,bess}/summary.{md,json}` + `figures/` |
| LLM agent benchmark (real Ollama) | `reports/agent_eval.md`, `reports/figures/agent_eval/`, `agent_eval/results/last_run_three_ablations_with_judge.{json,log}` |
| Integration / latency / 10-node fan-out | `reports/integration_eval.md`, `reports/figures/integration/`, `reports/integration/e2e_latency_*.json`, `data/orchestrator/events.jsonl` |
| Interactive fault injection (C7) | `dashboard/inject.py`, `dashboard/app.py` sidebar expander, `scripts/demo_fault_injection.py`, `reports/integration/fault_injection_demo.{md,json}`, `data/orchestrator/events_c7_demo.jsonl` |
| **Final academic report (C8)** | `reports/final_report.md` · `reports/final_report.pdf` |

## Regeneration

```powershell
python -m simulation.generate_dataset --seed 42 --n-pv 28000 --n-bess 22500 `
    --n-pv-normal 8000 --n-bess-normal 5000
python -m training.train --system pv   --epochs 25 --early-stop-patience 6 --batch-size 256
python -m training.train --system bess --epochs 30 --early-stop-patience 8 --batch-size 256
python -m quantization.onnx_export --checkpoint quantization/artifacts/cnn1d_pv_best.pt   --output quantization/artifacts/cnn1d_pv.onnx
python -m quantization.onnx_export --checkpoint quantization/artifacts/cnn1d_bess_best.pt --output quantization/artifacts/cnn1d_bess.onnx
python -m quantization.int8_static --system pv
python -m quantization.int8_static --system bess
python -m evaluation --compare
python scripts/run_robustness_eval.py
$env:APP_ENV='dev'
$env:AGENTPV_JUDGE_API_BASE='http://127.0.0.1:11434/v1'
$env:AGENTPV_JUDGE_MODEL='llama3.2:latest'
python -m agent_eval --ablations full no_retrieve_knowledge no_reasoning_trace `
    --llm-backend ollama `
    --out-json agent_eval/results/last_run_three_ablations_with_judge.json `
    --out-md reports/agent_eval_last_run_with_judge.md
python scripts/render_agent_eval_report.py `
    --input agent_eval/results/last_run_three_ablations_with_judge.json `
    --log   agent_eval/results/last_run_three_ablations_with_judge.log
# Component 6 integration (run uvicorn services in two extra terminals first):
python scripts/e2e_latency_bench.py --mode edge_only  --iterations 50 --warmup 3 --out-json reports/integration/e2e_latency_edge_only.json
python scripts/e2e_latency_bench.py --mode full       --iterations 50 --warmup 3 --out-json reports/integration/e2e_latency_full.json
python scripts/e2e_latency_bench.py --mode cloud_only --iterations 50 --warmup 3 --out-json reports/integration/e2e_latency_cloud_only.json
python -m orchestrator --nodes pv6_bess4 --duration 60 --http-timeout 120 --out data/orchestrator/events.jsonl
python scripts/render_integration_eval_report.py
# Component 7 interactive fault injection demo (keep edge+agent running):
python scripts/demo_fault_injection.py `
    --events-path data/orchestrator/events_c7_demo.jsonl
# Live interactive: streamlit run dashboard/app.py
```
