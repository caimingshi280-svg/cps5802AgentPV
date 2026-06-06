# AgentPV — Component 5 LLM agent evaluation (Deliverable #6)

- **LLM backend**: `ollama`  (model identity: see `reports/agent_eval_artifact_meta.json` / Ollama tags).
- **Scenarios per ablation**: 33
- **Ablations**: `full, no_retrieve_knowledge, no_reasoning_trace`
- **Total scored records**: 99
- **LLM-as-judge rows scored**: 99
- **LLM-as-judge mean (1–5)**: 4.104

## 1. Headline (presentation snapshot)

![ablation summary](figures/agent_eval/ablation_summary.png)

The chart aggregates the four rubric checks (urgency match, must-contain keywords, forbidden phrases, knowledge-source minimum) plus the composite mean score per ablation. The `0.90` dashed line is the project quality bar.

## 2. Per-ablation aggregate

| Ablation | n | Mean | Median | Min | Max | % perfect | % urgency | % keywords | % forbidden | % knowledge | KB sources / scn | % with KB | Tool calls / scn | ReAct steps / scn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `full` | 33 | **0.9318** | 1.0000 | 0.7500 | 1.0000 | 72.73% | 100.00% | 72.73% | 100.00% | 100.00% | **2.61** | **90.91%** | 1.82 | 5.82 |
| `no_retrieve_knowledge` | 33 | **0.9242** | 1.0000 | 0.7500 | 1.0000 | 69.70% | 100.00% | 69.70% | 100.00% | 100.00% | **0.00** | **0.00%** | 1.06 | 5.82 |
| `no_reasoning_trace` | 33 | **0.9015** | 1.0000 | 0.7500 | 1.0000 | 60.61% | 100.00% | 60.61% | 100.00% | 100.00% | **2.67** | **90.91%** | 0.00 | 1.00 |

> The two **bold** columns are the operationally critical signal for the *No-RAG* ablation: even when the rubric's permissive `min_knowledge_sources` threshold (calibrated for the benchmark's Normal-class scenarios) cannot fire, the actual citation count drops to **zero** without retrieval — i.e. the recommendation is no longer *grounded* or *auditable*.

## 3. Provenance (No-RAG ablation operational signal)

![kb sources per ablation](figures/agent_eval/kb_sources_per_ablation.png)

**Left:** mean number of knowledge-base citations attached to each agent recommendation. **Right:** share of scenarios where the agent produced at least one grounded citation. The `no_retrieve_knowledge` ablation collapses both panels to zero — the LLM still produces plausible-looking recommendations, but they are no longer auditable. This is the headline finding for assignment §4.5 ablation #1.

## 4. Score distribution

![score distribution](figures/agent_eval/score_histogram.png)

## 5. Ablation impact (full − ablated, per scenario)

![ablation delta](figures/agent_eval/ablation_diff.png)

Positive bars (red) mean the *full* configuration scored higher than the ablated variant — i.e. removing that component cost quality. Bars at zero mean the ablation had no effect on this scenario. The `no_reasoning_trace` ablation strips the trace **after** ReAct completes, so any non-zero bars here come from heuristic checks that depend on the trace itself.

## 6. Tool usage and reasoning depth

![tool calls and reasoning](figures/agent_eval/tool_calls_reasoning.png)

## 7. Failure taxonomy

![failure taxonomy](figures/agent_eval/failure_taxonomy.png)

Across 99 scored records, **32** scored below 1.0. The chart above categorises *which check* failed; with a real LLM backend most failures are lexical drift on the `keywords` slot rather than wrong urgency or forbidden phrasing.

## 8. Example failures (up to 3 per ablation)

### `full` — 9 scenarios below the 1.0 bar

- **`scn_ambiguous_mixed_string_id`** → score **0.75** (failed: `keywords`, urgency=`scheduled`, kb=3):

  > Check the PV system for bypass diodes and ensure they are functioning correctly. Inspect the I-V curve of the affected cells to identify any multi-step curves indicative of partial shading. Perform a visual inspection of the PV string to verify that only part of it is receiving sunlight.

- **`scn_ambiguous_bess_cell_imbalance_warning`** → score **0.75** (failed: `keywords`, urgency=`scheduled`, kb=3):

  > Run a balancing cycle at low C-rate (0.2 C) on BESS_RACK_303 to address the cell imbalance issue.

- **`scn_ambiguous_pv_bypass_diode_warning`** → score **0.75** (failed: `keywords`, urgency=`scheduled`, kb=2):

  > Check the bypass diode for signs of partial shading, which can cause long-term thermal stress. Inspect the inverter's temperature and adjust the shading as needed to prevent damage.

### `no_retrieve_knowledge` — 10 scenarios below the 1.0 bar

- **`scn_pv_fault_string_disconnection`** → score **0.75** (failed: `keywords`, urgency=`immediate`, kb=0):

  > Trigger immediate intervention for PV_SITE_100 due to critical String_disconnection fault. Run diagnostic tests and perform repairs as soon as possible.

- **`scn_ambiguous_mixed_string_id`** → score **0.75** (failed: `keywords`, urgency=`scheduled`, kb=0):

  > Check the system's bypass diodes for proper function and ensure that the window hours are correctly set. Verify that there are no obstructions blocking sunlight from reaching the panels. Review the historical entries to confirm the correct sequence of faults. Consider running a diagnostic test to identify any further issues.

- **`scn_ambiguous_bess_cell_imbalance_warning`** → score **0.75** (failed: `keywords`, urgency=`scheduled`, kb=0):

  > Check the BESS system for any signs of physical damage or unusual environmental conditions. Run a diagnostic test to verify the cell imbalance reading and ensure it is not due to a temporary issue. If the issue persists, consider replacing the faulty cell.

### `no_reasoning_trace` — 13 scenarios below the 1.0 bar

- **`scn_bess_fault_thermal_anomaly`** → score **0.75** (failed: `keywords`, urgency=`immediate`, kb=0):

  > Immediately shut down the BESS system and initiate emergency repairs due to critical thermal anomaly. This is a high-priority action to prevent potential damage or safety risks.

- **`scn_bess_fault_cell_imbalance`** → score **0.75** (failed: `keywords`, urgency=`scheduled`, kb=3):

  > Perform a balancing cycle at low C-rate (0.2 C) to mitigate the cell imbalance. If the voltage spread does not converge below 30 mV after the cycle, schedule an in-rack capacity test.

- **`scn_ambiguous_mixed_string_id`** → score **0.75** (failed: `keywords`, urgency=`scheduled`, kb=3):

  > Check the bypass diodes and ensure they are functioning correctly. Inspect the PV string for any signs of partial shading or soiling. Consider performing a system diagnostic to identify any potential issues.

## 9. Notes on heuristic interpretation

The rubric was originally calibrated against the `mock` backend, which embeds the `[MOCK]` tag and the exact verb "Inspect". Real LLMs (Ollama `llama3.2`) produce semantically equivalent but lexically different outputs (e.g. "Check", "Verify", "Monitor"). The rubric's `_keyword_satisfied` now treats the imperative verbs `inspect / check / verify / assess / monitor / dispatch / isolate / review / investigate / examine` as an equivalence class, so the heuristic score remains a fair signal across backends. Hard keywords (fault-class names, system IDs) are still matched literally.