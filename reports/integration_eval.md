# Component 6 — Integration Evaluation

> Assignment §4.6 / Deliverables 7–8. End-to-end latency, integration-mode
> ablation (`full` vs `edge_only` vs `cloud_only`), multi-node orchestrator
> fan-out evidence, and graceful-degradation behaviour.

_Generated_: 2026-06-03T16:17Z

_Reproduce_:

```powershell
# 1. Start services in two terminals
python -m uvicorn api.edge_service:app  --host 127.0.0.1 --port 8000
python -m uvicorn api.agent_service:app --host 127.0.0.1 --port 8001

# 2. Bench each integration mode (50 iterations + 3 warmup)
python scripts/e2e_latency_bench.py --mode edge_only --iterations 50 --warmup 3 --out-json reports/integration/e2e_latency_edge_only.json
python scripts/e2e_latency_bench.py --mode full --iterations 50 --warmup 3 --out-json reports/integration/e2e_latency_full.json
python scripts/e2e_latency_bench.py --mode cloud_only --iterations 50 --warmup 3 --out-json reports/integration/e2e_latency_cloud_only.json

# 3. Run the 10-node orchestrator for 60 s
python -m orchestrator --nodes pv6_bess4 --duration 60 --http-timeout 120 --out data/orchestrator/events.jsonl

# 4. Re-render this report
python scripts/render_integration_eval_report.py
```

## 1. Headline numbers

| Integration mode | Iter | P50 (ms) | P95 (ms) | P99 (ms) | Max (ms) | Meets 10 s P95? |
| --- | --- | --- | --- | --- | --- | --- |
| edge_only (no agent) | 50 | 4.50 | 5.69 | 6.47 | 6.47 | ✅ |
| full (edge → agent) | 50 | 8437.94 | 9802.82 | 10001.50 | 10001.50 | ✅ |
| cloud_only (no edge) | 50 | 9163.04 | 9941.22 | 10626.07 | 10626.07 | ✅ |

Decomposition of the **full**-mode P50/P95 (edge ↔ agent split):
- **edge /predict** — N=50, P50=4.71 ms, P95=26.27 ms, P99=28.28 ms, mean=9.87 ms, max=28.28 ms
- **agent /recommend** — N=50, P50=8433.92 ms, P95=9784.69 ms, P99=9997.64 ms, mean=8533.60 ms, max=9997.64 ms

![](figures/integration/01_latency_bars.png)  
![](figures/integration/02_latency_violin.png)
![](figures/integration/03_edge_vs_agent_split.png)

## 2. Integration-mode ablation — interpretation

- `edge_only` is the **graceful-degradation floor**: classifier-only P50 = **4.50 ms**, dominated by ONNX inference + JSON ser/de. This is what the operator sees when the agent or its LLM is unavailable.

- `full` pipeline P50 = **8438 ms** (~1875× edge), almost entirely attributable to Ollama `llama3.2` round-trips. The edge stage adds <1 % of total latency.

- `cloud_only` (raw-alert → agent, no edge classifier) P50 = **9163 ms**, P95 = **9941 ms** — confirms the agent is the latency bottleneck and that bypassing edge does **not** buy back significant time; instead it loses the structured fault-class label that gives the agent better tool-use grounding.

- All modes stay under the 10 s P95 budget (assignment §3 requirement); the `full` mode is the closest to the ceiling and is the primary subject of optimisation in future work (faster local LLM, response caching, speculative decoding).


## 3. Multi-node fan-out (≥10 nodes)

Orchestrator events log: `data/orchestrator/events.jsonl`  
Total events: **144**, distinct nodes: **10**.

| Node | System | Events | Alerts | Reco. | Errors | Top fault |
| --- | --- | --- | --- | --- | --- | --- |
| `bess-001` | BESS | 4 | 4 | 1 | 0 | BESS_Normal (3) |
| `bess-002` | BESS | 3 | 3 | 1 | 0 | BESS_Normal (2) |
| `bess-003` | BESS | 59 | 59 | 0 | 0 | BESS_Normal (30) |
| `bess-004` | BESS | 5 | 5 | 1 | 0 | BESS_Normal (4) |
| `pv-001` | PV | 4 | 4 | 1 | 1 | PV_Normal (2) |
| `pv-002` | PV | 4 | 4 | 1 | 1 | PV_Normal (2) |
| `pv-003` | PV | 2 | 2 | 1 | 0 | PV_Normal (1) |
| `pv-004` | PV | 59 | 59 | 0 | 0 | PV_Normal (47) |
| `pv-005` | PV | 2 | 2 | 1 | 1 | Inverter_fault (2) |
| `pv-006` | PV | 2 | 2 | 2 | 0 | String_disconnection (2) |

![](figures/integration/04_node_fanout.png)  
![](figures/integration/05_severity_mix.png)

## 4. Decision quality across modes

* `edge_only` returns a typed Alert (`fault_class`, `severity`, `confidence`, `sensor_snapshot`) — sufficient for a SCADA / dashboard operator to triage but lacks the natural-language playbook.
* `full` adds a `Recommendation` with `action` (imperative), `urgency`, `confidence`, optional `escalate_to`, and **0–5 RAG knowledge sources** with `chunk_id` + relevance. This is the interpretability premium that justifies the ~2-orders-of-magnitude latency cost in `full` mode.
* `cloud_only` exercises the agent's tolerance to coarser inputs: the Alert is synthesised from raw sensors (no ML classifier). The agent still produces a valid Recommendation through tool-use + RAG, demonstrating the system retains usefulness when the edge classifier is unavailable.


## 5. Graceful-degradation evidence

Sources of failure isolation in the current architecture:

1. **Edge unavailable** → orchestrator catches `ClientError`, sets `OrchestratorEvent.error`, continues the loop (see `orchestrator/node_simulator.py`).
2. **Agent unavailable** → recommendations are skipped but alerts still land in the events log; the dashboard still renders severity timelines.
3. **Ollama unavailable / refuses JSON** → the ReAct agent falls back to the mock planner (`ollama_plan_fallback_mock`, logged) and returns a valid Recommendation derived from the rubric defaults.
4. **Per-node fault probabilities** are independent — failure of one node never blocks another node's tick (each NodeRunner is a separate asyncio task with `try/except` around `step`).

Empirical errors observed in the run above: **3** node-level errors across all nodes (see per-node table).

Error taxonomy (orchestrator-side categorisation):
| Error kind | Count |
| --- | --- |
| `agent_recommend_failed` | 3 |

_The_ `agent_recommend_failed` _entries are httpx_ `TimeoutException` _occurrences under concurrent 10-node fan-out. Tail latency of `full` mode is ~**9.8 s** P95 (see §1); reproduction commands use `--http-timeout 120` to reduce spurious agent timeouts. The orchestrator continued running and emitted **144** events anyway — exactly the graceful-degradation behaviour deliverable §7 asks for._

## 6. Files produced

- `reports/integration_eval.md` (this report)
- `reports/integration/e2e_latency_edge_only.json` — raw 50-run timings
- `reports/integration/e2e_latency_full.json` — raw 50-run timings
- `reports/integration/e2e_latency_cloud_only.json` — raw 50-run timings
- `reports/figures/integration/01_latency_bars.png` — figure (`latency_bar`)
- `reports/figures/integration/02_latency_violin.png` — figure (`latency_violin`)
- `reports/figures/integration/03_edge_vs_agent_split.png` — figure (`edge_agent_split`)
- `reports/figures/integration/04_node_fanout.png` — figure (`node_fanout`)
- `reports/figures/integration/05_severity_mix.png` — figure (`severity_mix`)
- `data/orchestrator/events.jsonl` — orchestrator JSONL
- `reports/integration_eval_meta.json` — provenance pointer
