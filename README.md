# AgentPV

A cloud-edge intelligent monitoring and decision-support system for photovoltaic
(PV) solar and battery energy storage operations. Built for **CPS 5802 — Machine
Learning and Innovations, Spring 2026**.

## Architecture

Three layers, communicating via a fixed JSON alert contract
(`docs/alert_schema.json`):

1. **Simulation Layer** — `simulation/` — physics-based time-series generator
   with labeled fault injection.
2. **Edge AI Layer** — `models/`, `training/`, `quantization/`, `inference/`,
   `evaluation/` — quantized ONNX classifier with severity output.
3. **Cloud Agent Layer** — `agent/`, `rag/`, `tools/`, `agent_eval/` — ReAct
   agent with retrieval-augmented generation and structured recommendations.

Glue layers: `api/` (FastAPI services), `dashboard/` (Streamlit UI),
`orchestrator/` (multi-node simulation), `configs/`, `utils/`, `tests/`.

The full module-by-module plan lives in `docs/AgentPV-项目方案.md`. The engineering
rules every contributor must follow are in `projectdesignrules.cursorrules`.

## Quickstart (development)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# .venv/bin/activate              # macOS/Linux

pip install -e ".[dev]"

cp .env.example .env             # then fill in API keys if using cloud LLM
pytest -q                         # ~284 tests on contracts and pipelines
```

Requires **Python 3.11 or 3.12** (`pyproject.toml`).

## Quickstart (full system — defence demo)

> Recommended for Final Presentation: **local uvicorn + Streamlit** (do not run
> `docker compose` on the same ports at the same time).

```powershell
# Terminal A
$env:APP_ENV = "dev"
python -m uvicorn api.edge_service:app --host 0.0.0.0 --port 8000

# Terminal B
python -m uvicorn api.agent_service:app --host 0.0.0.0 --port 8001

# Terminal C
streamlit run dashboard/app.py
# → http://localhost:8501
```

Optional whole-stack container demo: `docker compose up` (see
`docs/Reproducibility Guide.md` §3.13).

## Layout

```
api/            Pydantic contracts and FastAPI services
agent/          ReAct agent (workflows, prompts, memory, reasoning, orchestration)
agent_eval/     Agent benchmark, judge, ablations
configs/        base/dev/test/prod yaml + Pydantic Settings
dashboard/      Streamlit operator UI (C7)
data/           raw / processed / splits (git-ignored, regenerated)
docker/         per-service Dockerfiles
docs/           data card, reproducibility guide, demo guide, file index
evaluation/     model evaluation (per-class metrics, confusion, error analysis)
inference/      ONNX runtime + benchmark
models/         time-series classifier architectures
orchestrator/   multi-node concurrent simulation, latency tests
quantization/   pruning, INT8 quantization, ONNX export
rag/            chunking, embedding, retrieval, reranking, prompting
reports/        model_eval.md, agent_eval.md, final_report.pdf, figures/
scripts/        evaluation / integration report CLIs, benchmarks, C7 demo
simulation/     PV/BESS physics simulators with fault injection
tests/          unit / integration / e2e (~284 cases)
tools/          retrieve_knowledge, get_system_history, estimate_rul, escalate
training/       trainer, losses, callbacks, train.py
utils/          logging_config, seeds, paths, timing
```

## Documentation & deliverables

| Resource | Description |
|----------|-------------|
| [`docs/README.md`](docs/README.md) | Index of all docs and `reports/` deliverables |
| [`docs/Reproducibility Guide.md`](docs/Reproducibility%20Guide.md) | C1–C8 mapping, full PowerShell reproduction, FAQ |
| [`docs/Document Interpretation.md`](docs/Document%20Interpretation.md) | Directory and source-file index |
| [`docs/Dashboard Demo Guide.md`](docs/Dashboard%20Demo%20Guide.md) | ~5 min Streamlit defence walkthrough |
| [`docs/data_card.md`](docs/data_card.md) | Component 1 data card |
| [`reports/final_report.md`](reports/final_report.md) / [`.pdf`](reports/final_report.pdf) | C8 academic report |
| [`reports/AgentPV_Final_Presentation.pptx`](reports/AgentPV_Final_Presentation.pptx) | Final Presentation slides |

Each major package also has its own `README.md` under `api/`, `agent/`, `dashboard/`, etc.

## Headline numbers (2026-06 full pipeline)

| Metric | Value |
|--------|-------|
| Dataset | 50 500 samples (28k PV + 22.5k BESS) |
| Unit tests | 284 passed |
| Agent judge mean | 4.10 (99/99 scored) |
| Full pipeline P95 | ~9.8 s (LLM-dominated) |
| Orchestrator run | 144 events / 10 nodes / 3 absorbed agent failures |
| C7 fault injection | 5 scenarios all pass (`fault_injection_demo.md`) |
