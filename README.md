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

cp .env.example .env             # then fill in DEEPSEEK_API_KEY
pytest -q                         # run smoke tests on the contracts
```

## Quickstart (full system)

> Requires the components implemented per `AgentPV-项目方案.md` §5.

```bash
docker compose up
# dashboard:      http://localhost:8501
# edge service:   http://localhost:8000/docs
# agent service:  http://localhost:8001/docs
# vector db:      http://localhost:8002
```

## Layout

```
api/            Pydantic contracts and FastAPI services
agent/          ReAct agent (workflows, prompts, memory, reasoning, orchestration)
agent_eval/     Agent benchmark, judge, ablations
configs/        base/dev/test/prod yaml + Pydantic Settings
dashboard/      Streamlit operator UI
data/           raw / processed / splits (git-ignored, regenerated)
docker/         per-service Dockerfiles
docs/           data card, architecture, alert schema, API docs
evaluation/     model evaluation (per-class metrics, confusion, error analysis)
inference/      ONNX runtime + benchmark
models/         time-series classifier architectures
orchestrator/   multi-node concurrent simulation, latency tests
quantization/   pruning, INT8 quantization, ONNX export
rag/            chunking, embedding, retrieval, reranking, prompting
reports/        model_eval.md, agent_eval.md, final_report.pdf, figures/
scripts/        one-shot CLI helpers
simulation/     PV/BESS physics simulators with fault injection
tests/          unit / integration / e2e
tools/          retrieve_knowledge, get_system_history, estimate_rul, escalate
training/       trainer, losses, callbacks, train.py
utils/          logging_config, seeds, paths, timing
```

## 中文文档（作业 / 复现 / 索引）

| 文档 | 说明 |
|------|------|
| [`docs/复现指南.md`](docs/复现指南.md) | 作业条款 ↔ 仓库对照 + 推荐复现命令（答辩前自检）。 |
| [`docs/ppt制作指南.md`](docs/ppt制作指南.md) | 答辩 PPT 逐页英文内容 + 中英详细讲稿。 |
| [`docs/ppt旁白.md`](docs/ppt旁白.md) | 仅旁白稿（主汇报 Slide 1–30）。 |
| [`docs/Q&A.md`](docs/Q&A.md) | 答辩 Q&A 手册（中英 + 对应幻灯片页码）。 |
| [`reports/AgentPV_Final_Presentation.pptx`](reports/AgentPV_Final_Presentation.pptx) | 由 `python scripts/render_presentation.py` 生成（需 `python-pptx`）。 |
| [`docs/网页演示指南.md`](docs/网页演示指南.md) | Streamlit 现场演示流程与降级讲法。 |
| [`docs/文件解读.md`](docs/文件解读.md) | 各目录与主要源文件用途。 |
| [`docs/开发记录.md`](docs/开发记录.md) | 分 Session 开发日志。 |
| [`docs/data_card.md`](docs/data_card.md) | Component 1 数据卡片。 |

Each major package also has its own `README.md` (see table under `Layout`).
