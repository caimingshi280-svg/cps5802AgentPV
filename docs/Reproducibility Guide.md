# AgentPV 复现指南（作业逐条对照 + 从零到全链路）

本文面向 **CPS 5802 / AgentPV** 提交与答辩，包含：

1. **作业条款 ↔ 仓库证据**对照表
2. **从零训练模型并跑通整个项目**的完整流程（PowerShell，Windows）
3. **终端分工、耗时、依赖关系、自检与常见问题**

更细的文件级说明见 [`Document Interpretation.md`](Document%20Interpretation.md)；模块设计见各子目录 `README.md`。  
答辩 Streamlit 步骤见 [`Dashboard Demo Guide.md`](Dashboard%20Demo%20Guide.md)。

> 下文默认仓库根目录为 `c:\Users\Mansycc\Desktop\omar`，请按你的实际路径修改 `cd` 命令。

---

## 0. 总览：你要跑的几大块

```text
环境安装
  → 生成数据（C1）
  → 训练 PV / BESS（C2）
  → 导出 ONNX + INT8（C2）
  → 模型评测（C3）+ 鲁棒性（扩展）
  → 知识库向量索引（RAG，首次建议）
  → 启动 Edge + Agent HTTP 服务（长期占 2 个终端）
  → 智能体评测 agent_eval（C5，需 Ollama）
  → 集成延迟 + 10 节点编排（C6）
  → 故障注入演示（C7）
  → 子报告生成 + 维护终稿 PDF / PPT（C8）
  → （可选）Streamlit 仪表盘、Docker Compose 录屏
```

### 0.1 建议的终端分工


| 终端    | 用途          | 典型命令                                        |
| ----- | ----------- | ------------------------------------------- |
| **A** | 边缘服务（长期运行）  | `uvicorn api.edge_service:app --port 8000`  |
| **B** | 智能体服务（长期运行） | `uvicorn api.agent_service:app --port 8001` |
| **C** | 训练、评测、脚本、编排 | 本文 §3 中其余所有命令                               |


**Ollama** 需在本机后台可用（非上述终端内）：`ollama serve`，且模型与 `configs/dev.yaml` 一致（默认 `llama3.2`）。

### 0.2 硬性依赖关系（必读）


| 若未完成的步骤                | 后果                                                       |
| ---------------------- | -------------------------------------------------------- |
| 未跑 §3.1 数据生成           | 无法训练                                                     |
| 未跑 §3.2 训练             | 无 `.pt` 权重                                               |
| 未跑 §3.3 ONNX/INT8      | `edge_service` 可能 **degraded** 或无法推理                     |
| 未起 §3.7 的 8000/8001 服务 | `agent_eval`、延迟基准、编排器、C7 演示均会失败                          |
| 未安装/未启动 Ollama         | `agent_eval --llm-backend ollama` 失败（可改用 `mock` 仅做连通性测试） |


---

## 1. 环境准备（第 0 步，必做）


| 项         | 要求                                                                                              |
| --------- | ----------------------------------------------------------------------------------------------- |
| Python    | **3.11 或 3.12**（`pyproject.toml`：`>=3.11,<3.13`）                                                |
| 安装依赖      | 仓库根目录：`pip install -e ".[dev]"`                                                                 |
| 可选 `.env` | 复制 `.env.example` → `.env`（使用云端 OpenAI 兼容 API 时填写）                                              |
| Ollama    | 真链路 `agent_eval` / `agent_service`（dev）：本机 `ollama serve`，且 `ollama pull llama3.2`（或与你配置一致的模型名） |
| Docker    | 仅 **§3.13** 整系统演示 / 录屏需要；安装 **Docker Desktop** 后终端能执行 `docker compose version`                  |
| 浏览器       | 答辩演示 Streamlit 看板（`http://localhost:8501`）                                                       |


### 1.1 安装与自检

```powershell
cd c:\Users\Mansycc\Desktop\omar

# 建议虚拟环境（可选但推荐）
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 安装项目 + 开发依赖
pip install -e ".[dev]"

# 可选：环境变量文件
# Copy-Item .env.example .env

# 自检：测试应全部通过（约 284 条用例）
$env:APP_ENV = "test"
pytest tests -q
```

### 1.2 Ollama 准备（跑 C5 / dev 智能体前）

```powershell
ollama list
# 若无 llama3.2：
ollama pull llama3.2

# 确认服务在跑（另开窗口或系统托盘 Docker Desktop 式常驻）
# ollama serve
```

配置位置：`configs/dev.yaml`（`llm_backend: ollama`）、`configs/settings.py`（`ollama_model` 等）。环境变量可覆盖：`AGENTPV_LLM_BACKEND`、`AGENTPV_OLLAMA_MODEL` 等。

---

## 2. 作业要求逐条对照（摘要）


| 作业组件 / 交付物             | 要求要点                                              | 本项目对应证据（路径或命令）                                                     |
| ---------------------- | ------------------------------------------------- | ------------------------------------------------------------------ |
| **C1 数据**              | ≥50k 样本；PV≥7 类、BESS≥5 类；≥3 种工况；train/val/test 可复现 | `simulation/`；`docs/data_card.md`；§3.1                             |
| **C2 模型**              | 多类；时序架构；≥1 种压缩；ONNX；≤50MB、CPU P95≤100ms           | `models/cnn1d.py`；`quantization/`；§3.2–3.3；`reports/model_eval.md` |
| **C3 评估**              | Macro-F1、混淆矩阵、P95、压缩权衡、≥2 变体                      | §3.4 `python -m evaluation --compare`                              |
| **C4 智能体**             | ReAct；≥30 文档 RAG；四工具；结构化输出                        | `agent/workflows/react.py`；`rag/`；`tools/`；§3.6–3.7                |
| **C5 评测**              | ≥30 场景；消融；评分；可选 LLM-as-judge                      | `agent_eval/`；§3.8                                                 |
| **C6 集成**              | ≥10 节点；E2E P95≤10s；Compose；三模式                    | `pv6_bess4`；§3.9；`docker-compose.yml`                              |
| **C7 原型**              | Web 仪表盘；交互演示                                      | `dashboard/`；§3.10–3.11；[`Dashboard Demo Guide.md`](Dashboard%20Demo%20Guide.md) |
| **交付物**                | Data Card、Dataset、Edge Model、各报告、Final Report     | `docs/`、`data/`、`quantization/artifacts/`、`reports/`               |
| **Final Presentation** | 现场演示与答辩                                           | 课堂完成，不在仓库命令内                                                       |


---

## 3. 全链路复现（详细步骤）

以下与 `reports/final_report.md` §11.3 一致，并补充**检查命令、耗时估计、快速试跑路径**。

**时间粗算（CPU、作业规模）**：


| 步骤                             | 大致耗时           |
| ------------------------------ | -------------- |
| §3.1 数据生成                      | 10～40 分钟       |
| §3.2 PV 训练                     | 1～4 小时         |
| §3.2 BESS 训练                   | 1～4 小时         |
| §3.3 ONNX + INT8               | 数分钟            |
| §3.4 评测                        | 10～30 分钟       |
| §3.5 鲁棒性                       | 20～60 分钟       |
| §3.6 RAG ingest（首次）            | 10+ 分钟（下载嵌入模型） |
| §3.8 agent_eval（3 消融 × Ollama） | 约 15～30 分钟     |
| §3.9 集成基准 + 编排                 | 数分钟～十几分钟       |


有 NVIDIA GPU 且已安装 CUDA 版 PyTorch 时，训练可将 `--device cpu` 改为 `--device cuda` 加速。

---

### 3.1 数据生成（C1）

**作业规模（约 5 万+ 样本）**：

```powershell
cd c:\Users\Mansycc\Desktop\omar
$env:APP_ENV = "dev"

python -m simulation.generate_dataset --seed 42 `
    --n-pv 28000 --n-bess 22500 `
    --n-pv-normal 8000 --n-bess-normal 5000
```

**检查是否成功**：

```powershell
Get-ChildItem data\processed
Get-ChildItem data\splits
Get-Content data\version.txt -Head 30
```

**快速试跑（非作业规模，仅验证训练管线，数分钟级）**：

```powershell
python -m simulation.generate_dataset --seed 0 --n-pv 800 --n-bess 600 --out-dir data\_smoke
```

后续训练需指定数据目录，例如：

```powershell
python -m training.train --system pv --epochs 3 --batch-size 64 `
    --processed-dir data\_smoke\processed --splits-dir data\_smoke\splits
```

---

### 3.2 训练两个分类器（C2）

先 **PV**，再 **BESS**。默认 **CPU**（与作业边缘约束一致）。

```powershell
cd c:\Users\Mansycc\Desktop\omar
$env:APP_ENV = "dev"

python -m training.train --system pv `
    --epochs 25 --early-stop-patience 6 --batch-size 256 --device cpu

python -m training.train --system bess `
    --epochs 30 --early-stop-patience 8 --batch-size 256 --device cpu
```

**检查权重**：

```powershell
Get-ChildItem quantization\artifacts\*.pt
# 期望：cnn1d_pv_best.pt、cnn1d_bess_best.pt（及训练过程 checkpoint）
```

**常用可选参数**（见 `training/train.py --help`）：

- `--lr`、`--dropout`、`--seed`
- `--processed-dir`、`--splits-dir`、`--artifacts-dir`

---

### 3.3 导出 ONNX + INT8 量化（C2）

```powershell
cd c:\Users\Mansycc\Desktop\omar

python -m quantization.onnx_export `
    --checkpoint quantization/artifacts/cnn1d_pv_best.pt `
    --output quantization/artifacts/cnn1d_pv.onnx

python -m quantization.onnx_export `
    --checkpoint quantization/artifacts/cnn1d_bess_best.pt `
    --output quantization/artifacts/cnn1d_bess.onnx

python -m quantization.int8_static --system pv
python -m quantization.int8_static --system bess
```

**检查**：

```powershell
Get-ChildItem quantization\artifacts\*.onnx
```

Edge 服务默认加载此目录下的 ONNX（见 `api/edge_service.py`、`configs/base.yaml`）。

---

### 3.4 模型评测对比（C3）

```powershell
cd c:\Users\Mansycc\Desktop\omar

python -m evaluation --compare
```

**产出**：`reports/pv/`、`reports/bess/` 下各变体 `summary.md` / `summary.json`、`comparison.md`，以及手写总览 `reports/model_eval.md`（索引子报告链接与关键数字）。

**其它用法**（见 `evaluation/__main__.py`）：

```powershell
python -m evaluation --systems pv
python -m evaluation --variants onnx_fp32
python -m evaluation --split val
```

---

### 3.5 鲁棒性评估（扩展，导师建议，建议纳入完整复现）

```powershell
cd c:\Users\Mansycc\Desktop\omar

python scripts/run_robustness_eval.py
```

**产出**：`reports/robustness_eval.md`、`reports/robustness/{pv,bess}/` 下图表与 JSON。

---

### 3.6 知识库向量索引（首次起 Agent 前建议执行）

领域文档已在 `rag/knowledge_base/documents/`（≥30 篇）。构建 Chroma 持久化索引：

```powershell
cd c:\Users\Mansycc\Desktop\omar
$env:APP_ENV = "dev"

python -m rag.ingest
```

重建索引（清空后重写）：

```powershell
python -m rag.ingest --reset
```

首次运行会下载 `sentence-transformers` 嵌入模型，需网络，可能较慢。

---

### 3.7 启动 Edge + Agent 服务（§3.8 及之后的前置条件）

以下两个命令需**分别占用终端 A、B**，并保持运行。

**终端 A — 边缘推理（端口 8000）**：

```powershell
cd c:\Users\Mansycc\Desktop\omar
$env:APP_ENV = "dev"
python -m uvicorn api.edge_service:app --host 0.0.0.0 --port 8000
```

**终端 B — 云端智能体（端口 8001，dev 下默认 Ollama）**：

```powershell
cd c:\Users\Mansycc\Desktop\omar
$env:APP_ENV = "dev"
python -m uvicorn api.agent_service:app --host 0.0.0.0 --port 8001
```

**终端 C — 健康检查**：

```powershell
Invoke-WebRequest -Uri http://127.0.0.1:8000/healthz -UseBasicParsing
Invoke-WebRequest -Uri http://127.0.0.1:8001/healthz -UseBasicParsing
```

浏览器可打开 OpenAPI 文档：

- Edge：`http://127.0.0.1:8000/docs`
- Agent：`http://127.0.0.1:8001/docs`

---

### 3.8 智能体评测（C5，真 Ollama）

**前置**：终端 A/B 服务已启动；Ollama 已 `pull` 对应模型。

**推荐（与 `reports/agent_eval.md` / `final_report.pdf` 一致，含 LLM-as-judge）**：

```powershell
cd c:\Users\Mansycc\Desktop\omar
$env:APP_ENV = "dev"
$env:AGENTPV_JUDGE_API_BASE = "http://127.0.0.1:11434/v1"
$env:AGENTPV_JUDGE_MODEL = "llama3.2:latest"

python -m agent_eval `
    --ablations full no_retrieve_knowledge no_reasoning_trace `
    --llm-backend ollama `
    --out-json agent_eval/results/last_run_three_ablations_with_judge.json `
    --out-md   reports/agent_eval_last_run_with_judge.md
```

生成报告与图表：

```powershell
python scripts/render_agent_eval_report.py `
    --input agent_eval/results/last_run_three_ablations_with_judge.json `
    --log   agent_eval/results/last_run_three_ablations_with_judge.log
```

**产出**：`reports/agent_eval.md`、`reports/figures/agent_eval/*.png`、`reports/agent_eval_artifact_meta.json`（judge mean **4.10** / 99）。

**仅启发式 rubric（跳过 judge，更快）**：在上式中加 `--no-llm-judge`，输出改为 `agent_eval/results/last_run_three_ablations.json`。

**仅 CI / 离线快速验证（不调 Ollama）**：

```powershell
python -m agent_eval --ablations full --llm-backend mock --no-llm-judge
```

#### LLM-as-judge 说明（1–5 分）

作业允许启发式 rubric **或** LLM-as-judge；后者为加分项，非硬性必须。

- **本机 Ollama**：**无需** OpenAI API key。设置 `AGENTPV_JUDGE_API_BASE` / `AGENTPV_JUDGE_MODEL`（见上方推荐命令），**不要**加 `--no-llm-judge`。
- **云端 OpenAI 兼容接口**：设置 `AGENTPV_JUDGE_API_KEY` 与 `AGENTPV_JUDGE_MODEL`。

详见 `agent_eval/README.md`、`agent_eval/llm_judge.py`。

---

### 3.9 集成评测 + 10 节点编排（C6）

**前置**：§3.7 中 Edge（8000）与 Agent（8001）仍在运行。

**终端 C**：

```powershell
cd c:\Users\Mansycc\Desktop\omar
New-Item -ItemType Directory -Force reports\integration | Out-Null

python scripts/e2e_latency_bench.py --mode edge_only   --iterations 50 --warmup 3 --out-json reports/integration/e2e_latency_edge_only.json
python scripts/e2e_latency_bench.py --mode full        --iterations 50 --warmup 3 --out-json reports/integration/e2e_latency_full.json
python scripts/e2e_latency_bench.py --mode cloud_only  --iterations 50 --warmup 3 --out-json reports/integration/e2e_latency_cloud_only.json

python -m orchestrator --nodes pv6_bess4 --duration 60 --http-timeout 120 --out data/orchestrator/events.jsonl

python scripts/render_integration_eval_report.py
```

**产出**：`reports/integration_eval.md`、`reports/integration_eval_meta.json`、相关图表。

**编排器其它节点集**（`orchestrator/__main__.py`）：

- `minimal` — 最少节点  
- `pv2_bess1` — 默认小规模  
- `pv6_bess4` — **10 节点**（作业 C6）

可选参数：`--edge`、`--agent`、`--duration`、`--http-timeout`。

---

### 3.10 交互故障注入演示（C7）

**前置**：Edge + Agent 服务已启动。

```powershell
cd c:\Users\Mansycc\Desktop\omar

python scripts/demo_fault_injection.py --events-path data/orchestrator/events_c7_demo.jsonl
```

**产出**：`reports/integration/fault_injection_demo.md`、`reports/integration/fault_injection_demo.json`。

Streamlit 侧栏注入与上述共用 `dashboard/inject.py`，见 §3.11。

---

### 3.11 Streamlit 操作员仪表盘（可选，答辩演示）

**前置**：建议已跑过编排器或已有 `data/orchestrator/events.jsonl`；Edge/Agent 用于侧栏一键注入。

**新终端**：

```powershell
cd c:\Users\Mansycc\Desktop\omar
$env:APP_ENV = "dev"
streamlit run dashboard/app.py
```

浏览器访问：**[http://localhost:8501](http://localhost:8501)**

**答辩演示**：完整步骤与 5 分钟讲稿见 [`Dashboard Demo Guide.md`](Dashboard%20Demo%20Guide.md)。

---

### 3.12 终稿报告与答辩幻灯片（C8 / Presentation）

| 交付物 | 路径 |
|--------|------|
| 终稿 Markdown 源稿 | `reports/final_report.md` |
| 终稿 PDF | `reports/final_report.pdf` |
| 答辩 PPT | `reports/AgentPV_Final_Presentation.pptx` |

修改子报告数字后，请手工同步 `reports/final_report.md`，并在 Word / 浏览器中重新导出 PDF 与 PPTX。

---

### 3.13 Docker 整系统（作业「docker compose up」，可选录屏）

**前置**：已安装 Docker Desktop；终端执行 `docker compose version` 有版本号。  
需已具备 `quantization/artifacts/` 下 ONNX（构建镜像时挂载只读）。

```powershell
cd c:\Users\Mansycc\Desktop\omar
docker compose up --build
```

**检查**（另开终端或浏览器）：


| URL                                                            | 说明        |
| -------------------------------------------------------------- | --------- |
| [http://localhost:8501](http://localhost:8501)                 | Dashboard |
| [http://localhost:8000/healthz](http://localhost:8000/healthz) | Edge      |
| [http://localhost:8001/healthz](http://localhost:8001/healthz) | Agent     |


录屏检查：确认 §3.13 表中 8501 / 8000 / 8001 health 均为 200，Dashboard 能打开事件时间线。

结束：

```powershell
docker compose down
```

> `vector-db` 服务在 `docker-compose.yml` 中带 `profiles: ["polish"]`，默认 `docker compose up` **不会**启动；当前 agent 使用文档目录 + Chroma/嵌入，与作业要求的 edge + agent 最小集一致。

---

## 4. 全流程命令清单（复制版）

在**已完成 §1 环境安装**后，可按顺序复制执行。  
**注意**：§3.7 起需先手动在终端 A/B 启动两个 uvicorn 服务（见 §3.7），本清单中用注释标出。

```powershell
# ========== 0 环境（若已做过可跳过）==========
cd c:\Users\Mansycc\Desktop\omar
pip install -e ".[dev]"
$env:APP_ENV = "test"
pytest tests -q

# ========== 1 数据（C1）==========
$env:APP_ENV = "dev"
python -m simulation.generate_dataset --seed 42 --n-pv 28000 --n-bess 22500 --n-pv-normal 8000 --n-bess-normal 5000

# ========== 2 训练（C2）==========
python -m training.train --system pv   --epochs 25 --early-stop-patience 6 --batch-size 256 --device cpu
python -m training.train --system bess --epochs 30 --early-stop-patience 8 --batch-size 256 --device cpu

# ========== 3 ONNX + INT8（C2）==========
python -m quantization.onnx_export --checkpoint quantization/artifacts/cnn1d_pv_best.pt   --output quantization/artifacts/cnn1d_pv.onnx
python -m quantization.onnx_export --checkpoint quantization/artifacts/cnn1d_bess_best.pt --output quantization/artifacts/cnn1d_bess.onnx
python -m quantization.int8_static --system pv
python -m quantization.int8_static --system bess

# ========== 4 评测（C3）==========
python -m evaluation --compare

# ========== 5 鲁棒性（扩展）==========
python scripts/run_robustness_eval.py

# ========== 6 RAG 索引 ==========
python -m rag.ingest

# ========== 7 起服务（终端 A / B 分别执行，保持运行）==========
# 终端A: cd c:\Users\Mansycc\Desktop\omar; $env:APP_ENV='dev'; python -m uvicorn api.edge_service:app --host 0.0.0.0 --port 8000
# 终端B: cd c:\Users\Mansycc\Desktop\omar; $env:APP_ENV='dev'; python -m uvicorn api.agent_service:app --host 0.0.0.0 --port 8001

# ========== 8 智能体评测（C5，终端 C；Ollama 已启动）==========
$env:AGENTPV_JUDGE_API_BASE='http://127.0.0.1:11434/v1'
$env:AGENTPV_JUDGE_MODEL='llama3.2:latest'
python -m agent_eval --ablations full no_retrieve_knowledge no_reasoning_trace --llm-backend ollama --out-json agent_eval/results/last_run_three_ablations_with_judge.json --out-md reports/agent_eval_last_run_with_judge.md
python scripts/render_agent_eval_report.py --input agent_eval/results/last_run_three_ablations_with_judge.json --log agent_eval/results/last_run_three_ablations_with_judge.log

# ========== 9 集成（C6，终端 C；A/B 仍在跑）==========
New-Item -ItemType Directory -Force reports\integration | Out-Null
python scripts/e2e_latency_bench.py --mode edge_only  --iterations 50 --warmup 3 --out-json reports/integration/e2e_latency_edge_only.json
python scripts/e2e_latency_bench.py --mode full       --iterations 50 --warmup 3 --out-json reports/integration/e2e_latency_full.json
python scripts/e2e_latency_bench.py --mode cloud_only --iterations 50 --warmup 3 --out-json reports/integration/e2e_latency_cloud_only.json
python -m orchestrator --nodes pv6_bess4 --duration 60 --http-timeout 120 --out data/orchestrator/events.jsonl
python scripts/render_integration_eval_report.py

# ========== 10 C7 故障注入演示 ==========
python scripts/demo_fault_injection.py --events-path data/orchestrator/events_c7_demo.jsonl

# ========== 11 可选 ==========
# streamlit run dashboard/app.py
# docker compose up --build
```

---

## 5. 复现完成自检

### 5.1 自动化测试

```powershell
cd c:\Users\Mansycc\Desktop\omar
$env:APP_ENV = "test"
pytest tests -q
```

**主标准**：全部通过。全仓库 `ruff` 可能存在历史告警，不必作为首次复现阻塞项。

### 5.2 关键产物是否存在

```powershell
Test-Path quantization\artifacts\cnn1d_pv_best.pt
Test-Path quantization\artifacts\cnn1d_pv.onnx
Test-Path data\processed
Test-Path reports\model_eval.md
Test-Path reports\agent_eval.md
Test-Path reports\integration_eval.md
Test-Path reports\final_report.md
Test-Path reports\final_report.pdf
Test-Path reports\AgentPV_Final_Presentation.pptx
Test-Path reports\integration\fault_injection_demo.md
```

### 5.3 服务连通性（手动）

在 Edge/Agent 启动后：

```powershell
(Invoke-WebRequest http://127.0.0.1:8000/healthz -UseBasicParsing).Content
(Invoke-WebRequest http://127.0.0.1:8001/healthz -UseBasicParsing).Content
```

---

## 6. 常见问题


| 现象                            | 处理                                                                      |
| ----------------------------- | ----------------------------------------------------------------------- |
| `data/` 或 `artifacts/` 缺失     | 按 §3.1–3.3 顺序生成；目录内应有 `.gitkeep`，勿删空文件夹结构                               |
| 训练报找不到 npz/csv                | 确认 `--processed-dir` / `--splits-dir` 与数据生成输出一致                         |
| Edge 启动但 `/predict` 503       | ONNX 未导出或文件名不匹配；检查 `quantization/artifacts/`                            |
| Ollama 连接失败                   | `ollama serve`；`ollama list`；核对 `configs/dev.yaml` 与 `AGENTPV_OLLAMA_*` |
| `agent_eval` 极慢或超时            | 正常（真 LLM）；可先用 `--llm-backend mock` 验证管线                                 |
| `agent_recommend_failed` / 超时 | Agent 默认 HTTP 超时 10s；见 `reports/integration_eval.md` 说明                 |
| 终稿 PDF 无法保存                   | 关闭占用该文件的 PDF 阅读器后，在 Word / 浏览器中重新导出并覆盖 `reports/` 下交付文件              |
| Docker 找不到命令                  | 安装 Docker Desktop，**重启终端**；见本文 §3.13                                    |
| `rag.ingest` 很慢               | 首次下载嵌入模型；确保网络可达 Hugging Face                                            |
| PowerShell 续行符                | 多行命令末尾使用反引号 ```（反引号前有空格）                                                |


---

## 7. 与仓库其他文档的关系


| 文档 | 作用 |
|------|------|
| `docs/README.md` | 文档与交付物总索引 |
| `Document Interpretation.md` | 每个目录/主要源文件用途 |
| `Dashboard Demo Guide.md` | Streamlit 答辩演示 |
| `data_card.md` | Component 1 数据卡片 |
| `README.md`（根目录） | 项目总览、架构 |
| 各子包 `README.md` | 模块级设计 |
| `scripts/README.md` | 评测脚本入口与顺序 |
| `CPS-5802-Project-SP26.pdf` | 课程作业原文 |


---

## 8. 学习路径提示（非复现必需）

若目标是**理解代码**而非仅跑通命令，建议按以下顺序阅读（与复现顺序大体一致）：

1. `docs/alert_schema.json` + `api/schemas.py`
2. `simulation/` → `training/` → `quantization/` + `inference/`
3. `evaluation/` → `api/edge_service.py`
4. `rag/` + `tools/` + `agent/workflows/react.py` → `api/agent_service.py`
5. `agent_eval/` → `orchestrator/` + `dashboard/`

可与本文 **§3** 交替进行：每读完一块，跑对应 § 的小节命令加深印象。