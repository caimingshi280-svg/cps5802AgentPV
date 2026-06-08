# AgentPV 项目文件解读

本文说明仓库内**主要路径与源文件**的用途，便于对照代码学习。  
各**业务子包**（`api/`、`agent/`、`simulation/` 等）另有独立 **`README.md`**，与本文件互补，学习时建议「先读子包 README，再对照本索引定位源文件」。  
**不包含**：`.venv/`、`__pycache__/`、`.pytest_cache/` 等工具缓存；`data/processed/`、`quantization/artifacts/`、`reports/figures/` 下大量生成物仅列目录用途，不逐一枚举二进制文件。

---

## 1. 根目录（顶层）

| 文件 / 目录 | 用途 |
|-------------|------|
| `README.md` | 项目总览、架构、快速开始、目录索引。 |
| `pyproject.toml` | Python 包元数据、依赖、可编辑安装与 pytest/ruff 配置。 |
| `requirements.txt` | 聚合依赖（指向 `requirements/*.txt`）。 |
| `docker-compose.yml` | 多服务编排（可选；答辩推荐本机 uvicorn）。 |
| `.env.example` | 环境变量模板（复制为 `.env` 后填写密钥等）。 |
| `.gitignore` | Git 忽略规则（数据、产物、虚拟环境、安装包等）。 |
| `projectdesignrules.cursorrules` | 协作与工程规则（Cursor / 团队约定）。 |
| `docs/` | **中文文档、答辩材料、Data Card**（见 `docs/README.md`）。 |
| `reports/` | 评测报告、终稿 PDF/PPTX、混淆矩阵等交付物。 |
| `scripts/` | 一键评测、子报告渲染、压测与演示等 CLI。 |

### 1.1 `docs/` — 说明性文档

| 文件 | 用途 |
|------|------|
| `Reproducibility Guide.md` | 作业逐条对照 + 完整复现步骤 |
| `Document Interpretation.md` | **本文件** |
| `Dashboard Demo Guide.md` | Streamlit 答辩现场演示（约 5 分钟） |
| `data_card.md` | Component 1 数据卡片 |
| `alert_schema.json` | 边缘 → 云端告警 JSON Schema |
| `AgentPV-项目方案.md` | 项目工程方案与模块规划 |
| `CPS-5802-Project-SP26.pdf` | 课程作业原文 |
| `README.md` | 本目录索引 |

---

## 2. 应用包（按目录）

### 2.1 `api/` — HTTP 契约与服务

| 文件 | 用途 |
|------|------|
| `schemas.py` | Pydantic 模型：告警 JSON、推荐结果、枚举等**全系统共享契约**。 |
| `errors.py` | 统一错误体与 HTTP 异常封装。 |
| `edge_service.py` | FastAPI：边缘推理服务（加载 ONNX、输出结构化告警）。 |
| `agent_service.py` | FastAPI：云端智能体服务（ReAct + RAG + 工具）。 |
| `__init__.py` | 包标识。 |

### 2.2 `agent/` — ReAct 智能体

| 文件 / 子目录 | 用途 |
|---------------|------|
| `workflows/react.py` | **核心**：Observe→Reason→Act→Reflect→Report 循环与工具调度。 |
| `orchestration/llm_client.py` | LLM 客户端抽象（计划 / 合成等）。 |
| `orchestration/remote_llm.py` | 远程 / Ollama 等具体后端适配。 |
| `README.md` | 本模块设计说明。 |

### 2.3 `agent_eval/` — 智能体评测（作业 Component 5）

| 文件 | 用途 |
|------|------|
| `benchmark.json` | ≥30 条基准场景（JSON，含模糊场景标签）。 |
| `scenarios.py` | 场景加载、默认 benchmark 生成与校验。 |
| `heuristic_rubric.py` | 启发式四维打分（紧迫性、关键词、禁忌、知识来源）。 |
| `llm_judge.py` | 可选 LLM-as-judge（OpenAI 兼容；本机 Ollama 可无 API key）。 |
| `wiring.py` | 组装评测用 Agent（消融时禁用指定工具）。 |
| `runner.py` | 异步跑全量场景 × 消融、汇总、写 JSON/Markdown。 |
| `__main__.py` | CLI：`python -m agent_eval`。 |
| `README.md` | 评测流程、环境变量、性能说明。 |

### 2.4 `configs/` — 配置

| 文件 | 用途 |
|------|------|
| `base.yaml` / `dev.yaml` / `test.yaml` / `prod.yaml` | 分环境默认参数。 |
| `settings.py` | Pydantic Settings：环境变量 > YAML > 默认值。 |
| `README.md` | 配置加载顺序与各文件说明。 |

### 2.5 `dashboard/` — 操作员界面

| 文件 | 用途 |
|------|------|
| `app.py` | Streamlit 主应用：告警、节点、推理轨迹展示。 |
| `data.py` | 从 orchestrator JSONL 等读取展示数据。 |
| `inject.py` | **C7**：一键故障注入，调用 edge/agent 并写事件。 |
| `README.md` | 本地运行与 Docker 说明。 |

### 2.6 `evaluation/` — 模型评测（作业 Component 3）

| 文件 | 用途 |
|------|------|
| `metrics.py` / `classification_report.py` | 分类指标与报表。 |
| `confusion_matrix.py` | 混淆矩阵绘图。 |
| `latency_benchmark.py` / `model_size.py` | 延迟与体积测量。 |
| `pytorch_runner.py` / `predictor.py` | PyTorch / 通用预测封装。 |
| `compare_variants.py` | 多后端变体对比。 |
| `runner.py` / `__main__.py` | 评测入口：`python -m evaluation`。 |
| `robustness.py` / `figures.py` | 鲁棒性评估与统一绘图风格。 |
| `README.md` | 评测管线说明。 |

### 2.7 `inference/` — 推理运行时

| 文件 | 用途 |
|------|------|
| `onnx_runner.py` | ONNX Runtime CPU 推理封装。 |
| `postprocess.py` | 后处理与严重度映射等。 |
| `README.md` | 推理与 benchmark 说明。 |

### 2.8 `models/` — 网络结构

| 文件 | 用途 |
|------|------|
| `base.py` | 模型基类或共享接口。 |
| `cnn1d.py` | **1D CNN** 时间序列分类器（PV/BESS）。 |
| `README.md` | 结构选型说明。 |

### 2.9 `orchestrator/` — 多节点仿真与集成驱动

| 文件 | 用途 |
|------|------|
| `__main__.py` | CLI：节点目录 `minimal` / `pv2_bess1` / `pv6_bess4`、输出 JSONL。 |
| `orchestrator.py` | 调度多节点、调用 edge/agent 客户端。 |
| `node_simulator.py` | 单节点传感器窗口与故障模拟。 |
| `clients.py` | HTTP 客户端（edge / agent）。 |
| `event_log.py` | 事件持久化。 |
| `README.md` | 拓扑与用法。 |

### 2.10 `quantization/` — 压缩与导出

| 文件 | 用途 |
|------|------|
| `onnx_export.py` | PyTorch → ONNX。 |
| `int8_static.py` | 静态 INT8 量化管线。 |
| `README.md` | 产物路径与约束。 |

### 2.11 `rag/` — 检索增强

| 文件 | 用途 |
|------|------|
| `chunking.py` / `embedding.py` / `retrieval.py` / `reranking.py` | 切块、嵌入、检索、重排。 |
| `chroma_retrieval.py` | Chroma 向量检索实现。 |
| `ingest.py` | 知识库入库脚本逻辑。 |
| `prompting.py` | 提示模板。 |
| `knowledge_base/documents/*.md` | **≥30** 篇领域 Markdown 文档。 |
| `knowledge_base/chroma_db/` | 向量库持久化目录（通常 gitignore）。 |
| `README.md` | RAG 数据流说明。 |

### 2.12 `simulation/` — 数据生成（作业 Component 1）

| 文件 | 用途 |
|------|------|
| `pv_simulator.py` / `battery_simulator.py` | 光伏 / 电池时间序列仿真。 |
| `fault_injector.py` | 故障注入与标签。 |
| `generate_dataset.py` | **主入口**：生成 parquet、划分、可选 `version_path`。 |
| `README.md` | 类别、规模、随机种子说明。 |

### 2.13 `tools/` — 智能体工具

| 文件 | 用途 |
|------|------|
| `base.py` | 工具基类 / 注册约定。 |
| `retrieve_knowledge.py` | RAG 检索。 |
| `system_history.py` | 系统历史（mock / 简化实现）。 |
| `estimate_rul.py` | RUL 估计工具。 |
| `escalate_alert.py` | 告警升级工具。 |
| `README.md` | 工具列表与契约。 |

### 2.14 `training/` — 训练

| 文件 | 用途 |
|------|------|
| `train.py` | CLI：`python -m training.train`。 |
| `trainer.py` / `data.py` / `losses.py` | 训练循环、数据加载、损失。 |
| `README.md` | 超参与输出路径。 |

### 2.15 `utils/` — 通用工具

| 文件 | 用途 |
|------|------|
| `paths.py` | 项目根路径、数据/报告等常量（禁止硬编码散落路径）。 |
| `logging_config.py` | 日志格式与级别。 |
| `seeds.py` | 随机种子固定。 |
| `timing.py` | 计时辅助。 |
| `README.md` | 各工具模块说明。 |

---

## 3. `scripts/` — 一次性脚本

| 文件 | 用途 |
|------|------|
| `run_robustness_eval.py` | 鲁棒性 / OOD / selective prediction 主评估。 |
| `render_agent_eval_report.py` | 从 agent_eval JSON 生成 `reports/agent_eval.md` 与图表。 |
| `render_integration_eval_report.py` | C6 集成评估报告与图。 |
| `e2e_latency_bench.py` | 端到端延迟基准（full / edge_only / cloud_only）。 |
| `demo_fault_injection.py` | C7 故障注入演示与报告。 |
| `bootstrap_kb_documents.py` | 知识库文档 bootstrap。 |
| `run_dev_first_artifacts.py` | 开发期首批产物生成辅助。 |
| `_count_agent_eval_signals.py` | 解析日志统计遥测（含 UTF-16 日志兼容）。 |
| `README.md` | 各脚本调用顺序与依赖。 |

---

## 4. `tests/` — 测试

| 路径 | 用途 |
|------|------|
| `conftest.py` | pytest 公共 fixture（如 `APP_ENV=test`）。 |
| `unit/` | 单元测试：模型、评测、工具、API 契约、编排器等。 |
| `integration/` | 集成测试：edge/agent 服务（需环境时跳过或 mock）。 |
| `README.md` | 如何运行测试与注意事项。 |

---

## 5. `docker/` — 容器构建

| 文件 | 用途 |
|------|------|
| `edge.Dockerfile` | 边缘服务镜像。 |
| `agent.Dockerfile` | 智能体服务镜像。 |
| `dashboard.Dockerfile` | Streamlit 仪表盘镜像。 |
| `orchestrator.Dockerfile` | 多节点驱动镜像。 |

---

## 6. `reports/` — 报告与图表（生成物 / 终稿）

| 类型 | 用途 |
|------|------|
| `model_eval.md` | 模型与评测总览、子报告链接。 |
| `agent_eval.md` / `integration_eval.md` / `robustness_eval.md` 等 | 各子系统评估正文。 |
| `final_report.md` / `final_report.pdf` | C8 期末技术报告。 |
| `AgentPV_Final_Presentation.pptx` | Final Presentation 答辩幻灯片。 |
| `integration/fault_injection_demo.md` | C7 五场景脚本化报告。 |
| `figures/**` | 各报告配套 PNG（matplotlib 导出）。 |

---

## 7. `data/` — 数据目录（多数被 gitignore）

| 子目录 / 文件 | 用途 |
|---------------|------|
| `raw/` / `processed/` / `splits/` | 原始 / 处理后 / 划分数据（由 `simulation.generate_dataset` 等生成）。 |
| `orchestrator/` | 编排器输出的 `events.jsonl` 等，供 Dashboard 读取。 |
| `version.txt` | 数据集元数据与版本信息（生成脚本写入）。 |

---

## 8. 未发现可安全删除的「无用源码」

当前 **99 个 `.py` 文件**均有测试引用或明确入口；`scripts/` 下带下划线脚本为工具用途。若后续出现一次性迁移脚本，删除前请再全局搜索引用。

---

## 9. 延伸阅读

- 作业对照与一键复现：见 [`Reproducibility Guide.md`](Reproducibility%20Guide.md)。  
- 答辩 Streamlit 演示：见 [`Dashboard Demo Guide.md`](Dashboard%20Demo%20Guide.md)。  
- 工程规则：见 `projectdesignrules.cursorrules` 与根目录 `README.md`。
