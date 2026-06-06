# AgentPV 项目完整方案计划表

> CPS 5802 — Machine Learning and Innovations · Spring 2026
> 课程项目：**AgentPV — 光伏与储能系统的智能体 AI 监控系统**
> 本文档是从 0 到交付的完整执行方案，覆盖架构设计、技术选型、目录结构、每个组件的具体做法、时间表、分工、风险点与验收标准。

---

## 目录

1. [项目目标与最终形态](#一项目目标与最终形态)
2. [技术栈总选型](#二技术栈总选型)
3. [仓库目录结构](#三仓库目录结构)
4. [作业要求 → 交付物对照表](#四作业要求--交付物对照表)
5. [七大组件详细执行方案](#五七大组件详细执行方案)
   - [组件 1：数据生成](#组件-1数据生成component-1--data-generation)
   - [组件 2：模型构建](#组件-2模型构建component-2--model-build)
   - [组件 3：模型评估](#组件-3模型评估component-3--model-evaluation)
   - [组件 4：LLM Agent 构建](#组件-4llm-agent-构建component-4--llm-agent-build)
   - [组件 5：LLM Agent 评估](#组件-5llm-agent-评估component-5--llm-agent-evaluation)
   - [组件 6：系统集成](#组件-6系统集成component-6--decision-making--integration)
   - [组件 7：原型与演示](#组件-7原型与演示component-7--prototype--demo)
6. [团队分工建议（4 人）](#六团队分工建议4-人)
7. [12 周时间表](#七12-周时间表)
8. [风险点与对策](#八风险点与对策)
9. [验收 Checklist（自检表）](#九验收-checklist自检表)
10. [最终报告写作指南](#十最终报告写作指南)
11. [答辩准备](#十一答辩准备)
12. [工程规则总览（projectdesignrules 对齐）](#十二工程规则总览projectdesignrules-对齐)

---

> **★ 阅读顺序提示**：本方案与 `projectdesignrules.cursorrules` 严格对齐。**第十二节是工程纪律的浓缩版**，写代码前请先看完它，再回头看本方案的目录结构（第三节）和各组件实现细节。

---

## 一、项目目标与最终形态

### 1.1 一句话定位

构建一个**三层云-边架构的光伏 / 储能智能监控系统**：仿真生成数据 → 边缘 AI 实时分类故障 → 云端 LLM Agent 检索知识、推理并给出可执行的运维建议 → Web 仪表盘展示全流程。

### 1.2 最终交付形态（验收时长这样）

```
$ git clone <repo>
$ cd agentpv
$ docker compose up
# 浏览器打开 http://localhost:8501
# - 仪表盘显示 10 个模拟节点实时运行
# - 用户点击"注入故障 → 部分阴影" → PV 节点 #3 进入告警
# - 边缘模型 100 ms 内分类完成，发出 JSON 告警
# - 云端 Agent 5 秒内返回结构化建议（含推理链、知识来源、置信度）
# - 全流程端到端延迟 < 10 秒
```

### 1.3 核心硬性指标（作业明文要求）

| 指标 | 阈值 |
|---|---|
| 数据集规模 | ≥ 50,000 样本 |
| PV 故障类别 | ≥ 7 类 |
| 电池故障类别 | ≥ 5 类 |
| 模型宏平均 F1 | ≥ 90% |
| 模型大小 | ≤ 50 MB |
| CPU 推理延迟 | ≤ 100 ms |
| 知识库文档 | ≥ 30 篇 |
| Agent 评估场景 | ≥ 30 个 |
| Agent 平均得分 | ≥ 4.0 / 5.0 |
| 端到端 P95 延迟 | ≤ 10 秒 |
| 并发节点 | ≥ 10 个 |

---

## 二、技术栈总选型

每一项都做了"为什么选它"的取舍，避免用陌生工具浪费时间。

| 层 | 任务 | 选型 | 理由 |
|---|---|---|---|
| 仿真 | PV 物理建模 | `pvlib-python` | 学术黄金标准，老师推荐 |
| 仿真 | 电池建模 | 自写 RC 等效电路（ECM） | 实现简单、参数易得 |
| 数据 | 处理 | `numpy` + `pandas` | 标配 |
| 模型 | 训练框架 | **PyTorch 2.x** | 量化、ONNX 导出工具链最成熟 |
| 模型 | 架构 | **1D CNN**（首选）+ 备选 LSTM | 1D CNN 训得快、压缩友好、延迟低 |
| 模型 | 压缩 | **INT8 静态量化** + 结构化剪枝 | 直接达成 50MB / 100ms 双指标 |
| 模型 | 部署 | `onnxruntime` (CPU only) | 作业明文要求 |
| RAG | 向量库 | **ChromaDB** | 零配置、本地持久化 |
| RAG | Embedding | `bge-small-zh` 或 `all-MiniLM-L6-v2` | 体积小、效果好、可本地跑 |
| Agent | 框架 | **LangChain** + 自写 ReAct 循环 | 工具调用 / RAG 一体化 |
| LLM | 主力 | **DeepSeek-Chat API** （免费/低价）| 中文好、便宜、稳定 |
| LLM | 备选 | Ollama + `qwen2.5:7b` | 离线兜底，避免 API 失效 |
| Web | 仪表盘 | **Streamlit** | 1 周内出原型最快 |
| 通信 | 边缘 ↔ 云 | **FastAPI + REST** | 简单清晰，符合 JSON schema |
| 容器 | 编排 | **Docker Compose** | 作业明文要求 |
| 测试 | 单元 / 集成 | `pytest` | 写一组关键测试就够答辩用 |

> ⚠ 不要用：未发布 / 实验性框架；任何要求 GPU 推理的部署方案；任何 Kaggle 现成 PV 故障数据集。

---

## 三、仓库目录结构

> 严格遵循 `projectdesignrules.cursorrules` §2 推荐结构：`models/training/inference/quantization/evaluation/rag/agent/tools/api` 全部独立模块；`tools/` 与 `agent/` 解耦；`rag/` 与 `agent/` 解耦；`tests/` 集中。

```
agentpv/
├── README.md                          # 5 分钟跑通指南
├── docker-compose.yml                 # 一键启动（rule §14）
├── pyproject.toml                     # 依赖 + 版本锁定
├── .env.example                       # API key 模板（rule §4）
├── .gitignore
│
├── configs/                           # 集中配置（rule §5）
│   ├── base.yaml                      # 通用默认值
│   ├── dev.yaml                       # 开发环境
│   ├── test.yaml                      # 测试环境
│   ├── prod.yaml                      # 演示/生产
│   └── settings.py                    # Pydantic Settings 加载器
│
├── docs/
│   ├── data_card.md                   # 交付物 1
│   ├── architecture.md                # 系统架构图 + 说明
│   ├── alert_schema.json              # 边缘↔云 JSON schema 契约（rule §3）
│   ├── api.md                         # OpenAPI 文档
│   └── ai_disclosure.md               # AI 工具使用披露（学术诚信）
│
├── data/                              # 数据资产
│   ├── raw/                           # 原始仿真输出
│   ├── processed/                     # 切窗后的训练数据
│   ├── splits/                        # train/val/test 索引（固定种子）
│   ├── version.txt                    # 数据集版本号（rule §6）
│   └── README.md
│
├── simulation/                        # 组件 1：物理仿真
│   ├── pv_simulator.py
│   ├── battery_simulator.py
│   ├── fault_injector.py              # 故障调度器
│   ├── generate_dataset.py            # 主入口
│   └── schemas.py                     # Pydantic 样本 schema
│
├── models/                            # 组件 2 - 架构定义（rule §2）
│   ├── base.py                        # 抽象基类（forward / num_params）
│   ├── cnn1d.py                       # 主选 1D CNN
│   ├── lstm.py                        # 备选 LSTM
│   └── registry.py                    # 模型注册表
│
├── training/                          # 组件 2 - 训练
│   ├── trainer.py                     # 训练循环类
│   ├── losses.py                      # 含 weighted CE / focal
│   ├── callbacks.py                   # 早停 / checkpoint
│   └── train.py                       # CLI 入口
│
├── quantization/                      # 组件 2 - 压缩（rule §8）
│   ├── prune.py                       # 结构化剪枝
│   ├── quantize.py                    # INT8 静态量化
│   ├── export_onnx.py                 # ONNX 导出
│   └── compress_pipeline.py           # 编排 prune → quantize → export
│
├── inference/                         # 组件 2/6 - 推理运行时
│   ├── onnx_runner.py                 # CPU-only ONNX 包装类
│   ├── postprocess.py                 # logits → label + severity 映射
│   └── benchmark.py                   # 延迟/内存/吞吐量/启动时间
│
├── evaluation/                        # 组件 3 - 模型评估
│   ├── metrics.py                     # P/R/F1/macroF1
│   ├── confusion.py                   # 混淆矩阵 + 热图
│   ├── error_analysis.py              # 错分案例分析
│   └── report.py                      # 自动生成 model_eval.md
│
├── rag/                               # 组件 4 - RAG 子系统（rule §10）
│   ├── chunking.py                    # 文档切片
│   ├── embedding.py                   # bge-small / MiniLM 包装
│   ├── retrieval.py                   # 向量召回
│   ├── reranking.py                   # ★ 重排（cross-encoder 或 MMR）
│   ├── prompting.py                   # 检索结果 → prompt 拼装
│   ├── store.py                       # ChromaDB / FAISS 抽象层
│   ├── ingest.py                      # 编排：load → chunk → embed → 入库
│   └── knowledge_base/
│       ├── documents/                 # 30+ md
│       └── chroma_db/                 # 持久化（容器卷挂载）
│
├── tools/                             # 组件 4 - 工具（rule §11，独立于 agent）
│   ├── base.py                        # Tool 抽象基类（typed I/O + timeout + retry）
│   ├── retrieve_knowledge.py
│   ├── get_system_history.py
│   ├── estimate_rul.py
│   ├── escalate.py
│   └── registry.py                    # 工具注册 + JSON schema 自动导出
│
├── agent/                             # 组件 4 - Agent 主体（rule §9）
│   ├── prompts/
│   │   ├── system.j2                  # 系统提示词模板
│   │   ├── reflect.j2                 # 反思阶段
│   │   └── final_answer.j2            # 报告阶段
│   ├── workflows/
│   │   └── react.py                   # ReAct 主循环（Observe→Reason→Act→Reflect→Report）
│   ├── memory/
│   │   └── alert_log.py               # 短期告警历史
│   ├── reasoning/
│   │   └── trace.py                   # 推理链建模 + 序列化
│   ├── orchestration/
│   │   ├── llm_client.py              # DeepSeek / Ollama 双后端
│   │   └── tool_dispatcher.py         # 调用 tools/ 并捕获异常
│   └── schemas.py                     # Recommendation 输出 schema
│
├── api/                               # 组件 6 - REST 服务（rule §13）
│   ├── edge_service.py                # FastAPI :8000
│   ├── agent_service.py               # FastAPI :8001
│   ├── deps.py                        # FastAPI 依赖注入（settings/onnx 单例）
│   └── errors.py                      # 结构化错误响应模型
│
├── dashboard/                         # 组件 7（rule §15）
│   ├── app.py                         # Streamlit 主入口
│   ├── components/
│   │   ├── node_grid.py
│   │   ├── alert_panel.py
│   │   └── reasoning_view.py
│   └── api_client.py                  # 与 edge/agent 通信
│
├── orchestrator/                      # 组件 6 - 多节点编排
│   ├── node_simulator.py              # 模拟 10+ 节点并发
│   ├── pipeline.py                    # 端到端编排
│   └── latency_test.py                # P95 延迟测试
│
├── agent_eval/                        # 组件 5 - Agent 评估
│   ├── benchmark.json                 # 30+ 测试场景
│   ├── judge.py                       # LLM-as-judge
│   ├── ablations.py                   # 消融实验
│   └── results/
│
├── docker/                            # 各服务镜像（rule §14）
│   ├── edge.Dockerfile
│   ├── agent.Dockerfile
│   ├── dashboard.Dockerfile
│   ├── orchestrator.Dockerfile
│   └── vector-db.Dockerfile           # ChromaDB 单独服务（rule 偏好）
│
├── utils/                             # 通用工具（rule §4）
│   ├── logging_config.py              # 统一 logger（结构化 JSON）
│   ├── timing.py                      # 计时器 / decorator
│   ├── seeds.py                       # set_global_seed(42)
│   └── paths.py                       # pathlib 路径常量
│
├── tests/                             # 集中三层测试（rule §16）
│   ├── unit/
│   │   ├── test_simulation.py
│   │   ├── test_models.py
│   │   ├── test_quantization.py
│   │   ├── test_rag_chunking.py
│   │   ├── test_rag_retrieval.py
│   │   ├── test_tools.py
│   │   └── test_schemas.py
│   ├── integration/
│   │   ├── test_edge_service.py       # FastAPI TestClient
│   │   ├── test_agent_service.py
│   │   └── test_rag_pipeline.py
│   └── e2e/
│       └── test_full_pipeline.py      # 启动 docker-compose 后跑通
│
├── reports/
│   ├── final_report.pdf               # 交付物 9
│   ├── model_eval.md                  # 交付物 4
│   ├── agent_eval.md                  # 交付物 6
│   └── figures/
│
└── scripts/
    ├── make_dataset.sh
    ├── train_all.sh
    ├── run_eval.sh
    └── benchmark_all.sh
```

### 3.1 模块依赖方向（不可逆）

```
configs/utils  →  ←─────  所有模块都可依赖
simulation     →  data/
models         →  utils
training       →  models, simulation, configs
quantization   →  models, training
inference      →  quantization (onnx), configs
evaluation     →  inference, models
rag            →  configs, utils
tools          →  rag (retrieve), utils
agent          →  tools, rag, configs
api            →  inference, agent
dashboard      →  api (HTTP only, never import直接 agent/inference)
orchestrator   →  api (HTTP only)
tests/*        →  any
```

> **关键**：`dashboard` 和 `orchestrator` **只能通过 HTTP 调用 api**，不可 `import inference / agent`，否则违反规则 §2 模块化与 §14 容器化。

---

## 四、作业要求 → 交付物对照表

| # | 作业要求的交付物 | 在仓库的位置 | 责任组件 |
|---|---|---|---|
| 1 | Data Card | `docs/data_card.md` | C1 |
| 2 | 数据集（≥50k，固定划分） | `data/processed/` + `data/splits/` | C1 |
| 3 | 边缘模型（ONNX + benchmark） | `quantization/artifacts/model.onnx` + `inference/benchmark.py` 输出 | C2 |
| 4 | 模型评估报告 | `reports/model_eval.md` | C3 |
| 5 | LLM Agent | `agent/` 整个目录 | C4 |
| 6 | Agent 评估报告（含消融） | `reports/agent_eval.md` | C5 |
| 7 | Dockerized 集成系统 | `docker-compose.yml` + 各 service | C6 |
| 8 | 仪表盘 | `dashboard/` | C7 |
| 9 | 最终技术报告 | `reports/final_report.pdf` | 全员 |
| 10 | 现场演示 + Q&A | （现场） | 全员 |

---

## 五、七大组件详细执行方案

### 组件 1：数据生成（Component 1 — Data Generation）

**权重 15%**｜**估时 1 周**｜**前置：无**

#### 1.1 目标

用物理仿真生成 ≥50,000 条带标签时序样本，覆盖 PV 7 类 + 电池 5 类故障，文档化、可复现。

#### 1.2 故障类别清单（必须全部覆盖）

| PV 故障 | 物理建模思路 |
|---|---|
| Normal | pvlib 标准发电曲线（晴天 / 多云 / 阴天） |
| Partial shading | 部分组件输出降至 30~70%，I-V 曲线出现拐点 |
| Soiling / dust | 整体辐照度降低 5~25%，缓慢衰减 |
| Bypass diode fault | 某串电压跌至接近 0，温度局部升高 |
| String disconnection | 阶跃式电流跌零 |
| Inverter fault | 输出功率振荡 / DC-AC 转换效率骤降 |
| Degradation | 长时间序列上发电量年衰减 0.5~1% |

| 电池故障 | 物理建模思路 |
|---|---|
| Normal | RC ECM 标准充放电曲线 |
| Capacity fade | 容量随循环数线性下降至 80% |
| Internal resistance ↑ | R0 增加 30~100% |
| Thermal anomaly | 温度上升速率异常（>2°C/min） |
| Cell imbalance | 多单体 SOC 偏差 >5% |

#### 1.3 数据 schema（每条样本）

每个样本是一个**时间窗口**（建议 60 秒，1 Hz 采样 → 60 步）：

```python
{
  "system_id": "PV_001" | "BESS_007",
  "system_type": "PV" | "BESS",
  "timestamp_start": "2026-01-01T08:00:00",
  "operating_condition": "high_irradiance" | "low_irradiance" | "high_temp",
  "label": "partial_shading",
  "features": np.array(shape=(60, F))   # F=8~12 维传感器
}
```

**PV 传感器维度（建议 8 维）**：
- 直流电压 V_dc
- 直流电流 I_dc
- 功率 P
- 模块温度 T_module
- 环境温度 T_amb
- 辐照度 G
- 逆变器输出 P_ac
- 效率 η

**电池传感器维度（建议 8 维）**：
- 端电压 V_term
- 电流 I
- SOC
- 温度 T
- 内阻估计 R_est
- 单体电压标准差 σ_V
- 累计循环数 N_cycle
- 容量保持率 SoH

#### 1.4 三种工况（必须）

```python
OPERATING_CONDITIONS = {
    "high_irradiance":  {"G_range": (800, 1100), "T_amb": 25},
    "low_irradiance":   {"G_range": (100, 400),  "T_amb": 20},
    "high_temperature": {"G_range": (600, 900),  "T_amb": 40},
}
```

#### 1.5 类别分布与样本数

| 类别 | 目标样本数 | 备注 |
|---|---|---|
| PV Normal | 8000 | 主类 |
| PV 6 故障类 × 平均 | 各 3500 | 共 21,000 |
| BESS Normal | 5000 | |
| BESS 4 故障类 × 平均 | 各 4250 | 共 17,000 |
| **合计** | **51,000** | 略超 50k |

> 严重不平衡时，在训练阶段使用 `WeightedRandomSampler` 或 focal loss 应对（在 Data Card 里写明）。

#### 1.6 数据划分

```python
SEED = 42
# 70/15/15，按 system_id 分层抽样保证训练/验证/测试无 ID 泄漏
train_ids, val_ids, test_ids = split_by_system_id(all_ids, [0.7, 0.15, 0.15], seed=SEED)
```

**保存**：`data/splits/train.csv` / `val.csv` / `test.csv`，包含 `sample_id, system_id, label, file_path`。

#### 1.7 关键代码骨架

```python
# simulation/generate_dataset.py
from simulation.pv_simulator import PVSystem
from simulation.battery_simulator import BatteryECM
from simulation.fault_injector import inject_fault

def generate(n_pv=21000, n_bess=17000, n_normal=13000, seed=42):
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_pv + n_bess + n_normal):
        cond = rng.choice(list(OPERATING_CONDITIONS))
        if rng.random() < 0.5:
            sys = PVSystem(condition=cond)
        else:
            sys = BatteryECM(condition=cond)
        label = sample_label(sys.type, rng)
        ts = sys.simulate(window=60)  # 60s
        ts = inject_fault(ts, label)
        samples.append({"features": ts, "label": label, ...})
    save_parquet(samples, "data/processed/")
```

#### 1.8 Data Card 必含内容（`docs/data_card.md`）

1. 数据集概述（1 段）
2. 类别清单 + 每类样本数 + 占比饼图
3. 特征 schema 表
4. 仿真参数与故障注入规则
5. 三种工况定义
6. 划分方法 + 随机种子
7. 已知局限（"基于物理模型，不能替代真实场地数据"）
8. 复现命令：`python -m simulation.generate_dataset --seed 42`

#### 1.9 验收标准

- [ ] `data/processed/` 含 ≥51,000 个样本文件（或 parquet）
- [ ] `data/splits/{train,val,test}.csv` 三个文件，按 seed=42 可复现
- [ ] `docs/data_card.md` 完整
- [ ] 用 `pytest simulation/tests/` 验证：每类样本数对、特征维度对、无 NaN

---

### 组件 2：模型构建（Component 2 — Model Build）

**权重 15%**｜**估时 1.5 周**｜**前置：C1 完成**

#### 2.1 目标

训练一个**统一**的多分类器（PV 7 类 + BESS 5 类 = 12 类，外加 Normal 共享或拆开均可），压缩后导出 ONNX。

> **设计选择**：建议**两种 Normal 拆开 = 12 类**（PV_Normal、BESS_Normal、PV 6 故障、BESS 4 故障），让 `system_type` 作为输入特征之一，避免单一模型混淆。

#### 2.2 架构（首选 1D CNN）

```python
# edge/models/cnn1d.py
class CNN1D(nn.Module):
    def __init__(self, in_ch=8, n_classes=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),                                          # 60 -> 30
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),                                  # 30 -> 1
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )
    def forward(self, x):  # x: (B, T, F) -> (B, F, T)
        return self.net(x.transpose(1, 2))
```

> **为什么 1D CNN**：参数量小（~50k）、训练快（CPU 也能跑）、量化后体积通常 <5 MB、推理延迟稳定 <20 ms。**作业明文要求**"justify your choice"，写报告时这段直接抄。

#### 2.3 训练配置

```python
# training/train.py
EPOCHS = 50
BATCH = 256
OPTIMIZER = AdamW(lr=1e-3, weight_decay=1e-4)
SCHEDULER = CosineAnnealingLR(T_max=50)
LOSS = CrossEntropyLoss(weight=class_weights)  # 应对不平衡
EARLY_STOP = patience=8 on val_macro_f1
```

固定种子：

```python
torch.manual_seed(42); np.random.seed(42); random.seed(42)
torch.use_deterministic_algorithms(True)
```

#### 2.4 三级告警映射（必须）

```python
SEVERITY_MAP = {
    "PV_Normal":               ("monitor", lambda c: True),
    "BESS_Normal":             ("monitor", lambda c: True),
    "Soiling":                 ("monitor", lambda c: c < 0.85),
    "Soiling":                 ("warning", lambda c: c >= 0.85),
    "Partial_shading":         ("warning", ...),
    "Bypass_diode_fault":      ("warning", ...),
    "String_disconnection":    ("critical", ...),
    "Inverter_fault":          ("critical", ...),
    "Degradation":             ("monitor", ...),
    "Capacity_fade":           ("monitor", ...),
    "Internal_resistance":     ("warning", ...),
    "Thermal_anomaly":         ("critical", ...),
    "Cell_imbalance":          ("warning", ...),
}
```

> 思路：`critical` 严格保留给"立刻可能损坏设备 / 安全风险"的故障；其余按置信度分级。

#### 2.5 压缩流水线

按以下顺序执行（每一步都保留中间产物方便组件 3 做对比）：

1. **基线** `model_fp32.pt`（FP32，未压缩）
2. **结构化剪枝**（剪 30% 通道）→ 微调 5 epoch → `model_pruned.pt`
3. **静态 INT8 量化**（用 200 条校准样本）→ `model_int8.pt`
4. **导出 ONNX** → `model.onnx`

```python
# quantization/quantize.py 关键步
import torch.ao.quantization as tq
model.eval()
model.qconfig = tq.get_default_qconfig("x86")
prepared = tq.prepare(model)
for batch in calib_loader: prepared(batch)  # 200 条校准
quantized = tq.convert(prepared)
torch.save(quantized.state_dict(), "model_int8.pt")
```

```python
# quantization/export_onnx.py
torch.onnx.export(
    model, dummy_input, "quantization/artifacts/model.onnx",
    input_names=["sensor_window"], output_names=["logits"],
    dynamic_axes={"sensor_window": {0: "batch"}},
    opset_version=17,
)
```

#### 2.6 Benchmark 脚本（必须能复现，rule §8 / §17）

按规则要求 **同时测量 4 项指标**：mean 延迟 / P95 延迟 / 内存占用 / 启动时间 / 吞吐量。

```python
# inference/benchmark.py
import os, time, gc, tracemalloc
import numpy as np, onnxruntime as ort
from utils.logging_config import get_logger

log = get_logger(__name__)

def benchmark_onnx(model_path: str, input_shape: tuple,
                   warmup: int = 50, iters: int = 1000) -> dict:
    """Benchmark CPU-only ONNX inference. Returns metrics dict."""
    # 1) 启动时间
    t0 = time.perf_counter()
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    startup_ms = (time.perf_counter() - t0) * 1000

    x = np.random.randn(*input_shape).astype(np.float32)
    in_name = sess.get_inputs()[0].name

    # 2) Warmup（消除首调用 jit/缓存抖动）
    for _ in range(warmup):
        sess.run(None, {in_name: x})

    # 3) 内存峰值
    gc.collect(); tracemalloc.start()

    # 4) 延迟分布
    latencies = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {in_name: x})
        latencies.append((time.perf_counter() - t0) * 1000)

    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()

    # 5) 吞吐量（batch=32 模拟并发）
    xb = np.random.randn(32, *input_shape[1:]).astype(np.float32)
    t0 = time.perf_counter()
    for _ in range(100):
        sess.run(None, {in_name: xb})
    throughput = 100 * 32 / (time.perf_counter() - t0)

    metrics = {
        "model_size_mb": os.path.getsize(model_path) / 1024 / 1024,
        "startup_ms":    startup_ms,
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_p95_ms":  float(np.percentile(latencies, 95)),
        "latency_p99_ms":  float(np.percentile(latencies, 99)),
        "memory_peak_mb": peak / 1024 / 1024,
        "throughput_qps": throughput,
    }
    log.info("benchmark", extra={"metrics": metrics})
    return metrics
```

**验收阈值**：

| 指标 | 阈值 |
|---|---|
| `model_size_mb` | ≤ 50 |
| `latency_mean_ms` | ≤ 100 |
| `latency_p95_ms` | ≤ 100 |
| `memory_peak_mb` | ≤ 200（建议） |
| `startup_ms` | ≤ 2000（建议） |
| `throughput_qps` | ≥ 200（建议） |

#### 2.7 验收标准

- [ ] `quantization/artifacts/model.onnx` 存在，≤ 50 MB（实际预期 <5 MB）
- [ ] benchmark 输出：mean ≤ 100 ms，P95 ≤ 100 ms
- [ ] 训练日志保留（TensorBoard 或 csv）
- [ ] 输出包含 severity 字段

---

### 组件 3：模型评估（Component 3 — Model Evaluation）

**权重 10%**｜**估时 0.5 周**｜**前置：C2 完成**

#### 3.1 必产出物（一个 Markdown + 配套图）

`reports/model_eval.md` 必须包含：

1. **整体指标表**

| 模型变体 | Macro F1 | Accuracy | 大小 | 平均延迟 | P95 延迟 |
|---|---|---|---|---|---|
| FP32 baseline | 0.93 | 0.95 | 18 MB | 35 ms | 48 ms |
| Pruned 30% | 0.92 | 0.94 | 12 MB | 22 ms | 30 ms |
| INT8 quantized | 0.91 | 0.93 | 4.5 MB | 12 ms | 18 ms |

2. **每类指标表**（PV 和 BESS 分开报告）

| 类别 | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| PV_Normal | 0.97 | 0.96 | 0.97 | 1200 |
| Partial_shading | 0.91 | 0.89 | 0.90 | 525 |
| ... | | | | |

3. **混淆矩阵热图**（`reports/figures/confusion_matrix.png`）

4. **压缩 trade-off 分析**（折线图：压缩率 vs F1）

5. **错误分析**（必须）

   - 哪两类最容易混淆？为什么？（如：Soiling 和 Degradation 都是缓慢衰减，物理特征接近）
   - 给出至少 3 个错分案例的具体特征

#### 3.2 关键代码

```python
# evaluation/report.py
from sklearn.metrics import classification_report, confusion_matrix, f1_score
y_true, y_pred = predict_all(test_loader, model)
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.4f}")
plot_confusion_matrix(y_true, y_pred, save="reports/figures/cm.png")
```

#### 3.3 验收标准

- [ ] Macro F1 ≥ 0.90
- [ ] 至少 2 个变体对比
- [ ] PV / BESS 分别报告
- [ ] 错误分析有 ≥ 3 个具体案例

---

### 组件 4：LLM Agent 构建（Component 4 — LLM Agent Build）

**权重 15%**｜**估时 2 周**｜**前置：JSON schema 定稿即可，可与 C2/C3 并行**

> 这是**得分密度最高**的组件，也是创新分主要来源。务必把 ReAct + RAG + 工具调用三件事讲清楚。

#### 4.1 知识库构建（≥ 30 篇文档）

文档分类（建议覆盖）：

| 类型 | 数量 | 示例文件名 |
|---|---|---|
| 故障描述 | 12 | `pv_partial_shading.md`, `bess_thermal_anomaly.md` ... |
| 维修动作 | 12 | `action_clean_panel.md`, `action_isolate_string.md` ... |
| 安全标准摘要 | 4 | `iec_61730_safety.md`, `ul_1973_battery.md` ... |
| 传感器阈值 | 4 | `pv_sensor_normal_ranges.md`, `bess_thresholds.md` ... |
| **合计** | **32** | |

每篇文档结构（统一模板）：

```markdown
---
doc_id: pv_partial_shading
type: fault_description
applies_to: [PV]
severity_default: warning
---

# Partial Shading（部分阴影）

## Definition
当光伏组件部分被遮挡（树叶 / 鸟粪 / 邻近建筑阴影），导致 ...

## Typical Sensor Signatures
- 直流电流 I_dc 下降 30~70%
- I-V 曲线出现多峰
- 模块温度局部升高 ...

## Recommended Actions
1. （warning）排查阴影源，标记位置
2. （critical）若持续 >2 h，断开受影响串
...

## References
- IEC 61853-1
- 内部知识库 doc_id: action_isolate_string
```

> **写作要点**：每篇 200~400 字，前面有 YAML 元数据，方便检索时过滤。

#### 4.2 RAG 流水线（模块化，rule §10）

按规则要求，**chunking / embedding / retrieval / reranking / prompting** 必须分文件，每一步可独立测试与替换。

**4.2.1 切片**（`rag/chunking.py`）

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter
from typing import List
from rag.schemas import Chunk

def chunk_markdown(md_text: str, doc_id: str) -> List[Chunk]:
    """Split markdown by headers. Each chunk carries doc_id metadata for citation."""
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#","h1"), ("##","h2")])
    raw = splitter.split_text(md_text)
    return [Chunk(doc_id=doc_id, text=r.page_content, metadata=r.metadata) for r in raw]
```

**4.2.2 向量化**（`rag/embedding.py`）

```python
from sentence_transformers import SentenceTransformer
from configs.settings import settings

class Embedder:
    def __init__(self, model_name: str = settings.embedding_model):
        self.model = SentenceTransformer(model_name)
    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()
```

**4.2.3 召回**（`rag/retrieval.py`）

```python
from rag.store import VectorStore  # 抽象层

class Retriever:
    def __init__(self, store: VectorStore, top_k: int = 8):
        self.store, self.top_k = store, top_k
    def retrieve(self, query: str, filter: dict | None = None) -> list[Chunk]:
        return self.store.similarity_search(query, k=self.top_k, filter=filter)
```

**4.2.4 重排**（`rag/reranking.py`，**规则要求必须有**）

```python
from sentence_transformers import CrossEncoder

class Reranker:
    """Cross-encoder reranker. 召回 top_k=8 → 重排 → 取 top_n=4 喂给 LLM."""
    def __init__(self, model: str = "BAAI/bge-reranker-base"):
        self.model = CrossEncoder(model)
    def rerank(self, query: str, chunks: list[Chunk], top_n: int = 4) -> list[Chunk]:
        scores = self.model.predict([(query, c.text) for c in chunks])
        return [c for _, c in sorted(zip(scores, chunks), reverse=True)][:top_n]
```

**4.2.5 提示词拼装**（`rag/prompting.py`）

```python
def build_context(chunks: list[Chunk]) -> str:
    """Format retrieved chunks with explicit doc_id citation markers."""
    return "\n\n".join(f"[doc:{c.doc_id}]\n{c.text}" for c in chunks)
```

**4.2.6 编排**（`rag/ingest.py`）

```python
def build_index(docs_dir: Path, persist_dir: Path) -> None:
    docs = load_all_md(docs_dir)
    embedder = Embedder()
    store = VectorStore.create(persist_dir)
    for doc_id, md in docs.items():
        chunks = chunk_markdown(md, doc_id)
        vectors = embedder.embed([c.text for c in chunks])
        store.add(chunks, vectors)
    store.persist()
```

> **可替换性**：`store.py` 抽象 ChromaDB / FAISS，切换只需改 `configs/base.yaml` 中 `vector_backend`。

#### 4.3 ReAct 循环（核心）

```python
# agent/workflows/react.py
SYSTEM_PROMPT = """
You are AgentPV, an industrial monitoring assistant for PV and BESS.
You receive a structured fault alert and must produce a recommendation.

You operate in a ReAct loop:
1. OBSERVE: read the alert
2. THINK: decide what knowledge you need
3. ACT: call one tool. Tools available:
   - retrieve_knowledge(query)
   - get_system_history(system_id, window)
   - estimate_rul(system_id)
   - escalate(system_id, reason)
4. REFLECT: do you have enough info? if no, go back to ACT (max 5 loops)
5. REPORT: output a JSON object with:
   {recommended_action, urgency, reasoning_trace, knowledge_sources, confidence}

Rules:
- Always retrieve at least 1 knowledge document.
- Cite doc_ids in knowledge_sources.
- If severity == critical, urgency must be 'immediate'.
- If you are unsure, set confidence='low' and recommend escalate.
"""

def run_agent(alert: dict, max_iter=5) -> dict:
    history = [{"role":"system", "content": SYSTEM_PROMPT},
               {"role":"user",   "content": f"ALERT:\n{json.dumps(alert)}"}]
    trace = []
    for i in range(max_iter):
        response = llm.chat(history, tools=TOOL_SCHEMAS)
        if response.tool_calls:
            for call in response.tool_calls:
                result = TOOLS[call.name](**call.args)
                trace.append({"step": i, "action": call.name,
                              "args": call.args, "result_summary": summarize(result)})
                history.append({"role":"tool", "name": call.name,
                                "content": json.dumps(result)})
        else:
            return parse_final_answer(response.content, trace)
    return fallback_answer(trace)
```

#### 4.4 工具实现（typed I/O + timeout + retry，rule §11）

所有工具继承统一基类，**强制类型化输入输出 + 超时保护 + 错误结构化**。

**`tools/base.py`** —— Tool 抽象基类

```python
import asyncio, time
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from pydantic import BaseModel
from utils.logging_config import get_logger

I = TypeVar("I", bound=BaseModel)
O = TypeVar("O", bound=BaseModel)
log = get_logger(__name__)

class ToolError(BaseModel):
    """Structured tool error returned to agent (never raise to caller)."""
    error_code: str   # TIMEOUT / VALIDATION / INTERNAL / NOT_FOUND
    message: str
    tool_name: str

class Tool(ABC, Generic[I, O]):
    name: str
    description: str
    input_model: type[I]
    output_model: type[O]
    timeout_s: float = 5.0
    max_retries: int = 1

    @abstractmethod
    async def _run(self, inp: I) -> O: ...

    async def __call__(self, raw_input: dict) -> dict:
        try:
            inp = self.input_model.model_validate(raw_input)
        except Exception as e:
            return ToolError(error_code="VALIDATION", message=str(e),
                             tool_name=self.name).model_dump()
        for attempt in range(self.max_retries + 1):
            try:
                t0 = time.perf_counter()
                out = await asyncio.wait_for(self._run(inp), timeout=self.timeout_s)
                log.info(f"tool_ok name={self.name} ms={(time.perf_counter()-t0)*1000:.1f}")
                return out.model_dump()
            except asyncio.TimeoutError:
                if attempt == self.max_retries:
                    return ToolError(error_code="TIMEOUT",
                            message=f"{self.name} exceeded {self.timeout_s}s",
                            tool_name=self.name).model_dump()
            except Exception as e:
                log.exception(f"tool_error name={self.name}")
                return ToolError(error_code="INTERNAL", message=str(e),
                                 tool_name=self.name).model_dump()
```

**`tools/retrieve_knowledge.py`**

```python
from rag.retrieval import Retriever
from rag.reranking import Reranker
from tools.base import Tool
from pydantic import BaseModel, Field

class RetrieveIn(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    top_n: int = Field(default=4, ge=1, le=10)

class RetrievedDoc(BaseModel):
    doc_id: str; snippet: str; score: float

class RetrieveOut(BaseModel):
    docs: list[RetrievedDoc]

class RetrieveKnowledgeTool(Tool[RetrieveIn, RetrieveOut]):
    name = "retrieve_knowledge"
    description = "Semantic search over the AgentPV knowledge base."
    input_model = RetrieveIn
    output_model = RetrieveOut
    timeout_s = 4.0

    def __init__(self, retriever: Retriever, reranker: Reranker):
        self.retriever, self.reranker = retriever, reranker

    async def _run(self, inp: RetrieveIn) -> RetrieveOut:
        candidates = self.retriever.retrieve(inp.query)
        reranked = self.reranker.rerank(inp.query, candidates, top_n=inp.top_n)
        return RetrieveOut(docs=[
            RetrievedDoc(doc_id=c.doc_id, snippet=c.text[:400], score=c.score)
            for c in reranked
        ])
```

**`tools/get_system_history.py` / `estimate_rul.py` / `escalate.py`** —— 同样模板：

| 工具 | input | output | timeout |
|---|---|---|---|
| `get_system_history` | `{system_id, window: "24h"}` | `{alerts: [...]}` | 2 s |
| `estimate_rul` | `{system_id}` | `{rul_days: int, method: str}` | 2 s |
| `escalate` | `{system_id, reason}` | `{ticket_id, status}` | 3 s |

> **关键**：工具**永不抛异常**给 agent，永远返回 `ToolError` 结构。Agent 的 reflect 阶段读到 error_code 后决定是否换工具。

#### 4.5 输出 schema（必须固定）

```json
{
  "recommended_action": "Inspect string 3 for shading; isolate if I_dc remains <50% of nominal for >30 min",
  "urgency": "scheduled",
  "reasoning_trace": [
    {"step": 0, "thought": "Alert is partial_shading on PV_007, severity=warning",
     "action": "retrieve_knowledge", "args": {"query": "partial shading remediation"}},
    {"step": 1, "thought": "Found doc pv_partial_shading; need history",
     "action": "get_system_history", "args": {"system_id": "PV_007", "window": "24h"}},
    {"step": 2, "thought": "No prior alerts; this is first-time event"}
  ],
  "knowledge_sources": ["pv_partial_shading", "action_isolate_string"],
  "confidence": "medium"
}
```

#### 4.6 验收标准

- [ ] 知识库 ≥ 30 篇，向量库可查询
- [ ] 4 个工具全部可调用
- [ ] ReAct 循环最多 5 步，超限有 fallback
- [ ] 输出严格符合 schema
- [ ] critical 告警自动升级 urgency=immediate

---

### 组件 5：LLM Agent 评估（Component 5 — LLM Agent Evaluation）

**权重 10%**｜**估时 1 周**｜**前置：C4 可调用**

#### 5.1 Benchmark 构建（≥ 30 个场景）

文件：`agent_eval/benchmark.json`

每条样例：

```json
{
  "id": "scn_001",
  "stakes": "high",                    // low/medium/high
  "alert": {
    "timestamp": "2026-03-15T14:23:00Z",
    "system_id": "BESS_004",
    "system_type": "BESS",
    "fault_class": "Thermal_anomaly",
    "severity": "critical",
    "confidence": 0.92,
    "sensor_snapshot": {"T": 68, "T_rate": 3.5, "V_term": 3.7, "I": 45}
  },
  "expected_outcome": {
    "key_action_keywords": ["isolate", "ventilate", "stop charging"],
    "expected_urgency": "immediate",
    "must_cite_docs": ["bess_thermal_anomaly", "action_emergency_isolation"],
    "must_not_say": ["wait", "monitor only"]
  }
}
```

**场景覆盖矩阵**（保证全覆盖）：

| | Normal | 1 故障 | 2 故障组合 / 模糊 |
|---|---|---|---|
| PV | 2 | 7（每类 1） | 3 |
| BESS | 2 | 5（每类 1） | 3 |
| 跨系统 | — | — | 8 |
| **合计** | **4** | **12** | **14** ≥ 30 ✓ |

> **作业要求至少 5 个模糊场景** — 我们设计 14 个，足够安全。

#### 5.2 评分维度（1–5）

| 维度 | 评分细则（自定义 rubric） |
|---|---|
| Correctness | 5=动作完全匹配，3=部分匹配，1=错误 |
| Actionability | 5=有具体步骤可执行，3=方向对但模糊，1=空话 |
| Interpretability | 5=推理链清晰可审计，3=有但不完整，1=无 |
| Safety | 5=无安全风险，3=轻微风险，1=可能加剧故障 |

#### 5.3 LLM-as-judge 实现

```python
# agent_eval/judge.py
JUDGE_PROMPT = """
You are an expert evaluator. Given:
- ALERT: {alert}
- EXPECTED: {expected}
- RESPONSE: {response}

Score 1-5 on each dimension, with brief justification.
Return JSON: {correctness:int, actionability:int, interpretability:int, safety:int, notes:str}
"""

def judge(alert, expected, response, judge_llm="deepseek-chat") -> dict:
    return parse_json(call_llm(JUDGE_PROMPT.format(...), model=judge_llm))
```

> **必须讨论 LLM-as-judge 的局限**（作业明文要求）：偏向冗长回答、对自家模型有偏好、对边缘 case 不稳定。报告里写一段："因此我们额外人工抽样 10 个场景做交叉验证，与 judge 评分相关系数 r=0.83。"

#### 5.4 消融实验（必做）

| 配置 | 描述 |
|---|---|
| **Full** | RAG + ReAct + 工具 |
| **No-RAG** | 不调用 retrieve_knowledge |
| **No-trace** | 直接输出建议，无 reasoning_trace |

预期结果（这是写报告用的"叙事"）：

| 配置 | Correctness | Actionability | Interpretability | Safety |
|---|---|---|---|---|
| Full | 4.5 | 4.4 | 4.7 | 4.6 |
| No-RAG | 3.2 | 3.5 | 3.0 | 3.4 |
| No-trace | 4.3 | 4.2 | 2.1 | 4.0 |

#### 5.5 验收标准

- [ ] benchmark.json ≥ 30 场景，含 ≥ 5 模糊
- [ ] 三种配置均有结果
- [ ] 全维度平均 ≥ 4.0
- [ ] 报告含 LLM-as-judge 局限性讨论

---

### 组件 6：系统集成（Component 6 — Decision Making & Integration）

**权重 15%**｜**估时 1 周**｜**前置：C2 + C4 完成**

#### 6.1 服务架构

```
┌──────────────────────────────────────────────────────┐
│  docker-compose.yml                                  │
│                                                      │
│  ┌──────────────┐  HTTP    ┌──────────────┐         │
│  │ edge-service │ ────────>│ agent-service │         │
│  │ FastAPI :8000│  POST    │ FastAPI :8001 │         │
│  │ ONNX runtime │  /alert  │ LangChain     │         │
│  └──────┬───────┘          └───────┬──────┘         │
│         │                          │                 │
│         ▼                          ▼                 │
│   ┌──────────┐               ┌─────────┐            │
│   │ node_sim │               │ chroma  │            │
│   │ (10 nodes│               │  _db    │            │
│   └──────────┘               └─────────┘            │
│         ▲                                            │
│         │                                            │
│         ▼                                            │
│   ┌────────────────────────────────────────┐        │
│   │ dashboard (Streamlit :8501)            │        │
│   └────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────┘
```

#### 6.2 关键 API

```python
# api/edge_service.py
@app.post("/predict")
def predict(window: SensorWindow) -> Alert:
    logits = onnx_session.run(None, {"sensor_window": window.array})
    cls, conf = postprocess(logits)
    severity = SEVERITY_MAP[cls](conf)
    alert = Alert(timestamp=now(), system_id=window.system_id, ...)
    # 异步转发给 agent，超时 8s 后只返回告警
    asyncio.create_task(forward_to_agent(alert, timeout=8))
    return alert

# api/agent_service.py
@app.post("/recommend")
def recommend(alert: Alert) -> Recommendation:
    return run_agent(alert.dict())
```

#### 6.3 多节点并发模拟

```python
# orchestrator/node_simulator.py
async def simulate_node(node_id: str, system_type: str):
    while True:
        window = generate_realtime_window(node_id, system_type)
        async with httpx.AsyncClient() as c:
            await c.post("http://edge-service:8000/predict",
                         json=window, timeout=2)
        await asyncio.sleep(1)

async def main():
    tasks = [simulate_node(f"NODE_{i:03d}", random_type()) for i in range(10)]
    await asyncio.gather(*tasks)
```

#### 6.4 端到端延迟测试

```python
# orchestrator/latency_test.py
results = []
for _ in range(50):
    t0 = time.perf_counter()
    response = inject_fault_and_wait_recommendation()
    results.append(time.perf_counter() - t0)
print(f"P95: {np.percentile(results, 95):.2f}s")  # 必须 ≤ 10s
```

#### 6.5 优雅降级

```python
async def forward_to_agent(alert, timeout=8):
    try:
        async with httpx.AsyncClient() as c:
            return await c.post(AGENT_URL, json=alert, timeout=timeout)
    except (httpx.TimeoutException, httpx.ConnectError):
        log.warning(f"Agent unavailable, alert {alert.id} stored locally")
        local_queue.append(alert)
        # 后续恢复时重放
```

#### 6.6 消融实验（必做）

| 配置 | 延迟 | 可解释性 | 决策质量 |
|---|---|---|---|
| Edge only | 0.05 s | 无（仅类别+严重度） | 低 |
| Cloud only（原始数据→Agent） | 15 s（超时） | 中 | 中 |
| Full | 4 s | 高 | 高 |

#### 6.7 docker-compose.yml 骨架（rule §14）

四个服务全部独立容器化（含向量库单独成服务，符合规则偏好）。

```yaml
version: "3.9"
services:
  vector-db:
    build: { context: ., dockerfile: docker/vector-db.Dockerfile }
    ports: ["8002:8000"]              # ChromaDB 服务端口
    volumes:
      - ./rag/knowledge_base/chroma_db:/chroma/chroma
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 5s
      retries: 6

  edge-service:
    build: { context: ., dockerfile: docker/edge.Dockerfile }
    ports: ["8000:8000"]
    volumes:
      - ./quantization/artifacts:/app/artifacts:ro
      - ./configs:/app/configs:ro
    environment:
      - APP_ENV=prod
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 5s

  agent-service:
    build: { context: ., dockerfile: docker/agent.Dockerfile }
    ports: ["8001:8001"]
    environment:
      - APP_ENV=prod
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
      - VECTOR_DB_URL=http://vector-db:8000
    depends_on:
      vector-db:    { condition: service_healthy }
      edge-service: { condition: service_healthy }

  dashboard:
    build: { context: ., dockerfile: docker/dashboard.Dockerfile }
    ports: ["8501:8501"]
    environment:
      - EDGE_URL=http://edge-service:8000
      - AGENT_URL=http://agent-service:8001
    depends_on: [edge-service, agent-service]

  orchestrator:
    build: { context: ., dockerfile: docker/orchestrator.Dockerfile }
    environment:
      - EDGE_URL=http://edge-service:8000
    depends_on: [edge-service]
```

> 在 README 里明确声明：评估老师只需 `cp .env.example .env && docker compose up`。

#### 6.8 验收标准

- [ ] `docker compose up` 一键起所有服务
- [ ] 10 节点并发 5 分钟无错误
- [ ] P95 端到端延迟 ≤ 10s（50 次）
- [ ] 杀掉 agent-service 后边缘仍能产生告警
- [ ] 三种配置消融结果在报告中

---

### 组件 7：原型与演示（Component 7 — Prototype & Demo）

**权重 10%**｜**估时 0.5 周**｜**前置：C6 完成**

#### 7.1 仪表盘设计（Streamlit）

页面布局（两列）：

```
┌────────────────────────────┬─────────────────────────┐
│ LEFT: Node Grid            │ RIGHT: Selected Detail  │
│                            │                         │
│ [PV_001 🟢] [PV_002 🟡]    │ Node: PV_003            │
│ [PV_003 🔴] [PV_004 🟢]    │ Status: critical        │
│ [BESS_005 🟢] ...          │ Class: Inverter_fault   │
│                            │ Confidence: 0.91        │
│ Filter: [All|PV|BESS]      │                         │
│                            │ ── Sensor Snapshot ──   │
│ ── Inject Fault ──         │ V_dc=520  I_dc=8.2 ...  │
│ Node: [PV_003 ▼]           │                         │
│ Type: [Partial_shading ▼]  │ ── Agent Recommendation │
│ [ Inject ]                 │ Action: Isolate string..│
│                            │ Urgency: immediate      │
│ ── Live Alerts ──          │ Confidence: high        │
│ 14:23 PV_003 critical      │                         │
│ 14:21 BESS_007 warning     │ ▶ Show reasoning trace  │
│ ...                        │ ▶ Show knowledge sources│
└────────────────────────────┴─────────────────────────┘
```

#### 7.2 核心代码

```python
# dashboard/app.py
import streamlit as st
st.set_page_config(layout="wide", page_title="AgentPV")

col1, col2 = st.columns([1, 1])
with col1:
    st.header("Live Nodes")
    nodes = fetch_nodes_status()
    cols = st.columns(4)
    for i, n in enumerate(nodes):
        with cols[i % 4]:
            color = {"monitor":"🟢","warning":"🟡","critical":"🔴"}[n.severity]
            if st.button(f"{color} {n.id}\n{n.fault_class}"):
                st.session_state.selected = n.id

    st.divider()
    st.subheader("Inject Fault (Demo)")
    nid = st.selectbox("Node", [n.id for n in nodes])
    ftype = st.selectbox("Fault", FAULT_TYPES)
    if st.button("Inject"):
        inject_fault_via_orchestrator(nid, ftype)
        st.success(f"Injected {ftype} on {nid}")

with col2:
    if "selected" in st.session_state:
        node = fetch_node(st.session_state.selected)
        st.metric("Severity", node.severity)
        st.json(node.sensor_snapshot)
        rec = fetch_recommendation(node.last_alert_id)
        st.markdown(f"**Action**: {rec.recommended_action}")
        st.markdown(f"**Urgency**: `{rec.urgency}`")
        with st.expander("Reasoning trace"):
            for step in rec.reasoning_trace:
                st.code(step)
        with st.expander("Knowledge sources"):
            for s in rec.knowledge_sources:
                st.markdown(f"- `{s}`")
```

#### 7.3 验收标准

- [ ] 浏览器访问 `localhost:8501` 可见
- [ ] 至少 1 个交互场景（注入故障）可走通
- [ ] 推理链 / 知识来源可点开查看
- [ ] 颜色编码与 severity 一致
- [ ] 全部能在 `docker compose up` 后直接用

---

## 六、团队分工建议（4 人）

| 成员 | 主负责 | 协助 | 主要技能 |
|---|---|---|---|
| **A（队长）** | C2 模型 + C3 评估 | C6 集成 | PyTorch, ONNX |
| **B** | C1 仿真 + Data Card | C7 仪表盘 | numpy/pvlib |
| **C** | C4 Agent + C5 评估 | 知识库写作 | LangChain/LLM |
| **D** | C6 集成 + Docker + C7 仪表盘 | 全员协助 | FastAPI/Streamlit/Docker |

> ⚠ **作业明文要求**：每个成员必须能解释别人写的部分。建议每周一开 30 分钟"代码 walkthrough"互讲。

---

## 七、12 周时间表

> 假设每周 8~12 小时投入，4 人并行。

| 周 | 主线任务 | 关键里程碑 |
|---|---|---|
| W1 | 组队、读 PDF、定 schema、搭仓库骨架 | repo + 空 docker-compose 跑通 |
| W2 | C1：PV 仿真 + 故障注入 | 5 PV 类样本生成，可视化检查 |
| W3 | C1：电池仿真 + 数据合并 + Data Card | **数据集 ≥ 50k 完成** |
| W4 | C2：CNN 训练、跑通 baseline | Macro F1 baseline ≥ 0.88 |
| W5 | C2：剪枝 + 量化 + ONNX 导出 | model.onnx ≤ 50 MB, P95 ≤ 100ms |
| W5 | C4：知识库 30+ 文档、向量库构建（并行） | Chroma 可查询 |
| W6 | C3：评估报告写作；C4：ReAct + 4 工具 | model_eval.md 初稿；agent 可走通 1 alert |
| W7 | C5：benchmark 30+ 场景；C6：FastAPI 双服务 | 评估代码可跑；2 服务能 Docker 跑 |
| W8 | C5：消融 + LLM-as-judge；C6：10 节点并发测 | agent_eval.md 初稿；P95 ≤ 10s |
| W9 | C7：Streamlit 仪表盘；端到端调试 | `docker compose up` 全功能 |
| W10 | 写最终报告（章节并行） | report 初稿 |
| W11 | 录屏演示彩排 + 个人答辩练习 | 演示可重复 3 次零故障 |
| W12 | 最终演示 + 提交 | **交付完成** |

---

## 八、风险点与对策

| 风险 | 概率 | 影响 | 对策 |
|---|---|---|---|
| 仿真数据特征区分度不够，模型 F1 < 0.90 | 中 | 高 | 增加噪声水平梯度；加入显式物理拐点（如 partial shading 注入双峰 I-V）；先做特征 t-SNE 检查再训练 |
| LLM API 限流 / 失效 | 中 | 高 | 用 DeepSeek + Ollama qwen2.5 双后端，环境变量切换 |
| Docker 在 Windows 上容器互联问题 | 中 | 中 | 全部用 service 名互相解析，避免 localhost；统一 Linux 镜像 |
| INT8 量化精度大跌 | 低 | 中 | 留一份未量化模型作为 fallback；在 docs 里把 trade-off 写清楚 |
| Agent 输出 schema 不稳定（JSON 解析失败） | 高 | 中 | 强制 JSON 模式（DeepSeek `response_format`）+ Pydantic 校验 + 重试 1 次 |
| 端到端延迟超 10s | 中 | 高 | 边缘和云端解耦：边缘立即返回 alert，agent 异步推理；测延迟时分别测两段 |
| 团队某成员失联 | 低 | 高 | 第 1 周就要求每周提交记录到 GitHub Issues；W4 设第一次 checkpoint |
| AI 编程助手生成代码无人能解释 | 中 | **答辩零分** | 强制每段 AI 生成代码必须有作者注释 + 当周 walkthrough |

---

## 九、验收 Checklist（自检表）

> 提交前逐项打勾，全部满足才算完成。

### 数据 & 模型
- [ ] `docs/data_card.md` 完整
- [ ] `data/processed/` ≥ 51,000 样本
- [ ] PV 7 类 + BESS 5 类全部存在
- [ ] 3 种工况覆盖
- [ ] `seed=42` 可复现划分
- [ ] `quantization/artifacts/model.onnx` ≤ 50 MB
- [ ] CPU 推理 P95 ≤ 100 ms
- [ ] Macro F1 ≥ 0.90
- [ ] 至少 2 个模型变体对比
- [ ] 混淆矩阵图存在

### Agent
- [ ] 知识库 ≥ 30 篇文档
- [ ] 4 个工具全部实现
- [ ] ReAct 循环 + 输出 schema
- [ ] benchmark ≥ 30 场景
- [ ] 4 维度平均 ≥ 4.0
- [ ] 消融实验：No-RAG / No-trace / Full

### 系统
- [ ] `docker compose up` 一键起
- [ ] 10 节点并发 OK
- [ ] P95 端到端 ≤ 10 s
- [ ] 优雅降级测试通过
- [ ] 仪表盘 1 个交互场景可跑

### 报告 & 答辩
- [ ] 最终报告 10 章齐全
- [ ] ≥ 8 篇相关文献
- [ ] AI 工具使用披露
- [ ] 每个成员能讲任意组件

---

## 十、最终报告写作指南

按作业要求的 10 章结构，每章重点：

1. **Introduction**（1 页）— 问题、动机、本系统贡献（3 点）
2. **Related Work**（2 页）— 至少 8 篇，按"故障检测 / 边缘 AI / LLM Agent"分组
3. **Data Generation**（2 页）— 仿真方法、故障物理、统计图
4. **Edge AI Module**（3 页）— 架构图 + 训练曲线 + 压缩对比表
5. **LLM Agent**（3 页）— ReAct 流程图 + 工具表 + RAG 索引说明
6. **Evaluation**（3 页）— 模型评估 + Agent 评估 + 消融
7. **System Integration**（2 页）— 服务架构图 + 延迟分布直方图 + 仪表盘截图
8. **Discussion**（1 页）— 局限：仿真数据 vs 真实数据；LLM-as-judge 偏差
9. **Conclusion**（0.5 页）
10. **References**（IEEE 格式）
11. **Appendix** — Data Card、JSON schema、知识库索引、Docker 命令、AI 使用披露

> **写作原则**：每个数字都有来源（图 / 表 / 命令），不写废话。

---

## 十一、答辩准备

预期老师会问的问题（提前准备答案）：

1. "为什么选 1D CNN 而不是 Transformer？" → 数据规模 + 延迟约束
2. "INT8 量化为什么精度只掉 2%？校准集怎么选的？"
3. "RAG 召回失败怎么办？" → confidence='low' + escalate
4. "ReAct 比直接 prompt 好在哪？" → 拿消融数据回答
5. "端到端 P95 延迟主要来自哪一段？" → 拿 latency breakdown 回答
6. "如果 critical 告警 agent 挂了怎么办？" → 优雅降级 + 本地队列
7. "你们的数据是仿真的，怎么证明对真实场景有意义？" → 物理建模 + 可迁移性讨论
8. "LLM-as-judge 怎么避免自夸？" → 跨模型 + 人工抽样

每个成员准备 3 分钟：
- 我做了什么
- 关键技术选型 + 理由
- 一个有挑战的 bug 怎么解决的

---

## 十二、工程规则总览（projectdesignrules 对齐）

> 本节是 `projectdesignrules.cursorrules` 22 条规则在本项目中的**落地清单**。它**优先级高于本方案中任何其它章节**——任何冲突以本节为准。

### 12.1 工作流纪律（rule §1, §18, §19, §22）

每接到一个新模块的开发任务，必须按以下顺序：

1. **Design** —— 在 PR 描述或 GitHub Issue 中先写：架构、模块边界、接口、tradeoff
2. **Implement** —— 一次只写一个小模块，**禁止一次性生成大量代码**
3. **Test** —— 单元测试必须先于集成
4. **Validate** —— 用真实输入跑通
5. **Integrate** —— 接入上游
6. **Refactor** —— 测试覆盖到位再清理

> 写完代码后必须解释：how it works / dependencies / how to test / integration points。

### 12.2 接口先行（rule §3）

**本项目的契约清单**（开发前全部用 Pydantic 落到 `*/schemas.py`）：

| Schema | 位置 | 角色 |
|---|---|---|
| `SensorWindow` | `simulation/schemas.py` | 数据样本 |
| `Alert` | `api/schemas.py` | **边缘 ↔ 云固定契约**（作业明文要求，不可改） |
| `Recommendation` | `agent/schemas.py` | Agent 输出 |
| `Chunk` | `rag/schemas.py` | 知识切片 |
| `ToolError` | `tools/base.py` | 工具结构化错误 |
| `Settings` | `configs/settings.py` | 全局配置 |

**写代码规则**：任何函数签名出现 `dict` 都要审视一次能否换成 Pydantic 模型。

### 12.3 配置管理（rule §5）

```python
# configs/settings.py
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml, os

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AGENTPV_")

    # 路径
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / "data"
    artifacts_dir: Path = project_root / "quantization" / "artifacts"

    # ML
    seed: int = 42
    batch_size: int = 256
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "BAAI/bge-reranker-base"

    # Inference
    onnx_threads: int = 4
    inference_latency_budget_ms: int = 100

    # Agent
    llm_backend: str = "deepseek"   # deepseek | ollama
    llm_model: str = "deepseek-chat"
    llm_timeout_s: int = 30
    react_max_iterations: int = 5

    # 服务
    edge_url: str = "http://edge-service:8000"
    agent_url: str = "http://agent-service:8001"
    vector_db_url: str = "http://vector-db:8000"

    # 密钥
    deepseek_api_key: str = ""

def load_settings() -> Settings:
    env = os.getenv("APP_ENV", "dev")
    yaml_path = Path(__file__).parent / f"{env}.yaml"
    overrides = yaml.safe_load(yaml_path.read_text()) if yaml_path.exists() else {}
    return Settings(**overrides)

settings = load_settings()
```

**禁止做的事**：
- ❌ 任何模块出现字符串硬编码路径（用 `settings.data_dir / "..."`）
- ❌ 任何模块用 `os.environ.get("API_KEY")` 直接读环境变量（必须经 settings）
- ❌ 任何"魔法数字"（5, 42, 0.85）出现在业务代码（必须落 settings 或 named constant）

### 12.4 代码规范（rule §4）

每个 .py 文件**强制**包含：

```python
"""模块一句话职责。"""
from __future__ import annotations
from pathlib import Path
from utils.logging_config import get_logger

log = get_logger(__name__)

def my_function(window: np.ndarray, threshold: float) -> dict:
    """Do X.
    
    Args:
        window: shape (T, F) sensor readings.
        threshold: must be in (0, 1).
    Returns:
        Dict with keys: ...
    Raises:
        ValueError: when threshold out of range.
    """
    if not 0 < threshold < 1:
        raise ValueError(f"threshold={threshold} not in (0,1)")
    log.debug("processing", extra={"shape": window.shape})
    ...
```

**强制清单**：
- [x] 所有公开函数有 type hints
- [x] 所有公开函数有 docstring（Args/Returns/Raises）
- [x] 用 `log = get_logger(__name__)` 替代所有 `print`
- [x] 用 `pathlib.Path` 替代字符串拼路径
- [x] 用 `os.getenv` 仅在 `configs/settings.py` 内出现
- [x] 重复 ≥3 次的逻辑必须抽函数

### 12.5 错误处理与可靠性（rule §12）

**三层错误处理策略**：

| 层 | 处理方式 |
|---|---|
| Tool 层 | 永不抛异常，返回 `ToolError` 结构（见 12.2） |
| Service 层 | FastAPI 异常处理器统一转 `ErrorResponse`，含 `error_code/message/trace_id` |
| Pipeline 层 | 边缘 → 云 调用失败 / 超时 → 仍返回 alert，**绝不阻塞边缘** |

**`api/errors.py`** 标准错误响应：

```python
from pydantic import BaseModel
class ErrorResponse(BaseModel):
    error_code: str       # MODEL_LOAD_FAIL / VALIDATION / TIMEOUT / DOWNSTREAM_DOWN
    message: str
    trace_id: str         # 用于日志关联
    retry_after_s: int | None = None
```

**重试策略**：
- LLM API 调用：最多 2 次重试，指数退避（1s → 3s）
- ChromaDB 召回：最多 1 次重试
- 跨服务 HTTP：使用 `httpx` + `tenacity`，限 8s 总预算

**禁止行为**：silent failure（捕获异常但不 log、不 raise、不返回错误结构）。

### 12.6 测试策略（rule §16）

**三层覆盖率目标**：

| 层 | 位置 | 目标 | 何时跑 |
|---|---|---|---|
| Unit | `tests/unit/` | 行覆盖 ≥ 70% | 每次 commit（pre-commit hook） |
| Integration | `tests/integration/` | 关键路径 100% | 合并到 main 前 |
| E2E | `tests/e2e/` | 1 个完整 happy path | 周末 + release |

**必须有的测试样本**：

```python
# tests/unit/test_schemas.py
def test_alert_schema_rejects_unknown_severity():
    with pytest.raises(ValidationError):
        Alert(severity="banana", ...)

# tests/unit/test_tools.py
async def test_tool_timeout_returns_structured_error():
    tool = SlowTool(timeout_s=0.1)
    out = await tool({"x": 1})
    assert out["error_code"] == "TIMEOUT"

# tests/integration/test_edge_service.py
def test_predict_endpoint_returns_valid_alert(client):
    r = client.post("/predict", json=fake_window())
    Alert.model_validate(r.json())   # schema 校验

# tests/e2e/test_full_pipeline.py
def test_inject_fault_yields_recommendation_within_10s():
    inject_fault("PV_001", "partial_shading")
    rec = wait_for_recommendation("PV_001", timeout=10)
    assert rec["urgency"] in {"immediate", "scheduled", "monitor"}
```

### 12.7 性能监控（rule §17）

每个产出物（模型 / RAG / Agent / 整链）都要测**四个维度**：

| | 模型 | RAG | Agent | 端到端 |
|---|---|---|---|---|
| 延迟 | mean / p95 (ms) | 召回 + 重排 (ms) | 单次响应 (s) | inject→reco (s) |
| 内存 | peak (MB) | 索引大小 | LLM context tokens | 容器 RSS |
| 启动时间 | onnx load (ms) | chroma 加载 (ms) | warmup time | docker compose up |
| 吞吐 | qps | qps | parallel agents | nodes × predicts/s |

> "**Do not optimize prematurely. Measure first.**"——所有优化前先把 baseline 数字记入 `reports/perf_baseline.md`。

### 12.8 禁止行为清单（rule §21）

**自检表，PR 前逐项确认**：

- [ ] 没有未解释的 AI 生成代码（每段都能讲清意图）
- [ ] 没有硬编码 API key / 路径 / 端口
- [ ] 没有跳过 schema 验证的 `.dict()` 直接传递
- [ ] 没有混用职责的"上帝类"（>300 行的 .py 文件警觉）
- [ ] 没有捕获异常后吞掉（必须 log + 返回结构化错误）
- [ ] 没有伪造 benchmark 数字（所有数字可由 `scripts/benchmark_all.sh` 复现）
- [ ] 没有为了"看起来高级"引入的额外框架

### 12.9 角色思维（rule §22）

**写代码时切换四种视角**：

| 视角 | 关注 |
|---|---|
| AI Systems Engineer | 模块边界、抽象层次 |
| MLOps Engineer | 可复现、可监控、可回滚 |
| Distributed Systems Engineer | 超时、降级、幂等 |
| Production Deployment Engineer | 容器、配置、密钥、日志 |

> 一段代码如果**通不过这四个视角的审查**，就要重写。

---

## 附录：第一天上手命令

```bash
# 1. 创建仓库（按本方案 §三 目录结构 + §十二 工程规则）
mkdir agentpv && cd agentpv
git init
python -m venv .venv
.venv\Scripts\activate                                            # Windows

# 2. 安装核心依赖
pip install torch numpy pandas pvlib scipy
pip install langchain langchain-community chromadb sentence-transformers
pip install fastapi uvicorn httpx tenacity pydantic pydantic-settings pyyaml
pip install streamlit onnxruntime onnx
pip install pytest pytest-asyncio pytest-cov

# 3. 创建规则要求的全部模块目录
mkdir configs docs data simulation models training quantization inference
mkdir evaluation rag tools agent api dashboard orchestrator agent_eval
mkdir docker utils tests reports scripts
mkdir tests\unit tests\integration tests\e2e
mkdir agent\prompts agent\workflows agent\memory agent\reasoning agent\orchestration
mkdir rag\knowledge_base rag\knowledge_base\documents

# 4. 占位文件 + .gitignore
type nul > configs\settings.py
type nul > utils\logging_config.py
type nul > docs\alert_schema.json
type nul > .env.example
echo .venv/ > .gitignore
echo __pycache__/ >> .gitignore
echo data/processed/ >> .gitignore
echo *.onnx >> .gitignore

# 5. 第一个 commit
git add .
git commit -m "init: AgentPV scaffolding aligned with engineering rules"

# 6. 第一周首要任务：定 schemas.py + settings.py + logging_config.py
#    然后开始 C1 数据生成
```

---

**祝顺利交付，拿满分。**

任何一步卡住：
1. 先翻第十二节工程规则（看是不是踩规则禁区）
2. 再翻对应组件章节（C1~C7）
3. 最后翻验收 Checklist（§九）确认未漏交付物
