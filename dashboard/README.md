# `dashboard/` — Streamlit 操作员仪表盘（Component 7）

按规则 §24 的 14 节工程文档。本模块是 AgentPV 的**前端层**——读 orchestrator
写出的 JSONL 事件流并按操作员视角渲染。代码严格分两层：纯函数数据层
（`data.py`，可单测）+ Streamlit 渲染层（`app.py`）。

---

## 1. 模块目的（Purpose）

让操作员在浏览器里实时看到：每个 PV/BESS 节点的状态、最近告警时间线、单条事件
的完整 reasoning trace、严重度 / 故障类别 / 延迟统计。MVP 用手动刷新；
polish 阶段加 auto-refresh + 推送通知。

---

## 2. 为什么需要这个模块（Why it exists）

- 作业 §4.7 要求"web-based dashboard for operators with real-time alerts +
  agent reasoning"
- 没有 dashboard，agent 的 reasoning trace 只能在日志里翻——操作员体验 0
- 在答辩展示时，dashboard 是把整个三层系统讲清楚的最直接载体（节点总览 →
  时间线 → 单事件 reasoning → 全局统计）。现场步骤见
  `docs/Dashboard Demo Guide.md`。

---

## 3. 架构概览（Architecture）

```text
data/orchestrator/events.jsonl   ← orchestrator 写
            │
            ▼
   dashboard.data.load_events()      （纯函数，跳过坏行不阻塞）
            │
            ▼
   list[OrchestratorEvent]
            │
            ├──► events_to_dataframe()      → tab 2 / 3 表格
            ├──► per_node_summary()         → tab 1 节点总览
            ├──► severity_counts()          → tab 4 柱状图
            ├──► fault_class_counts()       → tab 4 柱状图
            ├──► latency_stats()            → tab 4 metric
            ├──► severity_over_time(5s)     → tab 2 活动柱图
            └──► get_event_by_id() / filter_events()
                                            │
                                            ▼
                                  dashboard.app  (Streamlit)
                                  ├── sidebar 控件
                                  ├── tab 1: 节点总览
                                  ├── tab 2: 事件时间线
                                  ├── tab 3: 事件详情
                                  └── tab 4: 全局统计
```

---

## 4. 关键文件（Key files）

| 文件 | 类型 | 作用 |
|---|---|---|
| `data.py` | 纯函数 | `LoadResult`, `load_events`, `events_to_dataframe`, `per_node_summary`, `severity_counts`, `fault_class_counts`, `latency_stats`, `severity_over_time`, `filter_events`, `get_event_by_id` |
| `app.py` | Streamlit | 4 个 tab 的渲染 + sidebar 过滤 + manual refresh |
| `components/` | 占位 | Polish 阶段拆出可复用组件（trace 折线图、单卡指标）|

---

## 5. 输入输出契约（Inputs & Outputs）

### 输入
- `data/orchestrator/events.jsonl`：每行 1 个 `OrchestratorEvent`
  （rule §3，`api.schemas.OrchestratorEvent`）

### 输出
- 浏览器 UI（`http://127.0.0.1:8501`）
- 无文件副作用（dashboard 是只读消费者）

### 健康端点
- `GET /_stcore/health` → `200 ok`（Streamlit 内置，docker-compose 用它）

---

## 6. 关键设计决策（Design decisions）

| 决策 | 备选 | 理由 |
|---|---|---|
| 数据 / 渲染分层 | UI 直接写聚合 | 数据层可单测；Streamlit 测试昂贵且脆弱 |
| 直接读 JSONL | dashboard 调 edge / agent HTTP | 单一事件源（rule §3）；simulation/recommendation 在 orchestrator 已经做完 |
| 手动 refresh | `st.autorefresh` 每 5 s | MVP 简化；自动刷新会与 `JsonlEventWriter` 写时锁竞争（Windows 文件锁更严） |
| 坏行 skip 不阻塞 | 整个加载失败 | 操作员宁愿看到 99 条也不要因 1 条坏行什么都看不到（rule §13 graceful） |
| 严重度 0-row 也展示 | 只展示出现过的 | 三档 severity 的 x 轴跨刷新稳定，不会"突然多出一根柱" |
| 无 plotly / matplotlib | `st.bar_chart` / `st.line_chart`（内建 altair） | MVP 不引入额外大依赖（已 pinned altair 6.1） |
| `[MOCK]` 标记保持原样 | 在 UI 隐藏 | rule §12：placeholder 必须在终端用户视野里仍可识别 |

---

## 7. 反例（What NOT to put here）

- ❌ HTTP 调用 edge / agent（dashboard 是事件流的**消费者**，不是产生者）
- ❌ 推理 / RAG / LLM 业务逻辑（属于 `inference/` / `agent/`）
- ❌ 持久化数据库写（dashboard 是只读的）
- ❌ 状态共享 / WebSocket（MVP 不需；polish 阶段加用户设置持久化）

---

## 8. 教学注释

- **为什么数据层不依赖 Streamlit？** 单测能在 5 s 内跑完 22 项；如果数据层
  混了 `st.session_state`，就要起 Streamlit testing harness 才能测——慢 10×
- **为什么 latency 用 `pd.Series.quantile(0.95)` 而不是 `numpy.percentile`？**
  pandas 已经在依赖里，避免引入新 import；二者结果一致
- **为什么 events_to_dataframe 即使 events 为空也返回带列名的 DF？**
  Streamlit 表格在空 DF 时若列名缺失会报错；保留列定义让 UI 稳定渲染
- **为什么 severity 颜色不用 plotly 主题？** Streamlit `st.bar_chart` 不支持
  per-series 配色——polish 阶段用 `alt.Chart` 替换

---

## 9. 与其它模块的接口（Interfaces）

| 上游 | 下游 |
|---|---|
| `orchestrator/event_log.py::JsonlEventWriter` 写 events | — |
| `api/schemas.py::OrchestratorEvent`（共享契约） | — |
| `utils/paths.py::ORCHESTRATOR_DIR` 默认事件路径 | — |

**不依赖**：`api/edge_service.py`、`api/agent_service.py`、`tools/`、`agent/`。

---

## 10. 测试覆盖（Tests）

`tests/unit/test_dashboard_data.py`（23 项）：

| 区域 | 覆盖点 |
|---|---|
| `load_events` | 缺文件 / valid round-trip / 坏行 skip / 空行忽略 |
| `events_to_dataframe` | 空表带列 / 嵌套展开 / error 行 |
| `per_node_summary` | 多节点聚合 / last_seen 取最大时间 / 空表带列 |
| `severity_counts` | 始终 3 行 |
| `fault_class_counts` | 降序 |
| `latency_stats` | 只 edge / 全空 → NaN |
| `get_event_by_id` | 命中 / miss |
| `filter_events` | severity / node / only_with_recommendation / 组合 |
| `severity_over_time` | bucket / 空表带列 / error-only event 跳过 |
| 集成 | JSONL on disk → load → aggregate |
| **`dashboard.app`** | import 不抛错 + `main` 可调用（捕获 wiring bug） |

---

## 11. 性能预算（Perf budget）

| 项 | 目标 | 实测（33 events） |
|---|---|---|
| `load_events` | < 50 ms / 1k events | < 5 ms / 33 events |
| `events_to_dataframe` | < 100 ms / 1k events | < 10 ms / 33 events |
| Streamlit 首屏 | < 2 s | ~1 s（headless 启动 ~3.5 s 是 Streamlit 自身） |
| `/_stcore/health` | < 50 ms | ~5 ms（Streamlit 内置） |

事件量 ≥ 10k 时建议 polish 阶段加 `@st.cache_data`。

---

## 12. 未来扩展（Future work）

- `st.autorefresh(interval=5000)` 自动刷新
- 用 `alt.Chart` 替换 `st.bar_chart` → 严重度配色 / 故障类别堆叠
- 节点详情子页：单节点的 sensor snapshot 历史曲线
- Reasoning trace 从表格变成"卡片流"，每步 expander
- Recommendation 触发"反馈"按钮 → 写回 `data/orchestrator/feedback.jsonl`
  供 agent_eval 学习
- WebSocket 推送 vs 轮询（避免重新读整个 JSONL）
- 多语言（中 / 英）切换
- 用 `httpx` 调 edge / agent `/healthz` 来标识"实时服务状态"

---

## 13. 运行示例（Usage example）

**本地直接跑**

```powershell
# 0. 先把 orchestrator 跑一下，产生 data/orchestrator/events.jsonl
python -m uvicorn api.edge_service:app --port 8000 &
python -m uvicorn api.agent_service:app --port 8001 &
python -m orchestrator --nodes pv2_bess1 --duration 30

# 1. 起 dashboard
streamlit run dashboard/app.py
# → http://127.0.0.1:8501
```

**docker compose**

```powershell
docker compose up --build
# → edge :8000, agent :8001, dashboard :8501, orchestrator 一次性 60 秒
# polish 阶段加 chroma：docker compose --profile polish up
```

---

## 14. 已知限制（Known limitations）

- **手动刷新**：每次想看新事件都要点按钮；polish 阶段加 auto-refresh
- **依赖 JSONL 共享卷**：在 docker-compose 里 dashboard 与 orchestrator 共享
  `./data/orchestrator`；Kubernetes 环境需要 PVC 而不是 host bind mount
- **没有用户认证**：MVP 不做登录；任何能访问 :8501 的人都能看；polish 阶段
  加 `streamlit-authenticator`
- **没有 sensor 历史曲线**：单事件详情里只显示 sensor_snapshot（一拍）；
  polish 阶段加完整窗口曲线
- **`st.bar_chart` 颜色不可配**：严重度三档颜色是 altair 默认；polish 阶段
  换 `alt.Chart` 自定义
- **大事件量未优化**：> 10k 事件时每次刷新会重读全文件；polish 阶段加
  `@st.cache_data` 或增量读取（按 `stat().st_size` cursor）
