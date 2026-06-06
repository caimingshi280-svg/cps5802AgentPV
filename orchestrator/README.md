# `orchestrator/` — 多节点设备模拟与三段链路编排（Component 7）

按规则 §24 的 14 节工程文档。本模块是 AgentPV 的**运行时联调层**——把
"`simulator → edge /predict → agent /recommend`"这条三段链路在多个虚拟资产上
并发跑起来，并把每一次执行落成可审计的 JSONL 事件流，供 dashboard 和 agent
benchmark 消费。

---

## 1. 模块目的（Purpose）

让作业 §4.7 的"≥3 个并发节点 + 时序生成 + 边端推理 + 云端 agent 反馈"在
MVP 阶段就跑得起来；同时为 dashboard 提供唯一的事件源，避免 dashboard 自己
再实现一份模拟逻辑。

---

## 2. 为什么需要这个模块（Why it exists）

- 作业明文要求多节点并发、覆盖正常 + 故障比例
- agent benchmark（Component 5）需要"已发生的事件流"做离线评估，事件流必须
  有 ground-truth 标签和完整的 alert / recommendation
- dashboard 不应直接调 simulator + edge + agent 三遍——它只应该读已发生事件
  并显示

---

## 3. 架构概览（Architecture）

```text
        NodeConfig × N
              │
              ▼
   ┌──────────────────────────────┐
   │     Orchestrator             │
   │  ┌──────────────────────┐    │
   │  │  NodeRunner #1       │    │
   │  │  ┌──────────────┐    │    │
   │  │  │ PVSim/BESSSim│    │    │
   │  │  └──────┬───────┘    │    │
   │  │         │ SensorWindow    │
   │  │         ▼            │    │
   │  │  EdgeClient.predict ─┼───────► edge_service /predict ──► Alert
   │  │         │            │    │
   │  │  Severity == warn|crit?   │
   │  │         │ yes        │    │
   │  │         ▼            │    │
   │  │  AgentClient.recommend ──────► agent_service /recommend ─► Rec
   │  │         │            │    │
   │  │         ▼            │    │
   │  │  OrchestratorEvent ──┼───────► JsonlEventWriter
   │  └──────────────────────┘    │
   │  ┌──────────────────────┐    │
   │  │  NodeRunner #2 ...   │    │
   │  └──────────────────────┘    │
   │  asyncio.gather 等齐 N 个     │
   └──────────────────────────────┘
              │
              ▼
   data/orchestrator/events.jsonl
   （dashboard tail 这个文件即可）
```

---

## 4. 关键文件（Key files）

| 文件 | 作用 |
|---|---|
| `node_simulator.py` | `NodeConfig`（pydantic）+ `NodeRunner`（每节点 1 个 asyncio loop） |
| `clients.py` | `EdgeClient` + `AgentClient`，都返回 `T \| ClientError`（rule §13 graceful） |
| `event_log.py` | `JsonlEventWriter` append-only + `summarize()` |
| `orchestrator.py` | `Orchestrator`：N 节点 gather + duration 控制 + summary 视图 |
| `__main__.py` | CLI：`--nodes minimal\|pv2_bess1`，`--duration`，`--out` |

---

## 5. 输入输出契约（Inputs & Outputs）

### 输入
- 一组 `NodeConfig`（运行时配置，**不**进 `api.schemas`，因为只有 orchestrator
  自己用）
- 两个 base URL（edge / agent）
- 一个 `JSONL` 输出路径

### 输出
- `data/orchestrator/events.jsonl`：每行一条 `OrchestratorEvent`（rule §3，
  `api.schemas.OrchestratorEvent`，dashboard / agent_eval 共用）
- `Orchestrator.summary()`：dict 快照（节点级 + 全局）
- 控制台日志（结构化，含 `node_id` / `step` / `severity` / `edge_ms` / `agent_ms`）

---

## 6. 关键设计决策（Design decisions）

| 决策 | 备选 | 理由 |
|---|---|---|
| `asyncio` + 单 loop | 多进程 | httpx async 天然适配；测试可用 ASGITransport 不起 socket |
| 节点共享 1 个 `JsonlEventWriter` + threading.Lock | 各节点写独立文件 | 单一文件 dashboard 易消费；append-only 单行写在 PIPE_BUF 内是原子的 |
| 触发 agent 的策略硬编码：severity ∈ {warning, critical} | 配置项 | 与 `inference/postprocess.py` 业务规则保持一致；MVP 暂不可调 |
| 启动 truncate 一次 | 始终追加 | 每次跑都重置，保证可重现（rule §6）；polish 阶段可改成按时间戳轮转 |
| `OrchestratorEvent.error` 与 `alert` **可同时存在** | 互斥 | 例：edge 成功（alert）但 agent 失败（error），仍然要保留 alert |
| 单步异常用 `try/except + log.exception` 包住 | 让循环挂掉 | 一个节点崩溃不能拖垮其它节点（rule §13） |
| schema 校验在客户端做（`Alert.model_validate`） | 信任服务端 | 服务也许返 200 但 body 损坏；rule §3：契约一律校验 |
| `NodeConfig` 留在 `orchestrator/` | 放进 `api/schemas.py` | 它是**运行时配置**不是跨服务契约；rule §3 的 carve-out |

---

## 7. 反例（What NOT to put here）

- ❌ Edge 推理 / RAG / LLM 逻辑（属于 `inference/` / `rag/` / `agent/`）
- ❌ Streamlit UI（属于 `dashboard/`）
- ❌ 数据集生成的 stratified split（属于 `simulation/`）
- ❌ Agent benchmark 评分（属于 `agent_eval/`）

---

## 8. 教学注释

- **为什么 NodeRunner 每步都重建 SensorWindow？** 因为 SensorWindow 是
  immutable Pydantic 模型；每步的 timestamp / values 都不同，复用没意义
- **为什么 fault_probability 是节点级而非全局？** 不同资产有不同故障率
  （PV 设备老化 vs 新部署 BESS）；polish 阶段会把概率从配置文件读取
- **为什么记录 `ground_truth_label`？** 给 agent benchmark 算 confusion matrix
  和正确率；在真实部署里这个字段会被替换成 `null`
- **为什么 timeout=10s？** edge 实测 p99 0.16 ms，agent ~30 ms；10 s 是给真
  LLM 后端预留的余量。MVP mock 永远秒回

---

## 9. 与其它模块的接口（Interfaces）

| 上游 | 下游 |
|---|---|
| `simulation.PVSimulator` / `BatterySimulator` | `dashboard/app.py` 读 `events.jsonl` |
| `simulation.fault_injector.inject_fault` | `agent_eval/` polish 阶段做离线评估 |
| `api.edge_service` 提供 `/predict` | — |
| `api.agent_service` 提供 `/recommend` | — |
| `api.schemas.{SensorWindow, Alert, Recommendation, OrchestratorEvent}` | — |

---

## 10. 测试覆盖（Tests）

| 测试文件 | 数量 | 覆盖点 |
|---|---|---|
| `tests/unit/test_orchestrator.py` | 17 | JsonlEventWriter / summarize / Edge & Agent client / NodeRunner 4 路径（monitor / warning / edge-error / agent-error） / 决定性 / Orchestrator duration & 拒空 / OrchestratorEvent 三个 validator |
| `tests/integration/test_orchestrator_e2e.py` | 1 | ASGI in-process 端到端：2 节点 0.6 s，写出 ≥2 events，至少 1 alert，不出 `*_Normal`（fault_probability=1.0） |

集成测试通过 `httpx.ASGITransport` + `app.router.lifespan_context` 在同一
进程里挂载 edge_service 和 agent_service，**不**起 socket。无需 ONNX 时自动
skip。

---

## 11. 性能预算（Perf budget）

实测（2026-05-09，3 节点 / 15 s / pv2_bess1 catalogue）：

| 维度 | 数据 |
|---|---|
| 总事件数 | 33（pv-001=15 + pv-002=10 + bess-001=8） |
| edge predict p50 | ~5 ms（含 HTTP 序列化，远 < 100 ms budget） |
| agent recommend p50 | ~30 ms（mock LLM；真 LLM 后预期 1–3 s） |
| 错误数 | 0 |
| 严重度分布 | monitor 25 / warning 4 / critical 4（与 fault_probability 一致） |
| 故障类别覆盖 | 8 类（PV/BESS Normal + Inverter / String / Bypass / Thermal / Degradation / Partial_shading） |

---

## 12. 未来扩展（Future work）

- `configs/orchestrator.yaml` 可加载节点表（rule §27 的 polish 项）
- 多进程模式（每节点 1 个 worker），支持 ≥10 节点 24h 负载测试（作业 §4.7）
- Prometheus exporter 替代 JSONL（生产环境）
- Backpressure：edge 503 时 NodeRunner 应退避而非每秒重打
- 注入"延迟尖峰"测试：在 SensorWindow 中模拟通信抖动
- 与 dashboard 双向：dashboard UI 上手动触发某节点 inject 特定故障

---

## 13. 运行示例（Usage example）

```powershell
# 起两个服务
python -m uvicorn api.edge_service:app --port 8000 &
python -m uvicorn api.agent_service:app --port 8001 &

# 起 orchestrator，跑 30 秒
python -m orchestrator --nodes pv2_bess1 --duration 30 --out data/orchestrator/events.jsonl

# 查看事件流（JSONL 一行一个 OrchestratorEvent）
Get-Content data/orchestrator/events.jsonl | Select-Object -First 5
```

CLI summary 输出示例（节选）：

```json
{
  "n_nodes": 3,
  "global": {
    "n_total": 33,
    "n_alerts": 33,
    "n_recommendations": 8,
    "n_errors": 0,
    "by_severity": {"monitor": 25, "warning": 4, "critical": 4},
    "by_fault_class": {
      "Inverter_fault": 1, "Bypass_diode_fault": 3, "Thermal_anomaly": 2, ...
    }
  }
}
```

---

## 14. 已知限制（Known limitations）

- **catalogue 硬编码在 `__main__.py`**：MVP 没有 YAML 加载；polish 阶段补
- **每次启动 truncate** `events.jsonl`：跨次运行不保留历史；polish 阶段按
  时间戳轮转
- **edge / agent 客户端无重试**：`httpx.HTTPError` 直接降级为 `ClientError`，
  不重发。polish 阶段加 `tenacity` exponential backoff
- **`step()` 不并发同节点**：一个节点的下一步必须等上一步完成；`period_seconds`
  是**最小**间隔，慢请求会拖累节奏
- **集成测试只跑 0.6 s**：CI 时间敏感；polish 阶段加 longer-haul 测试到独立
  marker
- **windowing 没溢写**：每节点的 simulator 重建一次窗口，没有"移动窗口" 实
  时模拟（即真实部署里 t→t+1 只新增 1 个采样点）；polish 阶段补
