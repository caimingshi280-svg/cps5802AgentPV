# `api/` — HTTP 契约与服务（Schemas + FastAPI 入口）

按规则 §24 的 14 节工程文档。本模块**同时承载两件事**：

1. **`schemas.py`**：所有跨模块共享的 Pydantic 数据契约（rule §3 单一信源）
2. **`edge_service.py` / `agent_service.py`**：FastAPI 服务入口

`schemas.py` 不能放在任何具体服务里，因为 simulation / training / inference /
agent / dashboard / orchestrator 全都要导入它。把它放在 `api/` 是项目的一致约定。

---

## 1. 模块目的（Purpose）

- **schemas.py**：跨模块的数据合同；通过 `extra="forbid"` 严格拒绝未声明字段
- **edge_service.py**：HTTP 入口；把 SensorWindow 路由给 OnnxClassifier，回 Alert
- **agent_service.py**（S08 完成）：HTTP 入口；把 Alert 路由给 ReAct agent，回 Recommendation
- **errors.py**：少量 HTTP 异常处理工具（按需扩展）

---

## 2. 为什么需要这个模块（Why it exists）

- 跨服务通信必须有强类型 schema（rule §11、§13）
- FastAPI 让 Pydantic schema 自动变成 OpenAPI 文档，零冗余
- 每个服务都需要 `/healthz`（rule §22）；统一用 `HealthResponse` 写出
- 错误响应必须结构化（rule §13）；统一用 `ErrorResponse`

---

## 3. 架构概览（Architecture）

```text
                                 ┌────────────────────────────────┐
                                 │       api/schemas.py           │
                                 │ SensorWindow / Alert /         │
                                 │ Recommendation / ReasoningStep │
                                 │ HealthResponse / ErrorResponse │
                                 │ ToolError / RawSample / ...    │
                                 └────────────┬───────────────────┘
                                              │ import
        ┌───────────────────────┬─────────────┼─────────────┬───────────────────┐
        ▼                       ▼             ▼             ▼                   ▼
  simulation/             training/     inference/    agent/              dashboard/
                                              ▲             ▲
                                              │             │
                                ┌─────────────┘             └────────────┐
                                │                                          │
                       api/edge_service.py                       api/agent_service.py
                       (FastAPI :8000)                            (FastAPI :8001)
                       /predict /healthz /metrics                /recommend /healthz
```

---

## 4. 关键文件（Key files）

| 文件 | 作用 |
|---|---|
| `schemas.py` | 全部 Pydantic 模型 + 故障类常量 + 枚举 |
| `errors.py` | HTTP 异常处理小工具 |
| `edge_service.py` | FastAPI 边缘推理服务 |
| `agent_service.py` | （S08）FastAPI 云端 agent 服务 |

---

## 5. 输入输出契约（Inputs & Outputs）

### `POST /predict`（edge_service）
- 输入：`SensorWindow` JSON
- 输出：`Alert` JSON
- 错误：
  - 422 `ValidationError`（字段/形状不合法）
  - 503 `ErrorResponse`（请求的 system_type 模型未加载）

### `GET /healthz`
- 输出：`HealthResponse`（`status: ok | degraded`）

### `GET /metrics`（edge_service）
- 输出：`{"PV": LatencyStats, "BESS": LatencyStats}`，n=50 的合成基准

---

## 6. 关键设计决策（Design decisions）

| 决策 | 备选 | 理由 |
|---|---|---|
| schemas 集中放 `api/schemas.py` | 每个模块自己写 | rule §3：避免重复定义、漂移 |
| 用 FastAPI lifespan | 用 `on_event("startup")` | FastAPI 0.110+ on_event 已 deprecated |
| 缺模型时 `degraded` 而不是 fail-startup | 启动失败 | 让 PV-only 部署 / BESS-only 部署可工作；rule §22 优雅降级 |
| 503 而不是 404 | 404 / 500 | 503 表示"服务可用但某资源暂时缺失"，语义最贴 |
| `/metrics` 跑合成 benchmark | 暴露 Prometheus | MVP 先用最小依赖；polish 阶段加 prometheus_client |
| 422 自动来自 Pydantic | 手写错误处理 | StrictBaseModel 拒额外字段，校验信息已够清晰 |

---

## 7. 反例（What NOT to put here）

- ❌ 业务逻辑（属于 `inference/`、`agent/`）
- ❌ 模型加载实现（属于 `inference/onnx_runner.py`）
- ❌ 数据库 / 持久化（不在本项目范围）
- ❌ HTML 模板 / UI（属于 `dashboard/`）

---

## 8. 教学注释

- **为什么 schemas.py 在 `api/` 里而不是顶级 `schemas/`？** 因为它定义的是"跨进程通信的契约"，HTTP API 自然消费它；放 `api/` 让"调 API 必先看 schemas"成为肌肉记忆。
- **`StrictBaseModel.extra="forbid"` 的好处是什么？** 防止"客户端多发了字段没人发现"——这种 bug 在 production 才会爆。校验阶段就 422 拒掉。
- **为什么 `/predict` 不接 batch？** MVP 单样本即可；orchestrator 从多个节点收来的请求由它自己去并发，不必下推到 edge。

---

## 9. 与其它模块的接口（Interfaces）

| 上游 | 下游 |
|---|---|
| 任何模块 import schemas | edge_service 调 `inference.onnx_runner.OnnxClassifier` |
| edge_service 启动时读 `configs/settings.py::artifacts_dir` | agent_service（S08）调 `agent/workflows/react.py` |
| orchestrator 调 `POST /predict` | dashboard 调 `GET /metrics` |

---

## 10. 测试覆盖（Tests）

| 测试文件 | 数量 | 覆盖点 |
|---|---|---|
| `tests/unit/test_schemas.py` | 21 | 每条 schema 的字段校验、严格模式、cross-field 一致性 |
| `tests/integration/test_edge_service.py` | 8 | /healthz / /predict（PV+BESS） / /metrics / 422 / 503 / degraded 模式 |

---

## 11. 性能预算（Perf budget）

| 项 | 目标 | 当前 |
|---|---|---|
| `/predict` p99（含 HTTP 往返） | < 100 ms | TestClient 内 ~5 ms；网络上预计 < 20 ms |
| 启动时间 | < 5 s | 读 2 个 onnx + 校验 metadata < 1 s |

---

## 12. 未来扩展（Future work）

- `agent_service.py` `/recommend`（S08）
- gRPC 协议备选（如果 HTTP 出现瓶颈）
- WebSocket 流式推理（多帧实时）
- prometheus_client 替换合成 `/metrics`

---

## 13. 运行示例（Usage example）

```bash
# 启 edge service
python -m uvicorn api.edge_service:app --host 0.0.0.0 --port 8000

# 调用
curl http://127.0.0.1:8000/healthz
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @window.json
curl http://127.0.0.1:8000/metrics

# OpenAPI 文档（Swagger UI）
open http://127.0.0.1:8000/docs
```

---

## 14. 已知限制（Known limitations）

- 没有鉴权 / 限流（MVP 内网部署，rule §27 最小可运行）
- `edge_service` 启动时模型路径硬编码到 `cnn1d_{pv,bess}.onnx`；polish 时让 settings 配置
- `/metrics` 每次都重跑 benchmark；polish 时缓存
