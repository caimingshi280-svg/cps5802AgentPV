# `inference/` — 推理后处理与 ONNX Runtime 封装

按规则 §24 的 14 节工程文档。本模块把"模型 logits"翻译成"业务可消费的 Alert"，
并提供 ONNX Runtime 的薄封装。

---

## 1. 模块目的（Purpose）

承担"模型与服务之间"的桥接层，提供两件事：

1. **`postprocess.py`**：纯函数，把 logits 与 system_id 翻译成校验过的
   :class:`api.schemas.Alert`，并按业务策略给出严重度。
2. **`onnx_runner.py`**：薄薄包一层 ONNX Runtime，提供 `predict_window`
   + `benchmark` 两个动作。

---

## 2. 为什么需要这个模块（Why it exists）

- 严重度策略（哪些类是 critical）是**业务逻辑**，必须可独立测试，不能塞到模型里
- ONNX Runtime 的 InferenceSession 创建开销大、metadata 校验也是状态相关，
  需要一个长生命周期对象——但又不能让它绑死 FastAPI（rule §2 单一职责）
- 所有"window → Alert"的代码必须用相同实现，避免 edge_service 与 orchestrator 跑分歧

---

## 3. 架构概览（Architecture）

```text
ONNX 文件                    SensorWindow JSON
   │                                │
   ▼                                ▼
   onnx_runner.OnnxClassifier ──── predict_window() ────┐
                                                          │
                              logits  np.ndarray         │
                                                          ▼
                                            postprocess.logits_to_alert()
                                                          │
                                                          ▼
                                                       Alert (validated)
```

---

## 4. 关键文件（Key files）

| 文件 | 作用 |
|---|---|
| `postprocess.py` | 严重度策略 `severity_for(fault_class, confidence)` + `logits_to_alert(...)` |
| `onnx_runner.py` | `OnnxClassifier`：加载 .onnx + 校验 metadata + 推理 + 基准测试 |

---

## 5. 输入输出契约（Inputs & Outputs）

### `OnnxClassifier`
- 输入：`SensorWindow`（Pydantic 校验通过）
- 输出：`Alert`（含 `fault_class / severity / confidence / sensor_snapshot`）

### `severity_for(fault_class, confidence, *, high_conf_threshold=0.85)`
- 输入：fault_class 字符串 + confidence ∈ [0, 1]
- 输出：`Severity` 枚举（monitor / warning / critical）

---

## 6. 关键设计决策（Design decisions）

| 决策 | 备选 | 理由 |
|---|---|---|
| postprocess 是纯函数 | 写成 OnnxClassifier 的方法 | 可被 quantized inference / 评估脚本 / 仿真回放复用，不绑 ONNX Runtime |
| 严重度策略是 frozenset 常量 | 动态从 yaml 读 | MVP 先 hard-code，rule §27 最小可运行；polish 阶段拆 yaml |
| 未知 fault_class 默认 monitor | 抛异常 | 防止"加新类没改 mapping"被默默升级到 critical（safety-first） |
| Soiling 由 confidence 门控 | 一律 warning | Soiling 长期累积型故障，低置信度时不升级，减少误报 |
| `OnnxClassifier` 校验 metadata 与 schema 一致 | 信任 ONNX 文件 | 防 schema 偏移：模型旧 / 代码新场景下立刻报错（rule §3） |
| benchmark 含 5 次 warmup | 立即测量 | ONNX Runtime 第一次调用有 JIT 编译，warmup 可剥离冷启动 |

---

## 7. 反例（What NOT to put here）

- ❌ HTTP 路由 / FastAPI 装饰器（属于 `api/`）
- ❌ 训练循环、loss、optimizer（属于 `training/`）
- ❌ ONNX 文件的导出（属于 `quantization/`）
- ❌ 仪表盘 UI（属于 `dashboard/`）

---

## 8. 教学注释（Why each piece）

- **为什么严重度不能放到 ONNX 里？** 因为它会随业务策略变化（运维团队可能调整哪些故障算 critical），而模型是固定二进制；策略归 Python 才能在不重训的情况下迭代。
- **为什么用 `frozenset` 而不是 `dict`？** 严重度只查询，不更新；frozenset 让"它是只读集合"在类型层面表达清楚。
- **为什么 sensor_snapshot 取窗口最后一拍？** 给 cloud agent 一个"最近的状态"，比传整窗 60 拍数据节省带宽。

---

## 9. 与其它模块的接口（Interfaces）

| 上游 | 下游 |
|---|---|
| `quantization/onnx_export.py` 产出 .onnx + metadata | `api/edge_service.py` 调 `predict_window` |
| `api/schemas.py` 提供 SensorWindow / Alert / Severity | `evaluation/` 用 `severity_for` 复盘评估 |
| `models/cnn1d.py` 通过 ONNX 间接服务 | `dashboard/` 把 Alert 渲染到 UI |

---

## 10. 测试覆盖（Tests）

| 测试文件 | 数量 | 覆盖点 |
|---|---|---|
| `tests/unit/test_postprocess.py` | 16 | 每条严重度分支 + logits→Alert 端到端 + 错误路径 |
| `tests/unit/test_onnx_runner.py` | 7 | metadata 加载、PV/BESS 路由、形状校验、system_type 拒错、benchmark 输出有限 |
| `tests/integration/test_edge_service.py` | 8 | FastAPI 端 e2e（调用 OnnxClassifier） |

---

## 11. 性能预算（Perf budget）

| 项 | 目标 | 实测（CPU MVP） |
|---|---|---|
| 单样本推理 p99 | < 100 ms | **0.16 ms**（PV）/ **0.15 ms**（BESS） |
| `OnnxClassifier` 启动时 metadata 校验 | < 100 ms | < 30 ms（从磁盘读取 ~180 KB 文件） |

---

## 12. 未来扩展（Future work）

- 加 `inference/batch_runner.py` 支持微批推理（多节点合并请求）
- 加 SHAP 解释器（解释为什么是 critical）
- 加多模型 ensemble（PV 主模型 + 轻量备份模型）

---

## 13. 运行示例（Usage example）

```python
from inference.onnx_runner import OnnxClassifier
from api.schemas import SensorWindow

clf = OnnxClassifier("quantization/artifacts/cnn1d_pv.onnx")
window = SensorWindow(...)  # 来自仿真或真实采集
alert = clf.predict_window(window)
print(alert.fault_class, alert.severity)

stats = clf.benchmark(n=200)
print(stats.to_dict())
```

---

## 14. 已知限制（Known limitations）

- 严重度策略不可热更新；polish 阶段会改成 yaml 配置
- `predict_window` 目前只支持单样本；多请求合并到下层 batch 由 orchestrator 处理
- `OnnxClassifier` 不支持 GPU；rule §17 要求 edge 一律 CPU
