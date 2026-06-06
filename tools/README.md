# `tools/` — Agent 工具集（Component 4）

按规则 §24 的 14 节工程文档。本模块定义 ReAct agent 可调用的工具——每个工具
都遵守严格的 typed I/O / timeout / 结构化错误契约（rule §11）。

---

## 1. 模块目的（Purpose）

把"agent 想做的具体动作"包装成统一接口的可执行单元。LLM 决定调哪个工具、传什么
参数；工具本身负责执行 + 校验 + 错误处理。

---

## 2. 为什么需要这个模块（Why it exists）

- 作业 §4.4 要求 ≥3 个工具且每个有 typed I/O；MVP 实现了 4 个
- 工具与 agent 编排解耦后，可以单独单测、单独替换 mock 后端
- 失败必须返回结构化 `ToolError` 而不是抛异常——agent 才能 reflect 处理

---

## 3. 架构概览（Architecture）

```text
                    Tool (ABC)  ← tools/base.py
                    ├── input_model: Pydantic
                    ├── output_model: Pydantic
                    ├── timeout_s: float
                    ├── _run(InputT) → OutputT       (子类实现)
                    └── __call__(dict) → dict        (统一调度 + 校验 + 超时)
                            │
        ┌───────────────────┼───────────────────┐──────────────────┐
        ▼                   ▼                   ▼                  ▼
RetrieveKnowledgeTool  SystemHistoryTool  EstimateRulTool   EscalateAlertTool
  → rag.Retriever      → mock backend     → rule-based       → audit log
                                            policy table
```

---

## 4. 关键文件（Key files）

| 文件 | 后端 | 用途 |
|---|---|---|
| `base.py` | — | `Tool` 抽象基类（已存在，S04 写） |
| `retrieve_knowledge.py` | `rag.Retriever` | 按 query 返回 top-k 文档 chunk |
| `system_history.py` | **mock**（确定性） | 给定 system_id 返回最近 N 小时的告警历史 |
| `estimate_rul.py` | rule_based | 按 fault_class + severity 估剩余寿命（粗粒度） |
| `escalate_alert.py` | audit_log | 记录一次升级到结构化日志 |

---

## 5. 输入输出契约（Inputs & Outputs）

每个工具的输入 / 输出 Pydantic 模型：

| 工具 | 输入 | 输出 |
|---|---|---|
| `retrieve_knowledge` | `query: str`, `top_k: int=3` | docs[], source_titles[] |
| `system_history` | `system_id`, `system_type`, `hours: 24` | entries[], n_critical/warning/monitor |
| `estimate_rul` | `system_id, system_type, fault_class, severity, confidence` | rul_days_estimate/lower/upper, requires_immediate_action |
| `escalate_alert` | `system_id, system_type, fault_class, severity, urgency, summary` | escalated, escalation_id, channel |

**通用错误**：每个工具失败时返回 `ToolError` envelope（`error_code: VALIDATION |
TIMEOUT | INTERNAL | NOT_FOUND`），从不 raise。

---

## 6. 关键设计决策（Design decisions）

| 决策 | 备选 | 理由 |
|---|---|---|
| 工具基类 + 装饰器（继承） | 函数式 + decorator | 多个工具有共享配置（timeout / max_retries / I/O 模型）；用类清晰 |
| **MVP 用 mock**（system_history） | 接 InfluxDB | rule §27：先打通端到端；polish 阶段换数据源 |
| Mock 输出含 `[MOCK]` 标记 | 让用户事后查看 backend 字段 | rule §12：人类阅读时一眼识别 |
| `estimate_rul` 用 rule-based 表 | 训练 survival model | MVP 表足够给出"是否立即处理"信号；polish 阶段升级到模型 |
| `escalate_alert` 写 audit_log | 真发 PagerDuty | 测试不能误发真告警；polish 阶段加可配置 backend |
| 输入校验失败 = `ToolError(VALIDATION)` | raise | rule §11：工具从不抛 |

---

## 7. 反例（What NOT to put here）

- ❌ ReAct 循环（属于 `agent/workflows/react.py`）
- ❌ LLM client（属于 `agent/orchestration/llm_client.py`）
- ❌ FastAPI 路由（属于 `api/`）
- ❌ 业务持久化逻辑（应该被工具调用，但实现在数据层 / 外部服务）

---

## 8. 教学注释

- **为什么工具不抛异常？** 因为 ReAct 的 reflect 步骤要看"哪些工具失败了、为什么"
  来决定下一步——如果工具直接 raise，整个 agent run 终止，无法降级
- **为什么 system_history 选确定性 mock？** 测试需要可重现：相同 input → 相同
  output。用 hashlib.sha256 + 模板池给定 system_id 选模板，时间戳由 now() 决定
  但故障序列不变
- **为什么 estimate_rul 给宽置信带？** 因为 rule-based 不是真模型；rule §12
  要求 placeholder 必须诚实，宽带反映"我们其实不太确定"

---

## 9. 与其它模块的接口（Interfaces）

| 上游 | 下游 |
|---|---|
| `rag.Retriever` 提供检索能力 | `agent/workflows/react.py` 注册并按 plan 执行 |
| `api.schemas` 提供 Severity / SystemType / Urgency | `api/agent_service.py` 启动时构造 |

---

## 10. 测试覆盖（Tests）

`tests/unit/test_tools.py`（12 项）：

- 每个工具至少 2 项：成功路径 + validation/internal 错误路径
- 确定性 mock 必须返回相同序列（system_history 重复调用）
- estimate_rul：critical=0 / normal=long-horizon / 高 confidence 缩窄区间
- escalate_alert：返回 32 字符 hex id

---

## 11. 性能预算（Perf budget）

| 工具 | timeout | 实测 |
|---|---|---|
| `retrieve_knowledge` | 5 s | ~5 ms |
| `system_history` | 3 s | <1 ms |
| `estimate_rul` | 2 s | <1 ms |
| `escalate_alert` | 3 s | <1 ms |

---

## 12. 未来扩展（Future work）

- `system_history` 接 InfluxDB / TimescaleDB
- `estimate_rul` 训练 cox-PH 或 GBT survival 模型，per-asset
- `escalate_alert` 加 PagerDuty / Slack / SMS 后端
- 加 `system_metadata` 工具（asset 静态信息：品牌、安装日期、保修截止）
- 加 `weather_lookup` 工具（光照 / 温度预报，辅助下一周风险评估）

---

## 13. 运行示例（Usage example）

```python
import asyncio
from tools.estimate_rul import EstimateRulTool

async def main():
    tool = EstimateRulTool()
    out = await tool({
        "system_id": "PV_007",
        "system_type": "PV",
        "fault_class": "Partial_shading",
        "severity": "warning",
        "confidence": 0.82,
    })
    print(out)

asyncio.run(main())
```

---

## 14. 已知限制（Known limitations）

- 三个工具是 mock / rule-based；polish 阶段需要把每个升级到真后端
- `retrieve_knowledge` 的 top_k 上限 20；ReAct 同一轮多次调用代价低，
  但单次扯回大量文档会膨胀 prompt
- 工具间没有依赖图（一个工具的输出直接 feed 给另一个工具）；MVP 由 LLM client
  在 plan 阶段静态决定调用顺序
