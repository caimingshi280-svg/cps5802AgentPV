# `agent/` — Cloud agent 编排与 ReAct 推理（Component 4）

按规则 §24 的 14 节工程文档。本模块是云端 agent 的"大脑"——把 LLM 客户端、
工具集、prompt 模板编排成一个可重复执行的 ReAct 循环。

---

## 1. 模块目的（Purpose）

接收一个 :class:`api.schemas.Alert`，按 ReAct 范式（Observe → Reason → Act →
Reflect → Report）调用工具，最终输出一个**校验过的 :class:`api.schemas.Recommendation`**。

---

## 2. 为什么需要这个模块（Why it exists）

- 作业 §4.5 要求 ReAct 循环 + 5 阶段 + 至少 3 个工具调用
- 输出必须是结构化 Recommendation（不是 free-form 文本）——agent 必须把 LLM
  的 raw output 转译成 schema
- 推理 trace 必须可审计——每一步都要记录"我观察到什么、为什么决定调这个工具、
  工具说了什么、最终怎么综合"

---

## 3. 架构概览（Architecture）

```text
agent/
├── workflows/
│   └── react.py           ← ReActAgent.run(alert) → Recommendation
├── orchestration/
│   └── llm_client.py      ← LlmClient ABC + MockLlmClient (MVP)
├── prompts/               ← polish 阶段从 rag.prompting 拆分到这里
├── reasoning/             ← polish 阶段：复杂多轮 reasoning state
└── memory/                ← polish 阶段：跨 alert 的 agent 记忆

调用链：

    Alert
      │
      ▼
  ReActAgent.run()
      │
      ├── observe   (logging only)
      │
      ├── reason    ── llm.plan_tools(alert) ──→ list[ToolCall]
      │
      ├── act      ── for ToolCall: tools[name](args) ──→ result dict
      │                  │
      │                  └── 工具失败时返回 ToolError envelope，不 raise
      │
      ├── reflect   (统计成功 / 失败工具数)
      │
      └── report   ── llm.synthesize_recommendation() ──→ (action, confidence)
                                                             │
                                                             ▼
                                                       Recommendation
```

---

## 4. 关键文件（Key files）

| 文件 | 作用 |
|---|---|
| `workflows/react.py` | `ReActAgent` + `ReActConfig`；五阶段循环 |
| `orchestration/llm_client.py` | `LlmClient` ABC + `MockLlmClient` + `ToolCall` + `urgency_for_severity` |
| `prompts/` | （polish）单独的 .j2 prompt 文件 |
| `reasoning/` | （polish）多轮 thought / reflection 抽象 |
| `memory/` | （polish）短期 / 长期 memory（跨 alert） |

---

## 5. 输入输出契约（Inputs & Outputs）

### `ReActAgent.run(alert: Alert) -> Recommendation`
- 输入：`Alert`（来自 edge service 或仿真）
- 输出：`Recommendation`，含：
    - `recommended_action`：人类可读建议（mock 后端 `[MOCK]` 前缀）
    - `urgency`：`Severity → Urgency` 固定映射
    - `reasoning_trace`：≥5 个 `ReasoningStep`（observe / reason / act* / reflect / report）
    - `knowledge_sources`：来自 retrieve_knowledge 的去重 title 列表
    - `confidence`：低（无证据） / 中（默认） / 高（critical + 有引用）

---

## 6. 关键设计决策（Design decisions）

| 决策 | 备选 | 理由 |
|---|---|---|
| ReAct 5 阶段固定 | 完全自由 ReAct（LLM 自决定何时停） | MVP 用 mock 客户端，行为必须可预测；polish 阶段切真 LLM 时再放开 |
| LLM client 抽象 + Mock | 直接调 **Ollama**（本地） | 无 Ollama 时可在 `test` 环境用 mock；CI 不需要外网 |
| 三档 confidence 策略 | LLM 自报 | mock 后端不能 honest 自报；策略明确：无证据=LOW，critical+证据=HIGH，其余=MEDIUM |
| 工具调用上限 `MAX_TOOL_CALLS=6` | 无限制 | 防止 LLM "stuck loop"；polish 阶段 LLM 异常时这是最后保护 |
| 未注册工具 = skip 而不是 raise | 直接 raise | 让 reflect 步骤看到 "skipped — unknown tool" 反馈，agent run 不死 |
| 严重度 → 紧急度固定映射 | LLM 自决定 | 业务规则不应被 LLM 改写；保留 `urgency_for_severity` 单一信源 |

---

## 7. 反例（What NOT to put here）

- ❌ HTTP 路由（属于 `api/agent_service.py`）
- ❌ RAG 索引构建（属于 `rag/`）
- ❌ 工具实现细节（属于 `tools/`）
- ❌ 模型推理（属于 `inference/`）

---

## 8. 教学注释

- **为什么 mock LLM 也要写完整的 plan / synthesize？** 因为 polish 阶段的真
  LLM client 实现同样的接口；workflow 一行不改即可切换后端
- **为什么 `[MOCK]` 前缀必须在最终 recommendation 里？** rule §12：placeholder
  必须明确标注；下游 dashboard / 报告里一眼能识别 mock 输出
- **为什么 reasoning trace 是结构化的？** 自由文本不能被 LLM-as-judge 评估；
  Component 5 的 agent benchmark 需要 grep 出哪一步用了什么工具

---

## 9. 与其它模块的接口（Interfaces）

| 上游 | 下游 |
|---|---|
| `tools/*` 提供 4 个工具 | `api/agent_service.py` 调 ReActAgent.run() |
| `rag/retrieval.py` 提供 Retriever（间接，通过 retrieve_knowledge tool） | `agent_eval/` polish 阶段评估 agent 输出 |
| `api.schemas` 提供 Alert / Recommendation / ReasoningStep | `dashboard/` 渲染 reasoning trace |

---

## 10. 测试覆盖（Tests）

`tests/unit/test_react.py`（14 项）：

- urgency 映射 3 项（每档 severity）
- MockLlmClient：plan 包括 retrieve / critical 加 escalate / `[MOCK]` 前缀 /
  confidence 三档分支
- ReActAgent：返回合规 Recommendation / warning 不 escalate / 阶段顺序 /
  未知工具 graceful / 空工具 reject / max_tool_calls 截断

---

## 11. 性能预算（Perf budget）

| 项 | 目标 | 实测（mock LLM） |
|---|---|---|
| 单次 `run()` 端到端 | < 200 ms（mock）/ < 5 s（真 LLM） | ~30 ms |
| reasoning trace step 数 | 5–8（warning）/ 8（critical） | ✓ |
| HTTP `/recommend` p99 | < 5 s（含网络） | < 100 ms（TestClient） |

---

## 12. 未来扩展（Future work）

- `agent/orchestration/remote_llm.py`：`OllamaChatLlmClient`
- `agent/prompts/` 拆出独立的 .j2 模板（plan / synthesize / reflect 各一份）
- `agent/reasoning/` 加 self-consistency / chain-of-verification
- `agent/memory/` 跨 alert 的 system-level 上下文（系统 X 这周已经报了 3 次）
- 自适应 plan：第一轮 act 失败时由 LLM 决定下一轮调什么

---

## 13. 运行示例（Usage example）

```python
import asyncio
from agent.workflows.react import ReActAgent
from agent.orchestration.llm_client import build_llm_client
from tools.retrieve_knowledge import build_default_tool as build_retrieve
from tools.system_history import SystemHistoryTool
from tools.estimate_rul import EstimateRulTool
from tools.escalate_alert import EscalateAlertTool
from utils.paths import KB_DOCS_DIR

async def main():
    retrieve = build_retrieve(KB_DOCS_DIR)
    agent = ReActAgent(
        tools={
            retrieve.name: retrieve,
            "system_history": SystemHistoryTool(),
            "estimate_rul": EstimateRulTool(),
            "escalate_alert": EscalateAlertTool(),
        },
        llm=build_llm_client("mock"),
    )
    rec = await agent.run(alert)  # alert 来自 edge service
    print(rec.model_dump_json(indent=2))

asyncio.run(main())
```

---

## 14. 已知限制（Known limitations）

- MVP 默认 mock；`dev` 配置默认 **Ollama** 后为「真 ReAct」推理路径
- mock plan 是静态的（按 severity 分支），不是 LLM 推理结果——polish 后 LLM
  会真自己决定调什么、调多少次
- 没有 retry / fallback 链路：第一步工具失败时 ReAct 不会重 plan；polish 阶段加
- 没有 long-context summarization；polish 阶段如果 reasoning_trace 太长会被截断
