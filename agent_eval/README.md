# `agent_eval/` — LLM Agent 离线评测（Component 5，约 10% 评分）

按规则 §24 的 14 节工程文档。本目录实现**可复现**的 agent benchmark：≥30 条结构化场景、**启发式 rubric**（始终可跑）、可选 **LLM-as-judge**（需 API Key）、以及 **工具注册表消融**（从 `ReActAgent` 工具字典中移除指定工具，观察 mock 计划中的 skip 行为）。

---

## 1. 模块目的（Purpose）

满足作业 Component 5 要求：

- **≥30 个测试场景**（当前默认 **33** 条，`ambiguous` 标签 ≥5）
- **LLM-as-judge**：对每条 `Recommendation` 输出四维 1–5 分（正确性 / 可执行性 / 可解释性 / 安全性）+ 短 rationale
- **工具调用消融**：`full` / `no_retrieve_knowledge` / `no_system_history` / `no_estimate_rul` / `no_escalate_alert` / `no_reasoning_trace`（后者保留工具但 **省略可审计 `reasoning_trace`**，用于 C5「No Reasoning Trace」对照）

---

## 2. 为什么需要这个模块（Why it exists）

- `agent/` 与 `api/agent_service.py` 解决**在线推理**；`agent_eval/` 解决**离线证据链**——报告里必须能回答「多少场景通过、去掉哪个工具后掉多少分」。
- 规则 §6：**禁止伪造**外部 LLM 分数。无 Key 时 judge 明确标记 `skipped`，**启发式分数**仍可 100% 复现（CI 友好）。

---

## 3. 架构概览（Architecture）

```text
benchmark.json (或内置 default_benchmark_scenarios)
        │
        ▼
load_benchmark_json ──► list[BenchmarkScenario]
        │
        ▼
run_benchmark (async)
   ┌────┴────┐
   │  对每个 ablation:
   │    build_benchmark_agent(disabled_tools=…)
   │         └── ReActAgent.run(alert, strip_reasoning_trace=…)
   │                 ▼
   │         score_heuristic(rec, expected)
   │                 ▼
   │         maybe_judge_sync (可选 HTTP)
   └────► BenchmarkRunSummary → JSON + Markdown
```

---

## 4. 关键文件（Key files）

| 文件 | 作用 |
|---|---|
| `benchmark.json` | 33 条场景（可由 `--write-default-benchmark` 再生） |
| `scenarios.py` | `BenchmarkScenario` / `ExpectedOutcome` / 默认场景生成器 |
| `heuristic_rubric.py` | 0–1 启发式：urgency / 关键词 / 禁用语 / 知识引用数量 |
| `llm_judge.py` | OpenAI-compatible Chat Completions + JSON 四维评分 |
| `wiring.py` | 与 `agent_service` 对齐的 `ReActAgent` 装配 + 消融 |
| `runner.py` | `run_benchmark` / `ABLATION_DISABLED_MAP` / CLI 入口 |
| `__main__.py` | `python -m agent_eval` |
| `results/` | 运行产物目录（`last_run.json` 等，默认 gitignore 内容） |

---

## 5. 输入输出契约（Inputs & Outputs）

### 输入

- `rag/knowledge_base/documents/*.md`（与线上一致；缺目录则 `run_benchmark` 抛 `FileNotFoundError`）
- `agent_eval/benchmark.json`（可选；缺省走 `default_benchmark_scenarios()`）

### 输出

- `agent_eval/results/last_run.json`（默认）：全量逐条记录 + `mean_heuristic` + LLM judge 统计
- `reports/agent_eval_last_run.md`（默认）：人类可读摘要 + 启发式失败列表

---

## 6. 关键设计决策（Design decisions）

| 决策 | 备选 | 理由 |
|---|---|---|
| 启发式 rubric 永远执行 | 仅 LLM 判分 | CI 无网可跑；符合 rule §6 |
| LLM judge 走 OpenAI-compatible | 绑死单一供应商 | 支持自建兼容网关 / Azure OpenAI |
| 消融 = **从注册表删工具** | 改 Mock LLM 计划 | 保持「计划不变、执行缺工具」的真实边缘故障形态 |
| 场景用 Pydantic 强校验 | 裸 dict | 与 `api/schemas.py` 风格一致 |
| `stakes` / `tags` 元数据 | 无 | 给 judge prompt 上下文；`ambiguous` 计数作业要求 |

---

## 7. 反例（What NOT to put here）

- ❌ 训练 CNN / ONNX（`training/`、`quantization/`）
- ❌ 边缘推理服务（`api/edge_service.py`）
- ❌ Streamlit UI（`dashboard/`）

---

## 8. 教学注释

- **为什么无 API Key 不算「假 judge」？** 因为 JSON 里 `llm_judge` 为 `null` 且 `judge_skip_reason` 写明 `AGENTPV_JUDGE_API_KEY unset`，与「编造的 4.8 分」有本质区别。
- **为什么启发式均分能到 1.0？** 当前 oracle 是为 `MockLlmClient` 量身定做的关键词集合；换成真实 LLM 后应放宽 oracle 或增加软匹配。

---

## 9. 与其它模块的接口（Interfaces）

| 上游 | 下游 |
|---|---|
| `api.schemas.Alert` / `Recommendation` | 本模块 |
| `agent.workflows.react.ReActAgent` | 本模块 |
| `tools/*` | 经 `wiring.build_benchmark_agent` |
| `configs.settings.knowledge_base_dir` | CLI 默认 KB 路径 |

---

## 10. 测试覆盖（Tests）

`tests/unit/test_agent_eval.py`（7 项）：

- 默认场景数 ≥30、`ambiguous` ≥5
- `benchmark.json` 往返加载
- 启发式满分 / 禁用语失败
- 未知消融工具名抛错
- `ABLATION_DISABLED_MAP` ⊆ `ALL_TOOL_NAMES`
- 异步 smoke：`run_benchmark` 单 ablation、`mean_heuristic≈1.0`（mock 对齐 oracle）

---

## 11. 性能预算（Perf budget）

| 项 | 目标 | 当前（本机粗测） |
|---|---|---|
| 33 场景 ×1 ablation | < 30 s | ~4 s |
| 33×5 ablation + 无 LLM judge | < 120 s | ~20 s |
| 单条 LLM judge | < 60 s | 取决于网络（超时 60 s） |

---

## 12. 未来扩展（Future work）

- 接入真实 `LlmClient`（**Ollama**）跑「模型 A vs 模型 B」双 judge
- 场景扩展至 ≥50 + 自动从 `orchestrator` JSONL 回放真实事件
- LLM judge 改用结构化输出 schema + 自动重试解析失败

---

## 13. 运行示例（Usage example）

```powershell
# 1) 再生 benchmark.json（可选）
python -m agent_eval --write-default-benchmark

# 2) 全量消融 + 启发式-only（CI 推荐）
python -m agent_eval `
  --ablations full no_retrieve_knowledge no_system_history no_estimate_rul no_escalate_alert no_reasoning_trace `
  --no-llm-judge
```

```powershell
# 3a) LLM-as-judge 走本机 Ollama（无需 OpenAI API key；可不设 AGENTPV_JUDGE_API_KEY）
$env:AGENTPV_JUDGE_API_BASE = "http://127.0.0.1:11434/v1"
$env:AGENTPV_JUDGE_MODEL = "llama3.2:latest"   # 与 ollama list 一致
$env:APP_ENV = "dev"
python -m agent_eval --ablations full --llm-backend ollama `
  --out-json agent_eval/results/with_judge.json
```

```powershell
# 3b) LLM-as-judge 走云端 OpenAI 兼容接口（需 key）
$env:AGENTPV_JUDGE_API_KEY = "<your-key>"
$env:AGENTPV_JUDGE_MODEL = "gpt-4o-mini"
python -m agent_eval --ablations full --llm-backend ollama
```

环境变量：

| 变量 | 含义 |
|---|---|
| `AGENTPV_JUDGE_API_BASE` | 默认 `https://api.openai.com/v1`；本机 Ollama 填 `http://127.0.0.1:11434/v1` |
| `AGENTPV_JUDGE_API_KEY` | 云端必填；**本机 Ollama（11434）可不设**，或填任意占位符（Ollama 常忽略） |
| `AGENTPV_JUDGE_MODEL` | 默认 `gpt-4o-mini`；Ollama 填如 `llama3.2:latest` |

---

## 14. 已知限制（Known limitations）

- 启发式 oracle 中 ``[MOCK]`` 为**软匹配**（见 `heuristic_rubric.py`）；其它关键词仍为硬子串匹配。换真实 LLM 后仍建议定期审阅 `benchmark.json` 期望集。
- LLM judge 调用 **云端**接口时依赖外网；失败时记录在 `judge_skip_reason`，**不会**静默成功。本机 Ollama 作 judge 时不走公网。
- `benchmark.json` 与内置生成器需保持同步；若手改 JSON 导致校验失败，`load_benchmark_json` 会抛 Pydantic 错误。
