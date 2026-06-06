# `rag/` — Retrieval-Augmented Generation 链路（Component 4）

按规则 §24 的 14 节工程文档。本模块是云端 agent 的"知识层"——把领域文档变成可
检索的向量索引，并按查询返回 top-k 相关片段。

---

## 1. 模块目的（Purpose）

让 agent 在生成 recommendation 前能"读到"领域文档，而不是凭空编造措施。
所有"知识库 → chunk → 向量 → 检索 → prompt"的链路在这里。

---

## 2. 为什么需要这个模块（Why it exists）

- 作业 §4 明文要求 RAG（chunking / embedding / retrieval / prompting）
- LLM 不能凭空知道现场具体的 SOP；必须把文档喂进 context
- agent 输出必须有可追溯的 citation（`Recommendation.knowledge_sources`），
  这要求 retrieve 步骤返回结构化结果

---

## 3. 架构概览（Architecture）

```text
knowledge_base/documents/*.md
                │
                ▼
    chunking.chunk_directory      (markdown-aware 分块)
                │
                ▼
            list[Chunk]
                │
                ├──→ embedding.TfidfEmbedder.fit / embed
                │
                ▼
       retrieval.Retriever
                │
                ▼
   search(query, top_k) → list[RetrievedChunk]
                │
                ▼
   reranking.IdentityReranker.rerank   (MVP 直通)
                │
                ▼
   prompting.PromptBuilder.render_recommendation_prompt
                │
                ▼
              LLM context
```

---

## 4. 关键文件（Key files）

| 文件 | 作用 |
|---|---|
| `chunking.py` | Markdown-aware 分块（按 heading + 段落） |
| `embedding.py` | `Embedder` 抽象 + `TfidfEmbedder`（MVP）；polish 阶段加 BGE |
| `retrieval.py` | 内存矩阵 + 余弦相似度的 `Retriever` |
| `reranking.py` | `IdentityReranker`（MVP 直通）；polish 阶段加 cross-encoder |
| `prompting.py` | Jinja2 prompt 模板（recommendation prompt） |
| `knowledge_base/documents/` | 5 篇 `placeholder_*.md`（rule §12 命名） |
| `knowledge_base/chroma_db/` | polish 阶段持久化向量库（MVP 不用） |

---

## 5. 输入输出契约（Inputs & Outputs）

### `Retriever.search(query, top_k) -> list[RetrievedChunk]`
- 输入：自然语言 query 字符串 + 正整数 top_k
- 输出：按余弦相似度降序的 RetrievedChunk 列表（含 score, source, title, section, text）

### `PromptBuilder.render_recommendation_prompt(alert, chunks) -> str`
- 输入：`Alert` + chunks 列表
- 输出：完整 prompt 字符串（含 alert 摘要 + 检索结果 + 任务指令）

---

## 6. 关键设计决策（Design decisions）

| 决策 | 备选 | 理由 |
|---|---|---|
| TF-IDF（MVP） | sentence-transformers BGE | sklearn 已装、零额外下载、确定性 → 单测可重现；polish 阶段升级 |
| 内存余弦相似度 | ChromaDB | 5–30 文档规模 << ChromaDB 启动成本；同样多了一个 docker 服务的复杂度 |
| markdown-aware 分块 | 固定字符切 | 把 `## 章节` 当成天然边界，prompt 可读性更高 |
| `IdentityReranker` 占位 | 直接不用 | 让 ReAct 调用点 `reranker.rerank(...)` 永远存在，polish 升级零侵入 |
| Jinja2 + StrictUndefined | f-string | 模板未来要拆 .j2 文件；StrictUndefined 在 prompt 缺变量时立刻报错 |
| `[MOCK]` placeholder doc 命名 | 用真实文档 | rule §12：placeholder 必须明确标注；polish 阶段替换后命名也跟着换 |

---

## 7. 反例（What NOT to put here）

- ❌ 工具实现（属于 `tools/`）
- ❌ ReAct 循环（属于 `agent/workflows/`）
- ❌ LLM 调用（属于 `agent/orchestration/llm_client.py`）
- ❌ HTTP 路由（属于 `api/agent_service.py`）

---

## 8. 教学注释

- **为什么 TF-IDF 在小语料库上够用？** 5 篇文档 ~3000 词，词汇覆盖度低；TF-IDF
  对"罕见词主导"的查询很有效。30 篇时（polish 阶段）会出现"语义但词面不一样"的查询
  （e.g. "energy storage" vs "BESS" vs "battery"），那时换成 dense embedding。
- **为什么 chunking 不切太细？** 太细的 chunk 失去上下文；prompt 里看到孤立句子，
  LLM 反而要花更多 token 重建语境。MVP 默认 max_chars=1200。
- **为什么必须走 reranker 这层？** 因为 retrieval 是 recall-oriented（回想率），
  rerank 是 precision-oriented（准确率）。两步分离让我们 polish 时可以独立优化。

---

## 9. 与其它模块的接口（Interfaces）

| 上游 | 下游 |
|---|---|
| `data/` 上的 `*.md` 文档 | `tools/retrieve_knowledge.py` 用 `Retriever` |
| `api/schemas.py` 提供 Alert | `agent/workflows/react.py` 调用 retrieve_knowledge |
| `configs/settings.py::knowledge_base_dir` 配置路径 | `api/agent_service.py` lifespan 启动时构造 |

---

## 10. 测试覆盖（Tests）

| 测试文件 | 数量 | 覆盖点 |
|---|---|---|
| `tests/unit/test_chunking.py` | 7 | 空文档 / 单 section / 多 section / 长段切分 / 目录加载 / Chunk 不可变 |
| `tests/unit/test_retrieval.py` | 14 | TF-IDF fit/embed / Retriever 排序 / 边界 / 重排器 / Prompt 渲染 |

---

## 11. 性能预算（Perf budget）

| 项 | 目标 | 当前（5 篇 placeholder） |
|---|---|---|
| Retriever 构建（含 fit + embed） | < 500 ms | ~50 ms |
| 单次 search top_k=3 | < 50 ms | ~3 ms |
| 知识库可包大小 | < 5 MB | < 50 KB（5 个 markdown） |

---

## 12. 未来扩展（Future work）

- `embedding.SentenceTransformerEmbedder` 用 `BAAI/bge-small-en-v1.5`
- `reranking.CrossEncoderReranker` 用 `BAAI/bge-reranker-base`
- 把 retriever 从内存切成 ChromaDB 持久化
- Hybrid retrieval（BM25 + dense）
- 知识库扩到 ≥30 篇真实文档（vendor 手册 + 论文摘要 + SOP）

---

## 13. 运行示例（Usage example）

```python
from rag.retrieval import build_retriever_from_dir
from rag.prompting import PromptBuilder
from utils.paths import KB_DOCS_DIR

retriever = build_retriever_from_dir(KB_DOCS_DIR)
chunks = retriever.search("inverter fault P_ac collapse", top_k=3)
for c in chunks:
    print(f"{c.score:.3f}  [{c.chunk.title}] {c.chunk.text[:80]}…")

# 拼 prompt 给 LLM
prompt = PromptBuilder().render_recommendation_prompt(alert, chunks)
```

---

## 14. 已知限制（Known limitations）

- TF-IDF 检索语义弱（同义词、跨语言）；MVP smoke 测试发现 "Inverter_fault"
  query 排第一的不是 inverter playbook 而是 partial shading；polish 阶段 BGE 后修复
- 5 篇 stub 不能代表真实 30 篇文档的检索难度；polish 阶段需要重新评估
- 没有 metadata filter（如按 system_type 过滤）；polish 阶段加
