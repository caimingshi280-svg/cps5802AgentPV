# AgentPV Final Presentation — PPT Build Guide

> **Purpose**: Build a **complete, defensible** academic deck from this repo’s full-pipeline artefacts (`reports/`, `data/`, `docs/`).  
> **Data source**: `reports/final_report.md` / `.pdf` (2026-06-03), `docs/data_card.md`, `reports/model_eval.md`, `reports/agent_eval.md`, `reports/integration_eval.md`, `reports/integration/fault_injection_demo.md`, `reports/robustness_eval.md`.  
> **Deck size**: **38 main slides + 6 appendix = 44 slides** (~30–40 min spoken; web UI demo is a **separate session**, not in this deck).  
> **Slide language**: **English only on slides** (titles, bullets, tables, figure captions).  
> **Narration**: **Bilingual and detailed below each slide** — use as speaker script; do not paste onto slides.  
> **Readable narration-only export**: [`ppt旁白.md`](ppt旁白.md) — regenerate with `python scripts/extract_presentation_narration.py`.  
> **Visual style**: White / `#F7F8FA` background; titles `#1A365D`; body `#2D3748`; accent `#2B6CB0` for numbers and ✅/⚠. Minimal animation.

---

## 1. Master layout (set up first)

| Element | Recommendation |
|---------|----------------|
| Aspect ratio | 16:9 |
| Header | Left: section tag (e.g. `§3 Data`); right: page `12 / 44` |
| Title font | 28–32 pt, left-aligned |
| Body font | 18–22 pt; tables 16–18 pt |
| Margins | 5% L/R, 8% T/B |
| Figures | 45–55% of content area; 1 pt light-gray border; never stretch |
| Tables | Navy header, white text; zebra rows; numbers right-aligned |
| Footer (optional) | Small gray source path, e.g. `Source: integration_eval_meta.json` |

**Layouts (reuse everywhere)**:

- **Layout A** — Text/table left 52%, figure right 45%.  
- **Layout B** — Table top 40%, figure bottom 55%.  
- **Layout C** — Full-width text (outline, Q&A backup).

**Narration convention in this guide**

- **On-slide tables**: Write as **Markdown pipe tables** inside each slide’s ` ``` ` block (see Slide 6, 27, 30). `scripts/render_presentation.py` converts them to styled PowerPoint tables (navy header, zebra rows)—do not paste screenshots of tables unless noted.
- **旁白（中文，详细）** — Full spoken script for learning & defense; explain *what, why, how measured, so what*.  
- **Narration (English, detailed)** — Same content in professional but spoken English (avoid rare words).  
- Target **~90–180 seconds per slide** on technical pages; **~45–60 s** on transition slides.

---

## 2. Slide-by-slide content (38 main slides)

> **Scope note**: This deck covers the **full project end-to-end** (C1–C8, reports, numbers). **Do not** mention Docker Compose (not validated on our machine). **Do not** walk through the Streamlit browser demo here—the live web demo is a **separate segment** (see `docs/网页演示指南.md` off-deck).

---

### Slide 1 · Title

**Layout**: Layout C, vertically centered.

**On-slide text (English only)**:

```
AgentPV
An End-to-End Fault Diagnosis and Reasoning Pipeline
for Photovoltaic (PV) and Battery Energy Storage (BESS) Systems

Course: AI-Powered Cyber-Physical Systems (CPS 5802 SP26)
Presenter: [Your Name]
Date: 2026-06-03
Reproducible full pipeline · 284 unit tests · 24-page final report (PDF)
```

**Figures**: Optional watermark — `reports/figures/integration/03_edge_vs_agent_split.png` at 15% opacity, bottom-right.

**旁白（中文，详细）**  
各位老师好。我汇报的项目叫 AgentPV，全称是面向光伏和电池储能的云边协同故障诊断与推理流水线。这个项目解决的是一个很实际的问题：电站和储能站有大量传感器数据，边缘设备算力有限但必须在毫秒到百毫秒级给出告警；运维人员又不满足于只知道「是什么故障」，还需要知道「现在该做什么、有多紧急、依据哪条内部规程」。  
AgentPV 的架构因此分成两层：边缘是量化 CNN，云端是 ReAct + RAG 智能体；中间还有多节点编排器写 JSONL，以及 Component 7 的操作员界面（本 PPT 只讲设计与脚本化证据，**网页现场演示另场进行**）。  
今天我会按 C1 到 C8 完整讲一遍：数据、模型与压缩、鲁棒性、智能体评测、集成与 graceful degradation、C7 交付物与终稿报告。所有 headline 数字来自 2026-06-03 全链路产物。

**Narration (English, detailed)**  
Good morning. I present **AgentPV**—an end-to-end cloud–edge pipeline for fault diagnosis and operator assistance in PV plants and grid-scale BESS.  
The operational problem is twofold. At the edge, gateways have limited CPU and memory but must classify faults within milliseconds to hundreds of milliseconds, often while cloud connectivity is degraded. At the operator desk, a bare class label is insufficient: crews need an actionable recommendation, an urgency level, and citations to the internal playbook so actions are auditable.  
AgentPV addresses both layers. A quantised 1D-CNN on the edge labels sixty-second, eight-channel windows into eleven fault classes. A cloud ReAct agent returns recommendations with knowledge-base citations. A ten-node orchestrator proves scale-out and graceful degradation on the **same HTTP path** as benchmarks. Component seven includes an operator UI and scripted fault-injection reports—we describe that deliverable here; the **live browser demo is a separate session**, not part of this slide deck.  
Today I walk through components C1 through C8 with numbers from our 2026-06-03 full pipeline run—traceable in `reports/final_report.pdf`.

**Anticipated Q&A**: *What does “AgentPV” mean?* — “Agent” = cloud ReAct LLM tier; “PV” = primary domain, extended to BESS.

---

### Slide 2 · Agenda

**Layout**: Layout C, two columns.

**On-slide text (English only)**:

```
Part I    Motivation & contributions          (Slides 3–5)
Part II   Assignment mapping C1–C8           (Slide 6)
Part III  System architecture                (Slides 7–8)
Part IV   C1 — Data & simulation             (Slides 9–12)
Part V    C2/C3 — Edge models & evaluation   (Slides 13–15)
Part VI   Robustness / OOD / selective pred. (Slides 16–19)
Part VII  C4/C5 — ReAct agent & benchmark    (Slides 20–26)
Part VIII C6 — Integration & 10-node orchestrator  (Slides 27–29)
Part IX   C7 — Interactive prototype (deliverable evidence) (Slide 30)
Part X    Engineering, discussion, conclusion, repro  (Slides 31–35)
Part XI   Q&A backup & closing                        (Slides 36–38)
Appendix  Class counts, figures index, commands       (A1–A6)
```

**Figures**: None.

**旁白（中文，详细）**  
这是今天的路线图。我会先讲为什么需要云边协同、我们和已有工作的差异、以及六大贡献；然后用一张表对照作业 C1 到 C8，让各位老师清楚每一项要求对应仓库里的哪份证据。  
之后按数据流顺序展开：仿真数据 → 边缘训练与 ONNX/INT8 → 六轴鲁棒性 → ReAct 智能体与 33 场景 benchmark → 三种集成模式的真 HTTP 延迟与十节点编排 → C7 脚本化故障注入报告（**不在此 PPT 里做网页演示**）。最后讲工程质量、局限、复现路径与 Q&A 备份。

**Narration (English, detailed)**  
This slide is the roadmap for a complete, structured defence. I start with motivation, related work, and six concrete contributions, then map every assignment component C1–C8 to verifiable artefacts in the repository.  
The technical narrative follows the pipeline: synthetic data, edge training and quantisation, six-axis robustness, the ReAct agent benchmark, live HTTP integration and orchestrator evidence, then component-seven **scripted** fault-injection results—not a live browser walkthrough in this deck. I close with engineering quality, limitations, reproducibility, and Q&A backup slides.

---

### Slide 3 · Problem — Why cloud–edge co-design?

**Layout**: Layout A. Text left, simple flow diagram right.

**On-slide text (English only)**:

```
Industrial pain points
• Dense PV/BESS telemetry — late fault detection → revenue loss, asset damage, safety risk
• Edge gateways: limited compute; need ms–100 ms decisions; must alert when cloud is down
• Operators need more than a label: what to do, how urgent, which playbook section applies

Two competing requirements
1. Edge constraints — low latency, small model, offline graceful degradation
2. Operator interpretability — actionable text + priority + auditable KB citations

AgentPV design
• Edge CNN-1D → structured Alert (class, severity, confidence, snapshot)
• Cloud ReAct + RAG → Recommendation (action, urgency, knowledge_sources[])
• Orchestrator + operator UI → multi-node scale + auditable JSONL events (same HTTP path as C6)
```

**Figures**: SmartArt or exported diagram:  
`Sensors → Edge POST /predict → Alert → Agent POST /recommend → Recommendation → Dashboard`

**旁白（中文，详细）**  
光伏和储能站点都是典型的信息物理系统：每秒都有电压、电流、温度、辐照度或 SOC 等遥测进来。如果故障发现晚，轻则发电损失，重则热失控或设备不可逆损伤。边缘网关通常只有 CPU、内存都很有限，而且现场经常断网——所以第一层必须本地、快速、可靠地给出告警。  
但运维一线反馈是：光告诉「Partial_shading」或「Thermal_anomaly」不够，他们还要知道是先降功率、先隔离、还是先叫人，以及依据哪条 SOP。这就必须有一层云端推理，把结构化告警变成自然语言建议，并且引用知识库 chunk，方便事后审计。  
AgentPV 不是把两件事硬拼在一起，而是用明确的 HTTP 契约连接：`SensorWindow` 进 edge，`Alert` 进 agent，输出 `Recommendation`。这样 benchmark、编排器、故障注入脚本都走同一条路，评测结果才代表真系统。操作员界面单独有网页演示环节，本 PPT 只讲设计与脚本化证据。

**Narration (English, detailed)**  
PV and BESS sites are classic cyber-physical systems: continuous telemetry on voltage, current, temperature, irradiance or state-of-charge. Late detection costs money and can create safety incidents. Edge gateways have tight resource budgets and often lose cloud connectivity—so the first tier must classify locally, quickly, and reliably.  
Operators consistently ask for more than a fault name. They need an imperative action, an urgency level, and a traceable link to the playbook—not a bare softmax label. That motivates a cloud tier that turns structured alerts into recommendations with knowledge-base citations.  
AgentPV connects the tiers through explicit HTTP schemas: `SensorWindow` into edge `/predict`, `Alert` into agent `/recommend`, yielding a `Recommendation`. Benchmarks, the orchestrator, and scripted fault injection all reuse that path—so evaluation results reflect the real system. The browser-based operator UI is demonstrated in a separate session; this deck covers design and scripted evidence only.

---

### Slide 4 · Related work & our differentiation

**Layout**: Layout C, three-column table.

**On-slide text (English only)**:

```
| Topic | Typical prior work | AgentPV difference |
| PV fault DL | ResNet / CNN / LSTM + confusion matrix | ONNX/INT8 deployment budgets + full-chain latency |
| BESS prognostics | RUL / SOH models | RUL as an agent *tool*, not the headline classifier |
| LLM agents | ReAct + RAG in isolation | Closed loop with live edge HTTP + 33-scenario benchmark |
| OOD detection | Energy score, MSP, etc. | Six stress axes + honest success/failure boundaries |

References (final report §11.4): Goodfellow et al. FGSM; Liu et al. energy OOD;
Lewis et al. RAG; Yao et al. ReAct; Hendrycks & Gimpel MSP
```

**Figures**: None.

**旁白（中文，详细）**  
相关工作可以分四块。第一，光伏故障深度学习文献很多停在准确率或混淆矩阵，很少同时报告 ONNX 导出、INT8 压缩、CPU P95 延迟和模型体积是否满足部署预算——我们把这些写进 C3 对照表。第二，储能领域常见的是 RUL 或 SOH 估计；在本项目里 RUL 是智能体可调用的工具之一，而不是边缘主分类任务，这样分工更清晰。第三，ReAct 和 RAG 在 NLP 里很常见，但很多 demo 是 isolated prompt；我们是真 edge 服务 in the loop，对 33 个场景跑消融。第四，OOD 检测我们采用 energy score，但不止报 AUROC，还做六轴 stress，并明确写「什么时候策略有效、什么时候无效」。  
差异化总结：AgentPV 评的是**整条运维链**——准确率、延迟、压缩、grounding、graceful degradation——而不是单一 offline metric。

**Narration (English, detailed)**  
Related work falls into four buckets. PV fault-detection papers often report accuracy or confusion matrices but rarely commit to ONNX export, INT8 trade-offs, CPU P95 latency, and model size against explicit deployment budgets—we report all of that in component three. BESS literature emphasises RUL or SOH; in AgentPV, RUL is a tool the agent may call, not the edge headline classifier. ReAct and RAG are well known in NLP, but many demos use isolated prompts; we close the loop with live edge HTTP and thirty-three benchmark scenarios under three ablations. For OOD, we use energy scores but go further with six stress axes and documented failure boundaries—not just AUROC.  
The differentiator is evaluating the **full operator-facing chain**: accuracy, latency, compression, grounding, and graceful degradation together.

---

### Slide 5 · Six contributions (from abstract)

**Layout**: Layout C, numbered list + small figure bottom-right.

**On-slide text (English only)**:

```
1. Reproducible codebase — simulation, training, quantisation, RAG, agent,
   orchestrator, dashboard; 284 unit tests; ruff clean
2. Two systems × three backends — PyTorch FP32, ONNX FP32, ONNX INT8;
   full Macro-F1 / P95 / size characterisation
3. Deployment-realism suite — six stress axes + energy-based selective
   prediction calibrated to 95% validation coverage
4. LLM agent benchmark — 33 scenarios × three ablations; LLM-as-judge mean 4.10 / 99
5. Integration ablation — edge_only / full / cloud_only (50 HTTP runs each)
   + 10-node orchestrator (144 events / 60 s)
6. C7 operator UI + scripted fault injection — dashboard/inject.py on orchestrator HTTP path;
   five scenarios in fault_injection_demo.md (all ✅)
```

**Figures**: `reports/figures/integration/01_latency_bars.png` (~35% width, bottom-right).

**旁白（中文，详细）**  
这页是全文贡献的「目录」。第一，代码库可复现：从 `simulation/generate_dataset.py` 到 `dashboard/inject.py` 全链路有脚本，284 条单元测试 guarding 关键路径。第二，PV 和 BESS 各训一套 CNN，每条线有 PyTorch、ONNX FP32、ONNX INT8 三种后端，Macro-F1、P95、MiB 都有表。第三，应导师 2026-05-13 反馈，我们做了 distribution shift、缺通道、噪声、漂移、FGSM、跨系统 OOD 六轴测试，并加 energy-based 选择性预测。第四，智能体用真 Ollama llama3.2，33 场景 × 三消融，LLM-as-judge 平均 4.10。第五，集成评测是真 HTTP：三种模式各 50 次，再加 10 节点编排 144 events。第六，C7 交付物包括双语操作员界面与 `demo_fault_injection.py` 脚本报告——与编排器同 HTTP 路径，五场景全部通过；**网页现场演示不在本 PPT 内**。  
右下角小图预告 C6：full 模式 P95 约 9.8 秒，edge_only 只有几毫秒——后面会分解。

**Narration (English, detailed)**  
These six bullets index everything I will unpack. First, end-to-end reproducibility with two hundred eighty-four unit tests guarding simulation through dashboard inject. Second, dual-system models with three deployment backends each—PyTorch FP32, ONNX FP32, ONNX INT8—with macro-F1, P95 latency, and size tabulated honestly, including BESS INT8 failure. Third, responding to instructor feedback, six stress axes plus energy-based selective prediction calibrated on validation coverage. Fourth, a real local LLM benchmark—thirty-three scenarios, three ablations, LLM-as-judge mean four point one zero. Fifth, live HTTP integration—fifty iterations per mode plus a ten-node orchestrator session yielding one hundred forty-four events. Sixth, component seven delivers an operator UI plus scripted fault-injection report on the same HTTP path as the orchestrator—all five scenarios pass; the live browser walkthrough is a separate session, not in this deck.  
The small latency bar chart previews component six: full-mode P95 near nine point eight seconds versus edge-only in single-digit milliseconds—we decompose that later.

---

### Slide 6 · Assignment compliance C1–C8

**Layout**: Layout B. Table top 55%; optional `final_report.pdf` cover screenshot bottom.

**On-slide text (English only)**:

```
| Component | Requirement (summary) | Evidence in repo | Status |
| C1 Data | ≥50k; PV≥7 + BESS≥5 classes; ≥3 conditions | 50,500 samples; docs/data_card.md; data/version.txt | ✅ |
| C2 Model | Time-series CNN; ONNX; ≤50 MB; CPU P95≤100 ms | models/cnn1d.py; quantization/; PV F1 0.9994 | ✅ |
| C3 Eval | Macro-F1, confusion matrices, compression trade-off | python -m evaluation --compare; model_eval.md | ✅ |
| C4 Agent | ReAct; RAG≥30 docs; four tools | agent/workflows/react.py; rag/ (30 playbooks) | ✅ |
| C5 Benchmark | ≥30 scenarios; ablations; scoring | 33×3 ablations; judge 4.10 | ✅ |
| C6 Integration | ≥10 nodes; E2E P95≤10 s; three modes | pv6_bess4; full P95 9803 ms | ✅ |
| C7 Prototype | Web UI + interactive demo | dashboard/app.py; inject.py; fault_injection_demo.md (5/5 ✅) | ✅ |
| C8 Final report | Academic PDF | reports/final_report.pdf (24 pages A4) | ✅ |
```

**Figures**: Optional screenshot of PDF title page.

**旁白（中文，详细）**  
这张表直接对应评分 rubric，每一行都可以在仓库里打开验证。C1：50500 样本，超过 50000 门槛；data card 十四节模板完整。C2：时序 CNN、ONNX 导出、INT8 静态量化，体积远小于 50MB，边缘 P95 亚毫秒到毫秒级。C3：三种后端对照、混淆矩阵、trade-off 图。C4：ReAct 五阶段、30 篇 playbook、四个工具。C5：33 场景、三种消融、启发式 rubric + LLM judge。C6：`pv6_bess4` 十节点、三种 integration mode、full P95 9803 ms 小于 10 秒。C7：`dashboard/app.py` 四 Tab 操作员界面 + `scripts/demo_fault_injection.py` 脚本报告（五场景全 ✅，事件写入 `events_c7_demo.jsonl`）。C8：24 页 A4 PDF。  
本 PPT 不展开浏览器操作步骤；网页演示按 `docs/网页演示指南.md` 另场进行。

**Narration (English, detailed)**  
This slide maps the grading rubric to concrete artefacts—every row is verifiable on disk or via pytest. C1: fifty thousand five hundred samples with a full data card. C2: temporal CNN, ONNX export, static INT8, sub-fifty-megabyte models, edge P95 within one hundred milliseconds. C3: three backends, confusion matrices, compression trade-offs—including honest BESS INT8 degradation. C4: ReAct workflow, thirty playbook documents, four tools. C5: thirty-three scenarios, three ablations, heuristic plus LLM-as-judge scoring. C6: ten-node catalogue `pv6_bess4`, three integration modes, full-mode P95 nine thousand eight hundred three milliseconds under the ten-second budget. C7: operator UI in `dashboard/app.py` plus scripted fault injection—all five scenarios pass in `fault_injection_demo.md`. C8: twenty-four-page A4 PDF.  
This deck does not walk through the browser UI; that live demo follows `docs/网页演示指南.md` in a separate segment.

---

### Slide 7 · System architecture (four layers)

**Layout**: Layout A.

**On-slide text (English only)**:

```
Four layers
① Data      simulation/ → data/processed, splits/, version.txt
② Edge      api/edge_service → POST /predict → Alert JSON
③ Cloud     api/agent_service → POST /recommend → Recommendation + KB sources
④ Ops UI    orchestrator/ + dashboard/ → JSONL events + operator console (inject via inject.py)

HTTP contracts (api/schemas.py)
• SensorWindow — 60×8 float32 window
• Alert — fault_class, severity, confidence, sensor_snapshot
• Recommendation — action, urgency, confidence, escalate_to?, knowledge_sources[]

Service ports (local dev): Edge :8000 · Agent :8001 · Ollama :11434
```

**Figures**: Architecture diagram:
```
[Simulators] → [Train/ONNX] → [Edge :8000] → [Agent :8001]
                                    ↓              ↓
                         [Orchestrator — 10 nodes] → events.jsonl → [dashboard/ UI]
```

**旁白（中文，详细）**  
架构分四层，每层对应清晰目录。数据层用三个仿真脚本生成 NPZ 和 split CSV，`version.txt` 冻结 seed 和样本数。边缘层 `edge_service` 加载 ONNX（生产默认 BESS 用 FP32），输入 Pydantic 校验过的 `SensorWindow`，输出 `Alert`。云端 `agent_service` 跑 ReAct，输入 `Alert`，输出 `Recommendation`，其中 `knowledge_sources` 是审计关键字段。第四层是运维视角：`orchestrator` 模拟十个资产并发 tick，写 JSONL；`dashboard` 读 JSONL 并通过 `inject.py` 注入——与 `NodeRunner` 同 HTTP 路径。  
为什么强调 HTTP 契约？因为 agent_eval、C6 bench、C7 脚本注入都调同一套 API——如果 UI 走 mock 而集成走真服务，答辩说服力会下降。我们刻意避免这种 split-brain。

**Narration (English, detailed)**  
The architecture has four layers with clear directory boundaries. The data layer uses simulators and `generate_dataset.py`, freezing metadata in `version.txt`. The edge tier loads ONNX artefacts—production defaults to BESS FP32—and exposes `/predict` returning validated `Alert` objects. The cloud tier runs ReAct via `/recommend`, returning `Recommendation` objects whose `knowledge_sources` field is the audit trail. The operations layer includes a ten-node orchestrator writing JSONL and a dashboard that reads and injects via `dashboard/inject.py` on the same HTTP path as the node runner.  
We stress HTTP contracts because agent evaluation, component-six benches, and component-seven scripted injection all call the same APIs. Split-brain demos—mock in the UI, real services in tests—would weaken the defence; we avoided that by sharing the inject module with the orchestrator.

---

### Slide 8 · Repository map & reproducibility entry points

**Layout**: Layout C, two columns.

**On-slide text (English only)**:

```
Train & evaluate                    Run & report
simulation/                         api/edge_service.py
models/cnn1d.py                     api/agent_service.py
training/train.py                   orchestrator/
quantization/                       dashboard/app.py
evaluation/                         agent_eval/
scripts/run_robustness_eval.py      scripts/e2e_latency_bench.py
                                    scripts/demo_fault_injection.py

Docs & authoritative numbers
• Reproduction: 复现指南.md §3 · final_report.md §11.3
• Data card: docs/data_card.md
• C7 scripted demo: reports/integration/fault_injection_demo.md
• Off-deck web demo (separate session): docs/网页演示指南.md
• This deck’s numbers: integration_eval_meta.json, last_run_three_ablations_with_judge.json
```

**Figures**: None.

**旁白（中文，详细）**  
左列是离线训练评测链：从仿真到 robustness script。右列是在线服务与报告渲染。老师问「某个数字从哪来」，按用途回表：延迟看 `integration_eval.md` 和 `e2e_latency_*.json`；智能体看 `agent_eval.md` 和 with_judge JSON；分类看 `reports/pv/comparison.md`。复现入口两处：`复现指南.md` 逐步版、`final_report.md` §11.3 一键 PowerShell 块。本 PPT 不引入新数字，只做终稿与子报告的 readable 索引。

**Narration (English, detailed)**  
The left column is the offline chain from simulation through robustness evaluation. The right column is online services and report scripts. When asked where a number comes from, point to the artefact: latency from `integration_eval.md` and JSON benches; agent scores from `agent_eval.md` and the with-judge JSON; classification from PV/BESS comparison tables. Reproduction is documented twice—step-by-step in `复现指南.md` and one-shot in final report section eleven point three. This deck indexes those sources; it introduces no new metrics.

---

### Slide 9 · C1 — Why synthetic data?

**Layout**: Layout A.

**On-slide text (English only)**:

```
Course constraint (§8.5)
• Pre-built PV/BESS fault datasets from Kaggle / HuggingFace are forbidden
• Real plant data is proprietary and non-shareable

Our approach
• Physics-inspired simulators: PVSimulator + BatterySimulator + fault_injector
• Fixed seed = 42 → byte-identical regeneration (data/version.txt)
• Three operating conditions for deployment heterogeneity

Generation command
python -m simulation.generate_dataset --seed 42 \
  --n-pv 28000 --n-bess 22500 --n-pv-normal 8000 --n-bess-normal 5000
```

**Figures**: Pipeline diagram: `PVSim / BessSim → window → fault_injector → (X, y)`  
Optional: screenshot of `docs/data_card.md` §3 pipeline box.

**旁白（中文，详细）**  
C1 首先要解释数据来源合法性。课程 §8.5 明确禁止直接用 Kaggle 或 HuggingFace 上的 PV/BESS 故障数据集；真实现场数据又往往签 NDA，无法提交。所以我们用自研仿真器：PV 侧有辐照度、组件温度、DC 电压电流功率等简化物理关系；BESS 侧是 RC 等效电路加 SOC、内阻、循环老化项；`fault_injector` 对干净窗口施加确定性扰动，每种故障一个纯函数，RNG 可 seed。  
固定 seed 42 后，任何人重跑 `generate_dataset` 应得到相同 NPZ 和 split，元数据写在 `data/version.txt`。三类 operating condition（高辐照、低辐照、高温）按 5:3:2 采样，保证每个 fault class 在每种工况下都有样本——这是后续 distribution shift 测试的基础。

**Narration (English, detailed)**  
Component one begins with data provenance. The course forbids off-the-shelf Kaggle or HuggingFace PV/BESS fault sets; real plant data is typically proprietary. We therefore built physics-inspired simulators: PV channels follow simplified irradiance-to-power relationships; BESS channels follow an RC equivalent-circuit with SOC, resistance, and ageing terms; `fault_injector` applies deterministic, seedable perturbations per fault class.  
With seed forty-two, regeneration is byte-identical—recorded in `data/version.txt`. Three operating conditions sampled five-three-two ensure every fault class appears in every condition, which later supports distribution-shift evaluation in component three extensions.

---

### Slide 10 · C1 — Dataset scale & splits

**Layout**: Layout B. Table top; optional `data/version.txt` screenshot bottom.

**On-slide text (English only)**:

```
| Metric | Value |
| Total samples | 50,500 (PV 28,000 + BESS 22,500) |
| Window shape | 60 timesteps × 8 channels @ 1 Hz |
| Train / Val / Test | 35,126 / 7,768 / 7,606 (~70 / 15 / 15%) |
| Split policy | Stratified by system_id — no asset leakage across splits |
| Generated (UTC) | 2026-06-03T15:27:30Z |
| Metadata | data/version.txt · data/splits/*.csv |
```

**Figures**: Optional screenshot of `data/version.txt` JSON header.

**旁白（中文，详细）**  
50500 总样本，PV 28000、BESS 22500，满足「≥50000」硬门槛。每个样本是 60 秒、8 通道窗口，采样率 1 Hz，与边缘推理输入一致。划分不是简单 random row split，而是按 `system_id` 分层：同一个模拟资产的所有窗口只出现在 train、val 或 test 之一，避免「同一条串」泄漏到测试集导致 F1 虚高。比例大约 70/15/15，具体数字以 `version.txt` 为准：35126 / 7768 / 7606。  
2026-06-03 这轮是全链路 refresh 的时间戳；如果老师问「你的数字是不是旧的」，可以当场打开 `data/version.txt` 核对 `generated_at` 和 splits 字段。

**Narration (English, detailed)**  
Fifty thousand five hundred samples—twenty-eight thousand PV and twenty-two thousand five hundred BESS—exceed the fifty-thousand requirement. Each example is a sixty-step, eight-channel window at one hertz, matching edge inference input. Splits are stratified by `system_id` so the same simulated asset never appears in both train and test—a common leakage mistake we avoided. Approximate seventy-fifteen-fifteen split yields thirty-five thousand one hundred twenty-six train, seven thousand seven hundred sixty-eight validation, and seven thousand six hundred six test rows per the live `version.txt`.  
The 2026-06-03 timestamp marks our full-pipeline refresh; auditors can open `data/version.txt` on the spot to verify `generated_at` and split counts.

---

<!-- Slides 11–42 continued below -->

### Slide 11 · C1 — Fault taxonomy (11 classes)

**Layout**: Layout A. PV list left 48%, BESS list right 48%.

**On-slide text (English only)**:

```
PV (7 classes)                         BESS (5 classes)
PV_Normal                              BESS_Normal
Partial_shading                        Capacity_fade
Soiling                                Internal_resistance_increase
Bypass_diode_fault                     Thermal_anomaly
String_disconnection                   Cell_imbalance
Inverter_fault
Degradation

Edge output: 11 fault labels + severity (monitor / warning / critical)
severity = monitor → Agent NOT invoked (AGENT_TRIGGER_SEVERITIES = {warning, critical})
```

**Figures**: Optional screenshot of `docs/data_card.md` §5 taxonomy table.

**旁白（中文，详细）**  
C1 的标签体系直接来自 `api/schemas.py` 里的单一真源元组，PV 七类故障加 BESS 五类，边缘一共输出十一种非重复故障名（含两套 Normal）。正常类在生成时故意过采样：PV_Normal 约 8002、BESS_Normal 约 5000，用来压低运维误报率，训练侧再用 class-balanced cross-entropy 保护少数故障类。  
severity 是云边契约的第二维：monitor 只落盘 Edge Alert，不调用 `/recommend`；warning 和 critical 才走全链路。这与编排器 `AGENT_TRIGGER_SEVERITIES`、C7 场景 4（PV_Normal）完全一致，避免「演示一套规则、集成另一套规则」。  
答辩时若被问「为什么 eleven classes」，答：作业要求 PV≥7、BESS≥5，我们按系统拆开训练两个 CNN head，但运维视图统一成 Alert JSON。

**Narration (English, detailed)**  
Component one’s label taxonomy is frozen in `api/schemas.py`: seven PV faults plus five BESS faults, eleven distinct class names at the edge. Normal classes are oversampled—roughly eight thousand PV_Normal and five thousand BESS_Normal windows—to reflect operator tolerance for false alarms; training uses class-balanced cross-entropy so minority faults still learn well.  
Severity is the second contract dimension. `monitor` persists an edge alert only; `warning` and `critical` trigger `/recommend`. That gate matches the orchestrator’s `AGENT_TRIGGER_SEVERITIES` and C7 scenario four, so demo, benchmark, and integration share one rule—not two.  
If asked why eleven classes: the assignment requires at least seven PV and five BESS labels; we train separate CNN heads per system but expose a unified `Alert` schema to the cloud tier.

**Anticipated Q&A**: *Are PV and BESS labels merged in one model?* — No; separate weights and heads, shared architecture only.

---

### Slide 12 · C1 — Operating conditions & feature channels

**Layout**: Layout B. Condition table top 42%; feature contract table bottom 45%.

**On-slide text (English only)**:

```
Operating conditions (sample weights 5 : 3 : 2)
| Condition          | Samples | Share  |
| high_irradiance    | 25,110  | 49.7%  |
| low_irradiance     | 15,359  | 30.4%  |
| high_temperature   | 10,031  | 19.9%  |

Feature order (contract — do NOT reorder)
PV:   V_dc, I_dc, P, T_module, T_amb, G, P_ac, eta
BESS: V_term, I, SOC, T, R_est, sigma_V, N_cycle, SoH

Every (class × condition) pair is non-empty → enables distribution-shift tests
```

**Figures**: `reports/robustness/pv/figures/condition_heatmap.png` (right 45%, bottom).

**旁白（中文，详细）**  
三类 operating condition 按 5:3:2 采样，高辐照占约一半，高温最少但每个故障类在三种工况下都有样本——这是作业「≥3 工况」的硬要求，也是后面 robustness「distribution shift」轴能切片的前提。具体计数 25110 / 15359 / 10031 与 `docs/data_card.md` §7 一致。  
八个通道的顺序写死在 Pydantic `SensorWindow` 和训练 NPZ 里，训练和 ONNX 推理必须同一顺序；鲁棒性实验里的 missing-feature mask 也是按通道索引施加的。PV 侧偏电气与热环境，BESS 侧强调内阻估计、电压散布和循环老化代理量。  
右下 condition heatmap 会在后面鲁棒性章节再次出现：三类工况下 macro-F1 仍高位，说明 shift 轴上模型没有「只会高辐照」。

**Narration (English, detailed)**  
Three operating conditions are sampled five-three-two: high irradiance about half the corpus, high temperature the rarest—but every fault class appears in every condition. That satisfies the assignment and underpins our distribution-shift stress axis. Counts twenty-five thousand one hundred ten, fifteen thousand three hundred fifty-nine, and ten thousand thirty-one match `docs/data_card.md` section seven.  
Eight channels are order-locked in `SensorWindow` and training tensors; ONNX inference and missing-feature masks rely on the same indices. PV channels emphasise electrical and thermal environment; BESS channels emphasise resistance estimates, voltage spread, and ageing proxies.  
The condition heatmap preview shows macro-F1 stays high across regimes—we return to that under robustness.

---

### Slide 13 · C2 — CNN-1D edge architecture

**Layout**: Layout A. Bullets left; simple block diagram right.

**On-slide text (English only)**:

```
Architecture (models/cnn1d.py, ≈48k parameters per system)
Input (60×8) → Conv1d→BN→ReLU ×2 → GlobalAvgPool → Dense → softmax

Training (training/train.py)
• Adam + class-balanced cross-entropy + gradient clipping
• Early stopping on validation macro-F1
• PV: 25 epochs · BESS: 30 epochs (CPU-friendly; GPU optional)

Export chain
best .pt → onnx_export (μ, σ baked in) → onnx_fp32 → int8_static (1,024 calib samples)
```

**Figures**: Simple three-block diagram: Input → Conv stack → Classifier (hand-drawn or SmartArt).

**旁白（中文，详细）**  
边缘模型是刻意小的 1D-CNN：两层卷积加 BN、ReLU，全局平均池化后接全连接，每系统约四万八千参数，远小于 50 MiB 预算。PV 与 BESS **共享结构、不共享权重**，避免跨系统 shortcut。训练目标用 validation macro-F1 做 early stop，比裸 accuracy 更适合轻度不平衡。  
导出链把标准化 (μ,σ) 烘焙进 ONNX 图前端，保证 edge 服务只做一次 forward；静态 INT8 用 1024 条校准样本做 per-tensor MinMax——这也是 BESS INT8 失败的根因之一，后面会诚实展示。  
老师若问「为何不用 Transformer/LSTM」，答：作业强调部署预算；我们实测 ONNX FP32 P95 约 0.15 ms，满足亚百毫秒约束，且可解释、可量化。

**Narration (English, detailed)**  
The edge tier is a deliberately small 1D-CNN: two convolution–batch-norm–ReLU stages, global average pooling, dense classifier—about forty-eight thousand parameters per system, far below the fifty-megabyte cap. PV and BESS share architecture but not weights, avoiding cross-system shortcuts. Early stopping tracks validation macro-F1, fairer than accuracy under mild imbalance.  
Export bakes input standardisation into the ONNX graph front; static INT8 uses one thousand twenty-four calibration windows with per-tensor MinMax—the same policy that later breaks BESS INT8, which we report honestly.  
If asked why not Transformer or LSTM: the assignment stresses deployment budgets; ONNX FP32 P95 near zero point one five milliseconds meets sub-hundred-millisecond edge goals with a transparent, quantisable CNN.

---

### Slide 14 · C2/C3 — Three-backend comparison (headline numbers)

**Layout**: Layout B. Full-width table top 50%; trade-off figures bottom row.

**On-slide text (English only)**:

```
| System | Variant      | Macro-F1 | p95 latency | Size (MiB) | Budget |
| PV     | pytorch_fp32 | 0.9994   | 0.89 ms     | 0.184      | ✅     |
| PV     | onnx_fp32    | 0.9994   | 0.15 ms     | 0.176      | ✅     |
| PV     | onnx_int8    | 0.9994   | 0.10 ms     | 0.058      | ✅ lossless |
| BESS   | pytorch_fp32 | 0.9980   | 1.03 ms     | 0.184      | ✅     |
| BESS   | onnx_fp32    | 0.9980   | 0.15 ms     | 0.175      | ✅     |
| BESS   | onnx_int8    | 0.7058   | 0.09 ms     | 0.058      | ⚠ see Slide 15 |

ONNX FP32 ≈6× speed-up, zero F1 drift · INT8 ≈2× faster again, 3.16× smaller
Production default: BESS FP32 ONNX (0.175 MiB — still ≪50 MiB)
Source: reports/model_eval.md · evaluation --compare (2026-06-03)
```

**Figures**: `reports/pv/comparison_tradeoff.png` and `reports/bess/comparison_tradeoff.png` side by side (48% width each).

**旁白（中文，详细）**  
这张表是 C3 交付的核心数字墙。PV 三条线 macro-F1 都是 0.9994，INT8 甚至无损，体积从 0.184 MiB 压到 0.058 MiB。BESS FP32 同样 0.9980，但 INT8 掉到 0.7058——不是训练失败，而是量化与窄特征带冲突，下一页用混淆矩阵证明。  
延迟上 ONNX FP32 相对 PyTorch 约六倍加速，INT8 再减半；全部远低于 100 ms CPU P95 作业上限。生产建议写死在报告里：BESS 默认 FP32 ONNX，仍只有 0.175 MiB。  
底图 trade-off 把 accuracy–size–latency 三点连成曲线，方便老师一眼看到「我们 characterise compression trade-off，而不是只报最好点」。

**Narration (English, detailed)**  
This table is the component-three number wall. PV macro-F1 stays zero point nine nine nine four across PyTorch, ONNX FP32, and INT8—effectively lossless compression to zero point zero five eight megabytes. BESS FP32 matches zero point nine nine eight zero; INT8 falls to zero point seven zero five eight—not a training failure but quantisation colliding with narrow feature bands, shown on the next slide.  
ONNX FP32 yields roughly six-fold CPU speed-up over PyTorch with zero F1 drift; INT8 halves latency again. All variants sit far below the one-hundred-millisecond edge P95 budget. Production guidance: default BESS to FP32 ONNX at zero point one seven five megabytes.  
Bottom trade-off plots connect accuracy, size, and latency—evidence we characterise compression honestly, not cherry-pick the best cell.

**Anticipated Q&A**: *Why report BESS INT8 if it fails?* — Assignment requires trade-off analysis; honest failure boundaries strengthen the defence.

---

### Slide 15 · C3 — Confusion matrices (FP32 clean + INT8 failure)

**Layout**: Layout B. Three matrices in one row (or 2+1).

**On-slide text (English only)**:

```
PV ONNX FP32: clean diagonal — macro-F1 0.9994
BESS ONNX FP32: clean diagonal — macro-F1 0.9980
BESS ONNX INT8: confusion among BESS_Normal / Thermal_anomaly / Internal_resistance_increase

Root cause: narrow numeric bands in R_est, sigma_V, SoC trajectories;
per-tensor MinMax INT8 collapses separability (−29.2 pp macro-F1)

Documented remediations (future work): Entropy calibration, per-channel quantisation
Production: BESS FP32 ONNX until recovery
```

**Figures** (required):
1. `reports/pv/onnx_fp32/confusion_matrix.png`
2. `reports/bess/onnx_fp32/confusion_matrix.png`
3. `reports/bess/onnx_int8/confusion_matrix.png`

**旁白（中文，详细）**  
左两图说明 FP32 部署形态下 PV、BESS 对角线干净，满足「macro-F1 ≥0.99」叙事。第三张是 BESS INT8 的 canonical failure：Normal、热异常、内阻上升三类互混，和我们在 §4.4 写的机理一致——三类在估计内阻和电压散布上数值带很窄，per-tensor MinMax 把尺度压扁。  
我们刻意把 INT8 失败放在答辩正文，而不是只放附录，因为这是作业 §4.3 要求的 trade-off characterisation。补救路线也写在 `开发记录.md`：Entropy 校准、per-channel 权重量化；当前 scope 内生产默认 BESS FP32。  
若老师问「INT8 是否完全不可用」，答：PV 可用；BESS 需换量化策略前不建议上线 INT8。

**Narration (English, detailed)**  
The first two matrices show clean FP32 diagonals for PV and BESS—supporting macro-F1 at or above zero point nine nine. The third is the canonical BESS INT8 failure: Normal, thermal anomaly, and internal-resistance increase collapse together, matching section four point four—narrow bands in resistance and voltage-spread features crushed by per-tensor MinMax, a twenty-nine point two percentage-point macro-F1 drop.  
We show INT8 failure in the main deck, not hidden in appendix, because the assignment demands compression trade-off characterisation. Documented remediations include entropy calibration and per-channel quantisation; production defaults to BESS FP32 until then.  
If asked whether INT8 is useless: PV INT8 is effectively lossless; BESS INT8 is a documented negative result pending better quantisation policy.

---

### Slide 16 · Extended evaluation — deployment realism motivation

**Layout**: Layout C.

**On-slide text (English only)**:

```
Baseline macro-F1 is necessary — not sufficient for deployment

Six stress axes (reports/robustness_eval.md)
1. Distribution shift — per operating_condition slice
2. Missing features — mask ratios 0 / 0.1 / 0.3 / 0.5
3. Sensor noise — σ multipliers 0 → 0.5
4. Calibration drift — scale factors 0.8 → 1.2
5. Adversarial FGSM — ε 0 → 0.1 (PyTorch FP32)
6. Cross-system OOD — other system's test windows

Training-free enhancement
Energy-based selective prediction (Liu et al. 2020)
E(x) = −logsumexp(logits); threshold calibrated to 95% val coverage
Below threshold → unknown_fault / operator_review
```

**Figures**: None (optional six-axis icon row).

**旁白（中文，详细）**  
2026-05-13 导师反馈的核心是：高测试 F1 不等于现场可靠。因此我们扩展了六轴压力测试，每一轴都有可复现 generator 和 sweep 表，结果写入 `reports/robustness_eval.md` 与九张/system 图。  
分布内切片用 meta CSV 的 operating_condition；缺通道、噪声、漂移、FGSM、跨系统 OOD 分别对应不同 failure mode，后面一页会写「何时成功、何时失败」。选择性预测是 training-free 的 energy score，在 validation 上校准到 95% coverage，低于阈值就拒绝给出自信但错误的类。  
这一整章回答的是 deployment realism，而不是再刷一个 baseline 分数。

**Narration (English, detailed)**  
Instructor feedback on 2026-05-13 stressed that high test F1 does not imply field reliability. We therefore added six stress axes—each with reproducible generators and sweeps—documented in `reports/robustness_eval.md` and nine figures per system.  
Distribution shift slices by operating condition; missing channels, noise, drift, FGSM, and cross-system OOD target distinct failure modes documented on slide eighteen. Selective prediction uses a training-free energy score calibrated to ninety-five percent validation coverage—rejecting over-confident wrong classes.  
This chapter answers deployment realism, not another leaderboard baseline.

---

### Slide 17 · Robustness — selective prediction headline numbers

**Layout**: Layout B. Table top 45%; risk-coverage figure bottom 50%.

**On-slide text (English only)**:

```
| System | Clean F1 | OOD AUROC | Score direction   | Selective acc @95% cov | OOD reject rate |
| PV     | 0.9994   | 0.6037    | inverted (out>in) | 1.0000                 | 0.3281          |
| BESS   | 0.9980   | 1.0000    | inverted (out>in) | 0.9994                 | 0.0000          |

In-distribution: selective accuracy ≈1.0 at 95% coverage — policy works
Cross-system OOD: energy direction inverted — still discriminative; deploy with flip rule or Mahalanobis
Dataset refresh: 2026-06-03 (aligned with data/version.txt)
```

**Figures**: `reports/robustness/pv/figures/risk_coverage_curve.png` (~50% content height).

**旁白（中文，详细）**  
表头数字全部来自 2026-06-03 重跑的 robustness 报告，与 fifty thousand five hundred 样本刷新一致。分布内：PV/BESS 在 95% coverage 下 selective accuracy 接近 1，risk-coverage 曲线贴近左上角，说明「拒绝最不确定的 5%」策略有效。  
跨系统 OOD 时 energy 分数方向 inverted——高能量反而更像 in-distribution，但 AUROC 仍高于随机（PV 0.6037，BESS 1.0），所以部署不能盲用默认阈值，需要 flip 或 Mahalanobis 补充。我们选择在报告里写清，而不是只报一个好看的 AUROC。  
右图 risk-coverage 是答辩时解释 selective prediction 的首选视觉证据。

**Narration (English, detailed)**  
All headline cells come from the 2026-06-03 robustness rerun aligned with `data/version.txt`. In-distribution, selective accuracy nears one at ninety-five percent coverage for both systems—the risk-coverage curve hugs the ideal corner.  
Cross-system OOD inverts the energy direction—high scores look in-distribution—yet AUROC remains above chance, so deployment needs a flip rule or Mahalanobis fallback, documented rather than hidden behind a single metric.  
The risk-coverage figure is the preferred visual for explaining selective prediction in Q&A.

---

### Slide 18 · Robustness — when it succeeds vs fails

**Layout**: Layout C, two columns ✅ Success | ❌ Failure / partial.

**On-slide text (English only)**:

```
✅ Succeeds
• In-distribution rejection: selective acc ≈1.0 at 95% coverage
• Mild noise / drift: PV σ≤0.10, BESS σ≤0.20; drift ∈[0.95,1.05] → F1>0.90

❌ Fails / partial
• Missing 10% channels: accuracy −40 pp, confidence rises — needs upstream sensor-up check
• Cross-system swap: inverted energy — calibrate or add Mahalanobis
• Large drift ±20%: accuracy and confidence both collapse
• FGSM ε≤0.05: BESS more fragile; rejection alone insufficient → adversarial training
```

**Figures**: Left `reports/robustness/bess/figures/missing_features_curve.png`; right `reports/robustness/bess/figures/fgsm_curve.png` (48% each).

**旁白（中文，详细）**  
成功边界：分布内 selective prediction 几乎完美；轻噪声和轻漂移下 macro-F1 仍高于 0.9，说明仿真器生成的干净信号在小幅扰动下仍可分。失败边界同样重要——缺 10% 通道时准确率掉四十个百分点，而 energy confidence 反而上升，说明 rejection policy **挡不住** 坏输入，必须上游做 NaN/缺测检查。  
跨系统 OOD 与 INT8 脆弱性形成呼应：BESS 对 FGSM 小 ε 更敏感，单靠事后拒绝不够，需要对抗训练或更强特征正则。答辩策略是主动讲失败，体现 critical thinking。  
左右曲线分别是 missing features 和 FGSM，老师追问时指图指轴，不要只背口号。

**Narration (English, detailed)**  
Success: selective prediction near perfect in-distribution; mild noise and drift keep macro-F1 above zero point nine. Failures matter equally—ten percent missing channels drop accuracy forty points while confidence rises, so rejection cannot replace upstream completeness checks.  
Cross-system OOD echoes INT8 fragility; BESS degrades faster under small FGSM epsilon—post-hoc rejection is insufficient without adversarial or regularisation work. We state failures proactively for critical thinking credit.  
Use the missing-feature and FGSM curves as evidence, not slogans.

---

### Slide 19 · Robustness — representative figures (OOD & conditions)

**Layout**: Layout A. OOD histogram left 48%; condition heatmap right 48%.

**On-slide text (English only)**:

```
PV OOD energy histogram
• Cross-system inputs: in vs out overlap; direction inverted — visible by eye

PV condition heatmap
• Macro-F1 high across all three operating conditions — shift axis stable

BESS: nine figures under reports/robustness/bess/figures/
(overview_macro_f1, noise_curve, scale_drift_curve, …)
```

**Figures**:
- `reports/robustness/pv/figures/ood_energy_histogram.png`
- `reports/robustness/pv/figures/condition_heatmap.png`

**旁白（中文，详细）**  
左图是跨系统喂 PV 模型时的 energy 分布：in 与 out 重叠且方向与教科书假设相反，这就是 slide 17「inverted」的直观来源。右图 condition heatmap 显示三类工况下 F1 仍高，支撑「distribution shift 轴相对稳」的结论。  
BESS 侧另有九张标准图，答辩若时间紧可指路径 `reports/robustness/bess/figures/`，不必全贴进 deck；正文已用 missing 与 FGSM 代表最难轴。  
两图合起来传达：我们不仅算 AUROC，还要求能 **看见** 失败形态。

**Narration (English, detailed)**  
The OOD histogram shows cross-system energy overlap with inverted direction—the visual behind slide seventeen. The condition heatmap shows high macro-F1 across three regimes—shift axis relatively stable.  
BESS ships nine standard figures under `reports/robustness/bess/figures/`; the deck cites missing-feature and FGSM curves as hardest axes. Together, the slides show we require visible failure morphology, not AUROC alone.

---

### Slide 20 · C4 — ReAct cloud agent architecture

**Layout**: Layout A. Bullets left; five-phase loop diagram right.

**On-slide text (English only)**:

```
Cloud agent (agent/workflows/react.py)
ReAct loop: Observe → Reason → Act → Reflect → Report

Four tools (tools/)
| Tool               | Role                                      |
| retrieve_knowledge | Chroma + bge-small-en-v1.5 RAG on playbooks |
| system_history     | Simulated past alert frequency              |
| estimate_rul       | Simulated RUL estimate                      |
| escalate_alert     | Simulated escalation notification           |

LLM: Ollama llama3.2 (~2 GiB) — plan + synthesis
Invalid planner JSON → ollama_plan_fallback_mock (graceful degradation)

RAG: 30 playbook documents (≥30 required) → Chroma index; citations include chunk_id
```

**Figures**: Five-arrow ReAct cycle diagram (Observe→…→Report).

**旁白（中文，详细）**  
C4 云端不是裸 ChatGPT，而是带工具调用的 ReAct：`observe` 读 Alert，`reason` 规划，`act` 调工具，`reflect` 校验，`report` 输出 Recommendation。四个工具中 `retrieve_knowledge` 是 grounding 核心，连接 30 篇 playbook 与 Chroma 向量库。  
LLM 用本机 Ollama llama3.2，计划阶段若 JSON 解析失败，会走 `ollama_plan_fallback_mock` 确定性计划，保证工具仍执行、最终仍有 recommendation——这在 agent_eval 和 C7 HTTP 日志里都能观察到。  
Citation 带 chunk_id，是为了运维审计，也是 No-RAG 消融时唯一能证明「依据规程」的指标。

**Narration (English, detailed)**  
Component four is a tool-using ReAct agent—not a bare chat model. Observe ingests the alert; reason plans; act invokes tools; reflect checks; report emits the recommendation. `retrieve_knowledge` grounds answers in thirty playbook documents indexed in Chroma.  
Ollama llama3.2 powers plan and synthesis; invalid planner JSON triggers `ollama_plan_fallback_mock` so tools still run—observed in agent_eval and C7 logs. Citations carry chunk IDs for operator audit and for measuring No-RAG grounding loss.

---

### Slide 21 · C4 — Alert → Recommendation data flow

**Layout**: Layout B. Flow top 40%; compact JSON field list bottom 35%.

**On-slide text (English only)**:

```
1. Edge POST /predict(SensorWindow) → Alert
   fault_class, severity, confidence, sensor_snapshot

2. If severity ∈ {warning, critical} → Agent POST /recommend(Alert)
   → Recommendation: action, urgency, confidence, escalate_to?, knowledge_sources[]

3. OrchestratorEvent → JSONL → Dashboard (4 tabs)

Trigger rule (same as C7 demo)
monitor → edge only · warning/critical → full pipeline
```

**Figures**: `reports/figures/integration/03_edge_vs_agent_split.png` (agent >99% of full-mode latency).

**旁白（中文，详细）**  
数据流是答辩的云边故事线：结构化 `SensorWindow` 进 edge，出 `Alert`；只有 warning/critical 才进 agent 出 `Recommendation`。monitor 类（如 PV_Normal）故意不调用 agent，节省约 8.5 s 的 LLM 时间，也符合运维「正常不需 LLM」的预期。  
JSONL 事件被 orchestrator 和 dashboard 共享，保证被动监控与主动 inject 看同一份时间线。右图强调 full 模式延迟几乎全是 agent，不是 edge 慢——部署优化应优先 LLM，而不是砍掉 CNN。  
`knowledge_sources[]` 是审计字段，No-RAG 消融会把它清零，这是后面最重要的发现。

**Narration (English, detailed)**  
The data-flow story: `SensorWindow` to edge `Alert`; only warning and critical severities call `/recommend` for a `Recommendation`. Monitor classes skip the agent—saving roughly eight point five seconds and matching operator expectations for healthy assets.  
JSONL feeds both orchestrator and dashboard so passive monitoring and active inject share one timeline. The split chart shows full-mode latency is over ninety-nine percent agent—not edge—so optimise the LLM path first.  
`knowledge_sources[]` is the audit field that No-RAG zeroes—the headline finding ahead.

---

### Slide 22 · C5 — Agent benchmark design

**Layout**: Layout C.

**On-slide text (English only)**:

```
agent_eval/benchmark.json — 33 scenarios
• 23 unambiguous + 10 deliberately ambiguous
• Per-scenario oracle: urgency, must-contain keywords, forbidden phrases, min KB count

Runner pipeline
Real edge model in the loop → Recommendation → dual scoring
① Deterministic rubric composite (0–1)
② Optional LLM-as-judge (1–5) — this run: 99/99 scored, mean 4.10

Three ablations (assignment §4.5)
full | no_retrieve_knowledge (No-RAG) | no_reasoning_trace (No-Trace)
99 records = 33 × 3 · backend: Ollama llama3.2
```

**Figures**: None.

**旁白（中文，详细）**  
Benchmark 不是手写几个 prompt，而是 33 个带 oracle 的 JSON 场景，其中 10 个故意模糊，逼 agent 在不确定时仍要给 urgency 和合规措辞。Runner 每场景走真 edge 分类再调 agent，保证「in the loop」而不是离线编造 fault_class。  
双通道评分：启发式 rubric 可复现、可自动化；LLM-as-judge 提供人类可读质量，本轮 99 条全有分，均值 4.10。三消融对应 full、关 RAG、关 trace，共 99 条记录，源文件 `last_run_three_ablations_with_judge.json`。  
设计意图是让 provenance（KB 数）与 surface score 可分离，为 No-RAG 幻灯片埋伏笔。

**Narration (English, detailed)**  
Thirty-three oracle-annotated scenarios—ten deliberately ambiguous—force sensible urgency and compliant wording under uncertainty. The runner uses the real edge classifier in the loop, not offline fault labels.  
Dual scoring: deterministic rubric for automation; LLM-as-judge for human-readable quality—ninety-nine of ninety-nine scored, mean four point one zero. Three ablations yield ninety-nine records in `last_run_three_ablations_with_judge.json`.  
The design separates provenance from surface score—setting up the No-RAG slide.

---

### Slide 23 · C5 — Ablation results summary

**Layout**: Layout B. Full-width table top 52%; `ablation_summary.png` bottom 45%.

**On-slide text (English only)**:

```
| Ablation | Mean   | % perfect | % urgency | % keywords | % forbidden | % knowledge | KB/scn | % with KB | Tools/scn |
| full     | 0.9318 | 72.7%     | 100%      | 72.7%      | 100%        | 100%        | 2.61   | 91%       | 1.82      |
| No-RAG   | 0.9242 | 69.7%     | 100%      | 69.7%      | 100%        | 100%        | 0.00   | 0%        | 1.06      |
| No-Trace | 0.9015 | 60.6%     | 100%      | 60.6%      | 100%        | 100%        | 2.67   | 91%       | 0.00      |

LLM-as-judge mean: 4.10 / 99 · Aggregate heuristic mean: 0.919
Source: agent_eval/results/last_run_three_ablations_with_judge.json (2026-06-03)
```

**Figures**: `reports/figures/agent_eval/ablation_summary.png` (bottom 45%).

**旁白（中文，详细）**  
Full 配置 heuristic mean 0.9318，LLM judge 4.10；urgency、forbidden、knowledge 三个硬槽位都是 100%，说明安全相关维度没崩。keywords 只有 72.7% perfect，是 lexical drift，不是 urgency 错。  
No-RAG 行最关键：mean 只降到 0.9242，但 KB/scenario 从 2.61→0，% with KB 从 91%→0%——**表面分几乎不变，grounding 全灭**。No-Trace mean 0.9015，但 KB 行为与 full 相近，说明带宽紧时可以丢 trace 不丢 RAG。  
底图 ablation_summary 四槽位并排，适合老师一眼对比。

**Narration (English, detailed)**  
Full scores mean zero point nine three one eight heuristic and four point one zero judge; urgency, forbidden, and knowledge slots hit one hundred percent—safety-critical dimensions hold. Keywords at seventy-two point seven percent reflect lexical drift, not wrong urgency.  
No-RAG barely moves the mean—zero point nine two four two—but KB per scenario collapses from two point six one to zero and grounded share from ninety-one to zero percent. That is the headline: surface score hides grounding loss. No-Trace drops mean to zero point nine zero one five while preserving KB—drop trace under bandwidth pressure, not RAG.  
The ablation summary figure compares all four rubric slots side by side.

---

### Slide 24 · C5 — No-RAG: grounding collapse (headline finding)

**Layout**: Layout A. Text left 48%; KB chart right 50%.

**On-slide text (English only)**:

```
No-RAG ablation (retrieve_knowledge disabled)
• Heuristic mean: 0.9242 vs full 0.9318 — almost unchanged
• KB sources / scenario: 2.61 → 0.00
• % with KB: 91% → 0%
• Tool calls / scenario: 1.82 → 1.06

Conclusion
Plausible text without playbook citations → not auditable for operators
→ Measure provenance (citation count), not rubric mean alone

No-Trace ablation
Mean 0.9015; KB behaviour ≈ full → drop trace, keep grounding under bandwidth limits
```

**Figures**: `reports/figures/agent_eval/kb_sources_per_ablation.png` (required, right 50%).

**旁白（中文，详细）**  
这是智能体章节最重要的单页。关掉 RAG 后，LLM 仍能生成读起来合理的建议，启发式总分几乎不掉，但 `% with KB` 归零，运维无法追溯规程依据——在真实电站这等于不可接受。  
因此我们主张：agent 评测必须同时报 provenance 指标（KB/scenario、% with KB），不能只看 composite mean。右图 kb_sources_per_ablation 是答辩必放图，一眼显示 No-RAG 柱为零。  
No-Trace 对照说明 trace 影响可解释性审计表，但不影响 grounding；与 full 比 mean 略低主要伤在 keywords 槽位。

**Narration (English, detailed)**  
This is the agent chapter’s headline slide. Disabling RAG leaves plausible recommendations and nearly unchanged heuristic means—but zero knowledge citations, unacceptable for audited operations.  
Measure provenance—KB per scenario and percent grounded—not composite mean alone. The KB-sources chart is mandatory: No-RAG bars drop to zero. No-Trace shows trace affects audit tables, not grounding; mean gap is largely keywords.

**Anticipated Q&A**: *Why not 100% keywords?* — Real LLM wording drifts from oracle phrases; urgency/forbidden/knowledge remain 100%.

---

### Slide 25 · C5 — Score distribution & ablation delta

**Layout**: Layout A. Histogram top 48%; ablation_diff bottom 48%.

**On-slide text (English only)**:

```
Score histogram
• Three ablations compared; 0.90 quality line (dashed)

Ablation diff (full − ablated)
• Red bars: quality loss when component removed
• No-RAG / No-Trace impact varies by scenario

32 sub-1.0 records — mostly keywords slot failures, not urgency/forbidden
```

**Figures**:
- `reports/figures/agent_eval/score_histogram.png`
- `reports/figures/agent_eval/ablation_diff.png` (stacked vertically if needed)

**旁白（中文，详细）**  
直方图回答「整体有没有过 0.9 线」：三消融分布重叠较多，说明 LLM 质量整体合格，但仍有长尾低于 1.0。delta 图按场景展示 full 减去消融的差，红色表示去掉 RAG 或 trace 后变差——可指出哪些 scenario 对 grounding 最敏感。  
32 条 sub-1.0 记录几乎全败在 keywords，不是 urgency 或 forbidden，这对安全叙事是好事：我们没有在「该不该立即停机」上翻车。答辩时可挑一个 ambiguous scenario 口述，细节放附录 A5。

**Narration (English, detailed)**  
The histogram shows whether mass sits above the zero point nine quality bar—three ablations overlap heavily, indicating generally strong LLM behaviour with a sub-one tail. Delta charts show per-scenario full-minus-ablated gaps—red where removing RAG or trace hurts most.  
Thirty-two sub-one records fail mainly on keywords, not urgency or forbidden—a positive safety story. Pick one ambiguous scenario orally; details live in appendix A5.

---

### Slide 26 · C5 — Tool calls, ReAct depth & failure taxonomy

**Layout**: Layout A. `tool_calls_reasoning.png` left 48%; `failure_taxonomy.png` right 48%.

**On-slide text (English only)**:

```
Tool calls & ReAct depth (full ablation)
• ~1.82 tool calls / scenario · ~5.82 ReAct steps / scenario
• No-RAG: fewer tools (no retrieve_knowledge)
• No-Trace: ReAct steps → 1.0 (recommendation only)

Failure taxonomy
• 32 scores <1.0: predominantly keywords mismatch
• urgency / forbidden / knowledge slot failures rare

Planner fallback: ollama_plan_fallback_mock in logs — chain still returns HTTP 200
```

**Figures**:
- `reports/figures/agent_eval/tool_calls_reasoning.png`
- `reports/figures/agent_eval/failure_taxonomy.png`

**旁白（中文，详细）**  
工具图证明 full 配置确实在调用 RAG 等工具，不是空转 ReAct。No-RAG 工具数下降符合预期；No-Trace 把步数压到 1，只剩最终 recommendation，适合解释「带宽受限时的降级选项」。  
failure_taxonomy 把 sub-1.0 的丢分槽位可视化，答辩强调「软失败在措辞，硬失败在安全槽位很少」。Planner fallback 说明 Ollama JSON 不稳定时系统仍 200 OK，与 C6/C7 graceful degradation 叙事一致。  
若老师问 ReAct 是否 over-engineered，答：步数和工具数是可观测的，且 No-Trace 消融量化了 trace 价值。

**Narration (English, detailed)**  
Tool charts prove full mode actually invokes RAG—not an empty ReAct theatre. No-RAG lowers tool counts; No-Trace collapses steps to one for bandwidth-limited degradation. Failure taxonomy shows soft failures concentrate on keywords, not safety slots. Planner fallback keeps HTTP two hundred when JSON parsing fails—consistent with integration graceful degradation.  
If asked whether ReAct is over-engineered: step and tool counts are observable, and No-Trace quantifies trace value.

---
### Slide 27 · C6 — Three-mode integration latency

**Layout**: Layout B. Table top 48%; latency bars (+ optional violin) bottom 50%.

**On-slide text (English only)**:

```
Setup: live edge :8000 + agent :8001 · 50 iterations/mode (3 warmup discarded)

| Mode       | P50 (ms) | P95 (ms) | P99 (ms) | 10 s P95 budget |
| edge_only  | 4.50     | 5.69     | 6.47     | ✅              |
| full       | 8438     | 9803     | 10001    | ✅              |
| cloud_only | 9163     | 9941     | 10626    | ✅              |

full-mode decomposition
• Edge /predict P95: 26.27 ms (<1%)
• Agent /recommend P95: 9785 ms (>99%) — Ollama bottleneck

Generated: 2026-06-03T16:17Z · Source: reports/integration_eval.md
```

**Figures**: `reports/figures/integration/01_latency_bars.png` + optional `02_latency_violin.png`.

**旁白（中文，详细）**  
C6 用真 HTTP 而非 mock：三种 integration mode 各跑 50 次，warmup 丢弃 3 次后统计 P50/P95/P99。edge_only P95 5.69 ms，是 graceful degradation 的地板；full P95 9803 ms，cloud_only 9941 ms，都低于 10 s 作业预算。  
分解图是关键证据：full 模式里 edge 只占不到 1%，agent/LLM 占 99% 以上——优化 E2E 应换更快本地模型、缓存或 speculative decode，而不是去掉 CNN。数字与 `integration_eval_meta.json` 一致，2026-06-03 生成。  
三张 latency 图（bars、violin、split）建议都放进 deck 或附录，答辩时至少讲 bars + split。

**Narration (English, detailed)**  
Component six uses live HTTP—fifty iterations per mode, three warmups discarded. Edge-only P95 is five point six nine milliseconds—the degradation floor. Full P95 nine thousand eight hundred three milliseconds; cloud-only nine thousand nine hundred forty-one—both under the ten-second budget.  
Decomposition shows edge under one percent of full-mode latency; the agent exceeds ninety-nine percent—optimise the LLM path, not the CNN. Numbers match `integration_eval_meta.json`, generated 2026-06-03.  
Show at least the bar and split charts in the deck or appendix.

---

### Slide 28 · C6 — Mode interpretation & deployment guidance

**Layout**: Layout C.

**On-slide text (English only)**:

```
1. edge_only = graceful-degradation floor (5.69 ms P95)
   Operators still receive typed Alerts when Agent/LLM is down

2. cloud_only P95 9941 ms ≈ full 9803 ms (Δ 1.4%)
   Bypassing edge saves almost no time but loses ML fault_class grounding
   → Deployment recommendation: edge always-on

3. full P99 touches 10.0 s — headroom target: faster LLM / cache / speculative decode

Orchestrator tip: --http-timeout 120 (复现指南 §3.9) under concurrent fan-out
```

**Figures**: `reports/figures/integration/03_edge_vs_agent_split.png` (full width, ~40% height).

**旁白（中文，详细）**  
三种模式的**比较意义**比单点延迟更重要。edge_only 定义了「LLM 挂了还能运维什么」——仍有 fault_class 和 severity。cloud_only 与 full 几乎一样慢，却失去边缘 ML 标签对 agent tool 选择的 grounding，因此部署建议永远是 edge always-on。  
full P99 碰到 10.0 s，说明在并发或 tail 场景下仍可能触预算上限；编排器默认 10 s HTTP timeout 会在 fan-out 下产生 `agent_recommend_failed`，下一页用 144 events / 3 failures 证明系统不崩。  
timeout 调到 120 s 是复现建议，不是掩盖问题，而是让 demo 完整跑完同时保留失败计数作为证据。

**Narration (English, detailed)**  
Compare modes for meaning, not single numbers. Edge-only defines what operators still get when the LLM is down—typed alerts with fault class and severity. Cloud-only is only one point four percent slower than full yet loses ML grounding for tool calls—keep edge always on.  
Full P99 touches ten seconds—tail risk under concurrency. The orchestrator’s ten-second default timeout yields `agent_recommend_failed` events; slide twenty-nine shows one hundred forty-four events with three failures absorbed. Raising timeout to one hundred twenty seconds is a reproduction tip, not hiding failures—we count them.

---

### Slide 29 · C6 — 10-node orchestrator & graceful degradation

**Layout**: Layout B. Bullets top 42%; fanout + severity figures bottom 55%.

**On-slide text (English only)**:

```
Command: python -m orchestrator --nodes pv6_bess4 --duration 60 --http-timeout 120

Results (2026-06-03)
• 10 distinct nodes (6 PV + 4 BESS)
• 144 events in 60 s
• 3 × agent_recommend_failed (httpx TimeoutException)
• Orchestrator kept ticking — per-node asyncio tasks independent

| Error kind              | Count |
| agent_recommend_failed  | 3     |

Several nodes still received Recommendations under load — not a total outage
Deliverable #7: empirical graceful-degradation evidence
```

**Figures**:
- `reports/figures/integration/04_node_fanout.png`
- `reports/figures/integration/05_severity_mix.png`

**旁白（中文，详细）**  
`pv6_bess4` 目录十节点、跑 60 秒，共 144 条 JSONL 事件，其中 3 次 agent 推荐因 httpx 超时失败——并发 tail latency 超过编排器 HTTP 上限时的真实行为，不是单元测试 mock。关键是 orchestrator **没有崩溃**，各节点 task 继续 tick，部分节点仍收到 Recommendation。  
fanout 图展示每节点 events/alerts/recommendations/errors；severity mix 展示 monitor/warning/critical 比例，与 AGENT_TRIGGER 规则一致。这页直接支撑 Deliverable #7「empirical graceful degradation」。  
Q&A：为何 120 s timeout 仍有 3 次失败？——并发下 agent 队列仍可能触顶；证据是 failure count + 系统存活，而非零失败吹牛。

**Narration (English, detailed)**  
Ten nodes for sixty seconds yield one hundred forty-four JSONL events and three `agent_recommend_failed` timeouts—real concurrent tail behaviour, not mocks. The orchestrator never crashed; nodes kept ticking and some still received recommendations.  
Fanout and severity-mix figures prove per-node behaviour and severity gating. This slide supports deliverable seven—empirical graceful degradation.  
If asked why failures remain with a one-hundred-twenty-second timeout: concurrency still queues the agent; evidence is bounded failures plus system survival.

**Anticipated Q&A**: *Is three failures acceptable?* — Yes as documented tail behaviour; the system degrades without crashing.

---

### Slide 30 · C7 — Operator UI & scripted fault injection

**Layout**: Layout B. Top 45% design summary; bottom 50% five-scenario results table.

**On-slide text (English only)**:

```
Component 7 deliverable (not a live browser walkthrough in this deck)

Operator UI — dashboard/app.py (bilingual EN + ZH)
• Four tabs: Node overview · Event timeline · Event detail · Global stats
• Reads data/orchestrator/events*.jsonl · inject via dashboard/inject.py
• Same HTTP path as orchestrator NodeRunner: POST /predict → POST /recommend

Scripted evidence — scripts/demo_fault_injection.py → fault_injection_demo.md
Generated 2026-06-03T16:19Z · events → events_c7_demo.jsonl

| # | Scenario                              | Severity | Urgency   | Edge ms | Agent ms | KB | OK |
| 1 | PV Inverter_fault (critical)          | critical | immediate | 26.2    | 8653     | 3  | ✅ |
| 2 | PV Partial_shading (warning)          | warning  | scheduled | 6.8     | 8476     | 3  | ✅ |
| 3 | BESS Thermal_anomaly (critical)       | critical | immediate | 8.2     | 8705     | 3  | ✅ |
| 4 | PV_Normal (monitor — agent skipped)   | monitor  | —         | 6.5     | —        | 0  | ✅ |
| 5 | String_disconnection + skip_agent     | critical | —         | 11.3    | —        | 0  | ✅ |

Scenario 3 highlight: edge critical Thermal_anomaly → agent cites kb_thermal_anomaly_bess.md
Live browser demo: separate session (docs/网页演示指南.md)
```

**Figures**: Optional callout box with scenario 3 recommendation excerpt from `fault_injection_demo.md` (no Streamlit screenshot required).

**旁白（中文，详细）**  
C7 要求「可交互的原型」，我们交付两层证据。第一层是 `dashboard/app.py`：双语操作员界面，四个 Tab 分别看节点概览、时间线、单条事件详情和全局统计；数据来自 orchestrator 写的 JSONL，注入走 `dashboard/inject.py`，与 C6 编排器 `NodeRunner` 完全同路径——不是 mock 分支。  
第二层是脚本化复现：`python scripts/demo_fault_injection.py` 生成 `fault_injection_demo.md`，五场景全部 ✅。场景 1–3 走全链路，edge 个位数到二十多 ms，agent 约 8.5–8.7 s，各有 3 条 KB 引用。场景 4 证明 monitor 不触发 agent（与 `AGENT_TRIGGER_SEVERITIES` 一致）。场景 5 用 `skip_agent=True` 演示 LLM 不可用时仍有 critical Alert——graceful degradation。  
场景 3 最能说明云边价值：边缘给出 Thermal_anomaly + critical；云端返回 immediate urgency，并引用 `kb_thermal_anomaly_bess.md` 的 containment 步骤，而不是重复标签。Recommendation 全文在 demo 报告里可查，knowledge_sources 可追溯到向量库 chunk。  
**本 PPT 不做浏览器逐步演示**；若老师要看界面，说明另场按 `docs/网页演示指南.md` 进行。答辩口述此表即可证明 C7 可审计、可重复。

**Narration (English, detailed)**  
Component seven delivers two layers of evidence—not a live browser walkthrough in this deck. First, `dashboard/app.py`: a bilingual operator UI with four tabs reading orchestrator JSONL and injecting via `dashboard/inject.py` on the **same HTTP path** as the node runner—not a mock branch. Second, scripted reproduction: `demo_fault_injection.py` produces `fault_injection_demo.md` with all five scenarios passing. Scenarios one through three are full pipeline with three KB sources; four skips the agent for monitor severity; five forces edge-only degradation when the LLM is unavailable.  
Scenario three best shows cloud value: edge emits thermal anomaly critical; cloud returns immediate urgency citing containment steps from `kb_thermal_anomaly_bess.md`. Full recommendation text and chunk IDs are in the report.  
If asked about the UI, note the live browser demo is a **separate session** per `docs/网页演示指南.md`. For this deck, the scripted table is auditable C7 evidence.

**Anticipated Q&A**: *Is the UI the real system?* — Yes: `inject.py` shares the node-runner HTTP path; eleven unit tests in `test_dashboard_inject.py`.

---

### Slide 31 · Engineering quality & tests

**Layout**: Layout C.

**On-slide text (English only)**:

```
284 unit tests — pytest tests -q exit 0
Ruff: 0 warnings (final report §1)

Key modules
• tests/unit/test_dashboard_inject.py — 11 tests, MockTransport on inject path
• evaluation / agent_eval / orchestrator — dedicated unit coverage

Local services for benchmarks & C7 scripts
• uvicorn api.edge_service:app --port 8000
• uvicorn api.agent_service:app --port 8001
• Ollama llama3.2 for agent_eval and integration benches

Config: configs/dev.yaml · APP_ENV=dev|test
RAG corpus: 30 playbook documents ingested via python -m rag.ingest
```

**Figures**: Optional green pytest terminal screenshot.

**旁白（中文，详细）**  
工程质量与模型分数同等重要：284 条单元测试覆盖 simulation 到 dashboard inject，其中 inject 路径用 MockTransport 测 11 例，保证 C7 注入逻辑在答辩前已被验证。ruff 零 warning 在终稿声明，体现可维护性。  
集成评测、智能体 benchmark 和 C7 脚本都需要本机起 edge/agent 与 Ollama——复现步骤见 `复现指南.md` §4 与终稿 §11.3。RAG 30 文档需 `rag.ingest` 后才可检索。  
老师问「有没有 CI」：可指 pytest 全绿 + 报告 JSON 时间戳；GitHub Actions 非本作业硬性要求。

**Narration (English, detailed)**  
Two hundred eighty-four unit tests span simulation through dashboard inject—including eleven mocked inject tests so C7 logic is verified before defence. Zero ruff warnings are claimed in the final report.  
Integration benches, agent evaluation, and C7 scripts require local edge and agent services plus Ollama—documented in `复现指南.md` section four and final report section eleven point three. Thirty playbooks require `rag.ingest` before retrieval works.  
Point to pytest green and timestamped JSON artefacts if asked about automation.

---

### Slide 32 · Discussion — what worked

**Layout**: Layout C, ✅ list.

**On-slide text (English only)**:

```
✅ PV/BESS FP32 macro-F1 ≥0.99 · edge P95 sub-millisecond (ONNX)
✅ Honest BESS INT8 failure characterization — credible trade-off story
✅ Orchestrator absorbs node failures, HTTP errors, Ollama JSON fallback
✅ Real LLM: full mean 0.932 · judge 4.10 · urgency/forbidden/knowledge 100%
✅ Orchestrator & dashboard share inject HTTP path — evaluation fidelity
✅ Six-axis robustness + selective prediction — beyond baseline F1
```

**Figures**: None.

**旁白（中文，详细）**  
讨论章先讲 worked：双系统 FP32 高精度与亚毫秒边缘延迟；INT8 失败被诚实记录反而增强可信度；编排器与 mock plan fallback 展示 graceful degradation；真 LLM 下 rubric 与 judge 双通道都过线；inject 与集成同路径；鲁棒性六轴回应导师 deployment realism。  
这页为下一页局限做铺垫，体现「我们知道哪里好、哪里不好」。不要只报喜——评委更信 balanced narrative。  
可口头补一句：19+ 报告图、6 份子报告、24 页 PDF 都是同一晚 refresh 产物，时间戳 2026-06-03。

**Narration (English, detailed)**  
What worked: dual-system FP32 accuracy and sub-millisecond ONNX edge latency; honest BESS INT8 failure strengthens credibility; orchestrator and mock-plan fallback show graceful degradation; real LLM passes rubric and judge with safety slots at one hundred percent; shared HTTP inject path; six-axis robustness beyond baseline F1.  
This slide sets up limitations—balanced narrative beats cheerleading. Mention nineteen plus figures and six sub-reports refreshed 2026-06-03.

---

### Slide 33 · Discussion — limitations & future work

**Layout**: Layout C, two columns Limitations | Future (prioritised).

**On-slide text (English only)**:

```
Limitations                          Future work (prioritised)
BESS INT8 F1 = 0.7058                ① Entropy calib + per-channel quant
Energy OOD direction inverted        ② Auto-flip + Mahalanobis at deploy
Missing channels unprotected         ③ Upstream sensor-up / NaN gate
Full P95 ≈9.8 s near 10 s budget     ④ Faster 7–8B LLM; response cache
llama3.2 planner JSON unstable       ⑤ Stronger model; lower fallback rate
Dashboard manual JSONL refresh       ⑥ Auto-poll events feed
Synthetic-only training data         ⑦ Domain adaptation when data available
```

**Figures**: None.

**旁白（中文，详细）**  
局限与终稿 §9 一一对应：BESS INT8、OOD 方向、缺通道、E2E 贴近 10 s、planner JSON、Dashboard 需手动刷新 JSONL、训练数据为合成。每条局限都绑了优先 future work，说明不是「不会做」而是 scope 内诚实交付。  
答辩时主动讲 limitation 通常比等提问得分更高，尤其 INT8 和 No-RAG 已在正文展开，这里做收束。未来工作避免空泛，全部可追溯到 `开发记录.md` 或 final report §9.3。  
若老师问「若重做会改什么」，答：先 BESS 量化策略 + 7B LLM，再 adversarial training。

**Narration (English, detailed)**  
Limitations mirror final report section nine—INT8, inverted OOD, missing channels, tail latency, planner JSON, manual JSONL refresh, synthetic-only data—each paired with prioritised future work, not hand-waving. Proactive limitation discussion scores well after INT8 and No-RAG already in the body.  
If asked what to change first: BESS quantisation policy and a faster local LLM, then adversarial training.

---

### Slide 34 · Conclusion

**Layout**: Layout C, centered headline + bullets.

**On-slide text (English only)**:

```
AgentPV delivers a reproducible PV/BESS cloud–edge fault diagnosis stack

Assignment numeric budgets met
• Macro-F1 ≥0.99 (FP32) · Edge P95 <100 ms · E2E P95 <10 s · Model <50 MiB

Deployment realism answered
• Six stress axes + selective prediction
• LLM grounding ablation (No-RAG → 0% KB citations)
• 10-node graceful degradation (144 events, 3 agent failures absorbed)

284 tests · 6 reports · 19+ figures · final_report.pdf (24 pages A4)
```

**Figures**: `reports/final_report.pdf` cover thumbnail, bottom-right.

**旁白（中文，详细）**  
结论句：AgentPV 不是单一高 F1 分类器，而是可运维、可降级、可审计的云边系统。数值上满足作业四条预算；方法上回应 deployment realism——鲁棒六轴、选择性预测、No-RAG provenance、编排器失败吸收。  
284 测试、6 份子报告、终稿 PDF 均可按 §11.3 一键复现。保持语调自信但克制，数字已在前文证明，这里做 synthesis 即可。追问时可打开 `final_report.pdf` 任意章节 supplement。

**Narration (English, detailed)**  
AgentPV is not just a high-F1 classifier—it is an operable, degradable, auditable cloud–edge stack meeting numeric budgets and deployment-realism questions: six stress axes, selective prediction, No-RAG provenance collapse, and orchestrator failure absorption.  
Two hundred eighty-four tests, six reports, and a twenty-four-page PDF reproduce from section eleven point three. Close confidently but modestly—numbers already earned; this slide synthesises. Offer the PDF for follow-up detail.

---

### Slide 35 · Reproducibility & artefact index

**Layout**: Layout B. Command summary top 45%; artefact table bottom 50%.

**On-slide text (English only)**:

```
One-shot pipeline (final_report §11.3 · 复现指南 §4)
generate_dataset → train pv/bess → onnx + int8 → evaluation --compare
→ run_robustness_eval → rag.ingest → (start services) agent_eval + judge
→ e2e_latency ×3 → orchestrator --http-timeout 120 → demo_fault_injection
→ render_final_report.py

| Purpose        | Path |
| Data metadata  | data/version.txt |
| ONNX weights   | quantization/artifacts/*.onnx |
| C5 results     | agent_eval/results/last_run_three_ablations_with_judge.json |
| C6 report      | reports/integration_eval.md |
| C7 demo        | reports/integration/fault_injection_demo.md |
| Final PDF      | reports/final_report.pdf |
```

**Figures**: None.

**旁白（中文，详细）**  
复现不需要猜命令：`复现指南.md` §4 与终稿 §11.3 提供 PowerShell 复制块，含 judge 环境变量与 `--http-timeout 120`。表中路径是答辩追问时的「导航页」——数据看 version.txt，模型看 artifacts，智能体看 with_judge JSON，集成看 integration_eval.md，C7 看 fault_injection_demo.md。  
强调：文档对齐的答辩不必重跑全链路；只有改代码、权重或 benchmark 才需重跑对应步骤。PDF 可用 `render_final_report.py` 再生。  
若时间紧，附录 A6 放完整命令块，本页只讲索引逻辑。

**Narration (English, detailed)**  
Reproduction is copy-paste from `复现指南.md` section four and final report section eleven point three—including judge env vars and orchestrator timeout one hundred twenty. The table navigates artefacts for follow-up questions.  
Aligned documentation defence need not rerun the full pipeline—only changed code or weights require partial reruns. Regenerate PDF via `render_final_report.py`. Full commands live in appendix A6.

---

### Slide 36 · Q&A backup (1/2) — data & models

**Layout**: Layout C, two-column Q&A.

**On-slide text (English only)**:

```
Q: Why not real plant data?
A: Course forbids off-the-shelf Kaggle/HF sets; industrial data is non-public; synthetic seed-42 reproducible.

Q: How is 50,500 built?
A: PV 28,000 + BESS 22,500; Normal oversampling; see docs/data_card.md §6.

Q: Why is BESS INT8 poor?
A: Narrow feature bands; per-tensor MinMax; production uses BESS FP32 ONNX.

Q: How is edge latency measured?
A: evaluation --compare, 1000 CPU inferences; ONNX FP32 P95 ≈0.15 ms class.

Q: Macro-F1 vs accuracy?
A: Macro-F1 fairer under imbalance; per-class reports in onnx_fp32/classification_report.md.
```

**Figures**: None.

**旁白（中文，详细）**  
备份页不念，供追问扫读。数据合法性：合成 + seed 42 + version.txt。50500 构成与 split 35126/7768/7606 指 data card。INT8 答窄带 + MinMax + 生产 FP32。延迟答 evaluation 脚本千次推理。指标答 macro-F1 与 per-class report 并用。  
若被问「合成是否过于简单」，答：六轴 stress 正是为了检验离开训练分布时的行为；且 full pipeline 在真 HTTP 下已验证。  
保持回答短句，引用路径即可。

**Narration (English, detailed)**  
Backup—do not read verbatim. Synthetic seed-forty-two data with `version.txt`; fifty thousand five hundred composition in the data card; BESS INT8 narrow bands and FP32 production default; edge latency via thousand-iteration CPU bench; macro-F1 plus per-class reports.  
If synthetic seems too easy: six stress axes and live HTTP integration test departure from training assumptions. Answer in short sentences with paths.

---

### Slide 37 · Q&A backup (2/2) — agent & integration

**Layout**: Layout C.

**On-slide text (English only)**:

```
Q: Why llama3.2?
A: Local ~2 GiB; dev.yaml default; mock plan fallback on bad JSON.

Q: What does No-RAG prove?
A: RAG value is citation/audit, not surface rubric mean.

Q: Why not cloud_only?
A: P95 ≈ full but loses fault_class grounding — keep edge always-on.

Q: Three agent_recommend_failed?
A: Concurrent tail latency; orchestrator catches and continues — graceful degradation.

Q: Is C7 the real system?
A: inject.py shares NodeRunner HTTP; 11 dashboard inject unit tests; fault_injection_demo.md 5/5 ✅.

Q: Must we rerun everything?
A: Not if docs match artefacts; rerun only after code/weight/benchmark changes.
```

**Figures**: None.

**旁白（中文，详细）**  
智能体与集成高频题：模型选型、No-RAG 含义、cloud_only 不推荐、三次 timeout 仍算成功降解、C7 路径一致性、何时重跑。答句都指向 JSON/MD 证据，避免口头新数字。  
可补充：judge mean 4.10 来自 99/99 scored；144 events 来自 orchestrator meta。网页 UI 演示另场进行，本 deck 用 fault_injection_demo 表作 C7 证据。  
与 slide 36 一样，现场扫一眼即可，不要占用主讲时间。

**Narration (English, detailed)**  
High-frequency agent and integration questions—model choice, No-RAG meaning, cloud_only rejection, three timeouts as absorbed failures, C7 inject path parity, when to rerun—all point to JSON and markdown evidence without new oral numbers.  
Add judge four point one zero on ninety-nine scores and one hundred forty-four orchestrator events if needed. The browser UI demo is off-deck; this deck cites the five-scenario fault-injection report for C7. Scan, do not monopolise main talk time.

---

### Slide 38 · Thank you

**Layout**: Layout C, centered.

**On-slide text (English only)**:

```
Thank you

Questions welcome

Backup on second screen: final_report.pdf · appendix A6 (reproduction)
Contact / repository: [your Git URL or local path]
```

**Figures**: Optional QR code to repository (if permitted).

**旁白（中文，详细）**  
致谢页保持简洁。说明可在第二屏打开终稿 PDF 或附录 A6 复现命令，愿意针对任意 C1–C8 组件 deep dive。网页系统演示若尚未进行，可说明将按 `docs/网页演示指南.md` 单独展示。  
中英文都可收尾；语气温和、邀请提问。不要在此页展开新数字。  
记录评委问题，事后可更新 FAQ 进附录。

**Narration (English, detailed)**  
Keep thanks minimal. Offer a second screen with `final_report.pdf` or appendix A6 for reproduction questions and any C1–C8 deep dive. If the browser demo has not run yet, note it follows `docs/网页演示指南.md` as a separate segment. Invite questions warmly—no new numbers here. Note follow-up questions for FAQ updates.

---

## 三、Appendix Slides（附录 A1–A6，按需插入或放 deck 末尾）

> **Appendix convention**: **On-slide text = English only**; **旁白 + Narration = bilingual, detailed** (same as main slides). Insert after Slide 38 or hide until Q&A.

---

### Appendix A1 · Full class sample counts

**Layout**: Layout B, full table (14 pt).

**On-slide text (English only)**:

```
Class distribution — total 50,500 samples (docs/data_card.md §6)

| Class                         | Count | % of total |
| PV_Normal                     | 8,002 | 15.85%     |
| Partial_shading               | 3,333 | 6.60%      |
| Soiling                       | 3,333 | 6.60%      |
| Bypass_diode_fault            | 3,333 | 6.60%      |
| String_disconnection          | 3,333 | 6.60%      |
| Inverter_fault                | 3,333 | 6.60%      |
| Degradation                   | 3,333 | 6.60%      |
| BESS_Normal                   | 5,000 | 9.90%      |
| Capacity_fade                 | 4,375 | 8.66%      |
| Internal_resistance_increase  | 4,375 | 8.66%      |
| Thermal_anomaly               | 4,375 | 8.66%      |
| Cell_imbalance                | 4,375 | 8.66%      |

Max/min ratio ≈2.40 — Normal classes oversampled for false-alarm control
```

**Figures**: None.

**旁白（中文，详细）**  
附录 A1 回答「某一类有多少样本」。总计 50500，Normal 类最多，故障类各 3333 或 4375，比例差约 2.4 倍，训练用加权损失缓解。若评委质疑某类过少，指本表与 `data/splits/meta_*.csv` 交叉验证。  
不必主讲背诵，追问时翻页即可。数字与 2026-06-03 data card 完全一致。

**Narration (English, detailed)**  
Appendix A1 lists per-class counts totaling fifty thousand five hundred—Normals oversampled, faults balanced via weighted loss. Point to split CSVs if challenged. Numbers match the 2026-06-03 data card—flip to this slide on demand, do not memorise in the main talk.

---

### Appendix A2 · Simulation formulas (summary)

**Layout**: Layout C, two columns PV | BESS.

**On-slide text (English only)**:

```
PV (physics-inspired, docs/data_card.md §3)
• NOCT-style module temperature from T_amb, G
• I_dc, V_dc, P, P_ac, eta from irradiance and electrical constraints

BESS (RC equivalent-circuit flavour)
• SOC, V_term, I, T, R_est, sigma_V, N_cycle, SoH trajectories
• Ageing and resistance cues drive fault separability (and INT8 fragility)

fault_injector: one pure function per fault · seedable numpy RNG
```

**Figures**: Optional data_card §3 pipeline diagram screenshot.

**旁白（中文，详细）**  
A2 给关心「仿真物理是否靠谱」的老师。PV 用简化 NOCT 与 DC/AC 功率关系；BESS 用 RC 等效与 SOC/内阻/循环计数代理量。故障注入对每个 label 独立纯函数，seed 可控。  
强调：仿真不是为了追真实电站每一伏特，而是为了可复现、可应力测试、可写 data card。真数据禁止时，这是合规且可辩护的路径。

**Narration (English, detailed)**  
Appendix A2 summarises PV NOCT-style thermal and electrical relationships and BESS RC-flavoured SOC/resistance trajectories—seedable per-fault injectors. The goal is reproducible stress testing and a complete data card, not perfect plant cloning—appropriate when real data is forbidden.

---

### Appendix A3 · Robustness figure index (PV)

**Layout**: Layout C, 3×3 thumbnail grid.

**On-slide text (English only)**:

```
PV robustness figures (reports/robustness/pv/figures/)
overview_macro_f1 · scale_drift_curve · noise_curve · fgsm_curve
missing_features_curve · condition_heatmap · confidence_sensitivity
ood_energy_histogram · risk_coverage_curve
```

**Figures**: Insert nine thumbnails from paths above (~15% width each).

**旁白（中文，详细）**  
A3 是 PV 九图导航，答辩若被问「某轴曲线在哪」直接翻页指缩略图。主文已用 heatmap、histogram、risk-coverage，其余可在 Q&A 展开。路径均在 `reports/robustness/pv/figures/`。

**Narration (English, detailed)**  
Appendix A3 indexes nine PV robustness figures—use thumbnails to answer “where is the noise curve?” Main slides already cite key plots; others are one page away under `reports/robustness/pv/figures/`.

---

### Appendix A4 · Robustness figure index (BESS)

**Layout**: Layout C, 3×3 thumbnail grid.

**On-slide text (English only)**:

```
BESS robustness figures (reports/robustness/bess/figures/)
Same filenames as PV — overview_macro_f1 through risk_coverage_curve
```

**Figures**: Nine thumbnails under `reports/robustness/bess/figures/`.

**旁白（中文，详细）**  
A4 与 A3 对称，BESS 侧 FGSM 与 missing features 通常更「难看」，适合解释 INT8 与 adversarial 脆弱性同一根源。指图时强调 2026-06-03 同批生成。

**Narration (English, detailed)**  
Appendix A4 mirrors A3 for BESS—FGSM and missing-feature plots often look worse, supporting the INT8 and adversarial narrative. All generated in the same 2026-06-03 batch.

---

### Appendix A5 · Agent eval keyword-drift examples

**Layout**: Layout C.

**On-slide text (English only)**:

```
Sub-1.0 scores — predominantly keywords slot (reports/agent_eval.md §8)

Examples (full ablation)
• scn_ambiguous_mixed_string_id — score 0.75 · failed: keywords · kb=3
• scn_ambiguous_bess_cell_imbalance_warning — 0.75 · keywords · kb=3
• scn_ambiguous_pv_bypass_diode_warning — 0.75 · keywords · kb=2

No-RAG may still show 0.75 with kb=0 — plausible text, zero provenance
Interpretation: lexical drift vs oracle phrases; urgency/forbidden remain strong
```

**Figures**: None.

**旁白（中文，详细）**  
A5 解释「为什么 keywords 不是 100%」。ambiguous 场景故意模糊，LLM 措辞与 oracle 短语不完全匹配，启发式扣在 keywords 槽。安全槽 urgency/forbidden/knowledge 仍稳。可对比 No-RAG 同场景 kb=0 仍 0.75，说明 surface 分误导。  
追问「举一条失败例」时读场景 ID 即可，不必展开全文 recommendation。

**Narration (English, detailed)**  
Appendix A5 explains keyword drift—ambiguous scenarios where LLM wording misses oracle phrases while safety slots hold. Contrast No-RAG: score may stay zero point seven five with zero KB—provenance failure masked by rubric. Cite scenario IDs, not full text.

---

### Appendix A6 · Full reproduction commands (PowerShell)

**Layout**: Layout C, monospace 14 pt.

**On-slide text (English only)**:

```
One-shot reproduction (复现指南.md §4 — excerpt)
cd <repo> · pip install -e ".[dev]" · pytest tests -q
python -m simulation.generate_dataset --seed 42 --n-pv 28000 --n-bess 22500 ...
python -m training.train --system pv|bess ...
python -m quantization.onnx_export / int8_static
python -m evaluation --compare
python scripts/run_robustness_eval.py
python -m rag.ingest
# Terminals A/B: uvicorn edge :8000 · agent :8001
$env:AGENTPV_JUDGE_API_BASE='http://127.0.0.1:11434/v1'
$env:AGENTPV_JUDGE_MODEL='llama3.2:latest'
python -m agent_eval --ablations full no_retrieve_knowledge no_reasoning_trace ...
python scripts/e2e_latency_bench.py --mode edge_only|full|cloud_only ...
python -m orchestrator --nodes pv6_bess4 --duration 60 --http-timeout 120
python scripts/demo_fault_injection.py
python scripts/render_final_report.py

Full copy-paste block: 复现指南.md §4 (lines 501–556)
```

**Figures**: None.

**旁白（中文，详细）**  
A6 放完整 PowerShell 块给「怎么复现」追问。幻灯片只列骨架，完整可复制段落在 `复现指南.md` §4，含 judge 环境变量与 http-timeout 120。强调 Ollama 需先 serve，agent_eval 耗时长属正常。  
若评委要 overnight 复现，指 §4 顺序与 §5 自检清单（pytest + Test-Path 产物）。

**Narration (English, detailed)**  
Appendix A6 holds the reproduction skeleton; the full copy-paste block is `复现指南.md` section four with judge variables and orchestrator timeout one hundred twenty. Note Ollama must be running—agent_eval is slow with a real LLM. Point to section five checklists for verification.

---

## 四、Figure Quick Reference（配图速查表）

| Section | Recommended image paths |
|---------|-------------------------|
| C3 confusion | `reports/pv/onnx_fp32/confusion_matrix.png`, `reports/bess/onnx_fp32/…`, `reports/bess/onnx_int8/…` |
| C3 trade-off | `reports/pv/comparison_tradeoff.png`, `reports/bess/comparison_tradeoff.png` |
| Robustness | `reports/robustness/{pv,bess}/figures/*.png` (6 cited in main deck) |
| C5 agent | `reports/figures/agent_eval/*.png` (all 6 used) |
| C6 integration | `reports/figures/integration/01～05_*.png` (all 5 used) |
| C7 demo | `reports/integration/fault_injection_demo.md` (+ optional UI architecture diagram) |
| Final report | `reports/final_report.pdf` |

---

## 五、Production & Rehearsal Checklist（制作与排练）

- [ ] Master slide: white / `#F7F8FA` background; titles `#1A365D`; page numbers `n/44`
- [ ] **All on-slide text English only** (titles, bullets, tables, figure captions)
- [ ] **Narration bilingual and detailed** under each slide (do not paste narration onto slides)
- [ ] All numbers match `reports/final_report.md` (2026-06-03 refresh)
- [ ] Data slides cite source paths in footer (e.g. `integration_eval_meta.json`)
- [ ] Figures ≤55% content area; no blurry stretch
- [ ] Rehearse Chinese and English scripts; terminology matches English slides
- [ ] Pre-defence: `pytest tests -q` green; confirm artefact paths in slide 35 table exist
- [ ] Second screen: `final_report.pdf` and/or appendix A6 for “how to reproduce?”
- [ ] **Do not** mention Docker Compose in the talk; **do not** walk through Streamlit in this deck
- [ ] Web UI demo (if scheduled): separate session per `docs/网页演示指南.md`
- [ ] Timing: main talk ~30–35 min + Q&A ~10 min (browser demo is off-deck)

---

## 六、Related Documentation（与仓库其他文档的关系）

| Document | Role in this deck |
|----------|-------------------|
| `reports/final_report.pdf` | Authoritative numbers and narrative (2026-06-03) |
| `docs/网页演示指南.md` | Off-deck browser demo (not in this slide deck) |
| `复现指南.md` | Slides 6, 35, appendix A6 |
| `docs/data_card.md` | C1 slides + appendix A1/A2 |
| `ppt旁白.md` | 仅中英旁白（主汇报 Slide 1–30）；`python scripts/extract_presentation_narration.py` 可再提取 |
| `Q&A.md` | 答辩 Q&A 手册（中英 + 对应幻灯片页码） |

---

*Guide version: synced with 2026-06-03 full-pipeline artefacts (`final_report.md`, `last_run_three_ablations_with_judge.json`, `integration_eval_meta.json`, `fault_injection_demo.md`). Main slides 1–38 on-slide English only; narration bilingual detailed per slide. No Docker Compose; no live web demo in deck.*
