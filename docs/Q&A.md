# AgentPV 答辩 Q&A 手册

> **主汇报止于 Slide 30**（C7 看板 + 现场演示）。Slide 31–38 与附录 A1–A6 供 Q&A 速查。  
> 口头回答宜短句 + 指向幻灯片/报告路径；数字以 `reports/final_report.pdf`（2026-06-03）为准。

---

## 使用说明

| 字段 | 含义 |
|------|------|
| **参考幻灯片** | 答辩时翻到哪一页、看哪块内容（表格/图/bullet） |
| **中文** | 主答话术 |
| **English** | 双语答辩或英文追问时使用 |

---

## 一、项目总览与团队

### Q1. AgentPV 是什么意思？项目解决什么问题？

**参考幻灯片**：Slide 1 · Title — 项目全称；云边分层设计说明

**中文**  
**问**：AgentPV 这个名字怎么理解？你们解决什么问题？  
**答**：AgentPV 中 Agent 指云端带工具与知识库的大模型智能体，PV 指光伏场景，并扩展到储能 BESS。我们要同时满足两件事：边缘在有限算力下毫秒级给出可靠告警；运维人员还要知道怎么处理、多紧急、依据哪条规程——不能只给一个故障标签。

**English**  
**Q**: What does “AgentPV” mean, and what problem does it solve?  
**A**: “Agent” is the cloud LLM tier with tools and a knowledge base; “PV” is photovoltaics, extended to BESS. We address two needs: fast, reliable edge alerts under limited compute, and operator-facing recommendations with urgency and auditable procedure citations—not just a fault label.

---

### Q2. 组员分工 / 团队介绍？

**参考幻灯片**：Slide 1 · Title — 组员姓名（口头补充，幻灯片可写 Presenter 行）

**中文**  
**问**：团队是谁？  
**答**：我们三位组员是夏梓峻、董学语和蔡明仕。今天由我们共同汇报；Slide 30 的可视化看板可由组员现场演示。

**English**  
**Q**: Who is on the team?  
**A**: We are Zijun Xia, Xueyu Dong, and Mingshi Cai. We present together; a teammate can demo the Streamlit dashboard at Slide 30.

---

### Q3. 今天汇报结构是什么？讲到哪里结束？

**参考幻灯片**：Slide 2 · Agenda — Part I–IX 路线；Slide 6 · C1–C8 合规表

**中文**  
**问**：汇报顺序？  
**答**：从动机、贡献、架构开始，按数据 → 边缘模型 → 鲁棒性 → 智能体评测 → 系统集成 → C7 看板，主汇报在 Slide 30 结束；工程质量、局限、复现和更多 Q&A 在附录页，需要时再翻。

**English**  
**Q**: What is the talk structure? Where does the main presentation end?  
**A**: Motivation, contributions, architecture, then data, edge models, robustness, agent evaluation, integration, and the C7 dashboard. The main talk ends at Slide 30; engineering quality, limitations, reproducibility, and extra Q&A are on backup and appendix slides.

---

### Q4. 六大贡献/核心亮点是什么？

**参考幻灯片**：Slide 5 · Six contributions — 六条 numbered list；右下角 latency 小图（C6 预告）

**中文**  
**问**：用三句话概括贡献？  
**答**：全链路可复现代码 + 284 测试；PV/BESS 各三种部署后端完整对比；六轴鲁棒性 + 选择性预测；33 场景智能体消融 + judge 4.1；真 HTTP 三模式 + 十节点并发；双语看板 + 脚本化故障注入五场景全过。

**English**  
**Q**: Summarise your main contributions.  
**A**: Reproducible pipeline with 284 tests; PV and BESS each with PyTorch, ONNX FP32, and INT8 compared honestly; six stress tests plus selective prediction; 33 agent scenarios with ablations and judge score 4.1; real HTTP integration and ten-node concurrency; bilingual dashboard and scripted fault injection—all five scenarios pass.

---

## 二、架构与接口

### Q5. 为什么要云边协同，而不是只放云端或只放边缘？

**参考幻灯片**：Slide 3 · Problem — Industrial pain points；Two competing requirements；AgentPV design 三 bullet

**中文**  
**问**：为什么必须云边一起设计？  
**答**：边缘要低开销、断网也能告警；云端要可执行的运维建议 + 知识库引用。只云端：断网时没有快速告警；只边缘：运维只有标签没有规程依据。我们两层用 HTTP 契约连接，测试、调度、注入都走同一套 API。

**English**  
**Q**: Why co-design cloud and edge instead of one tier only?  
**A**: The edge must alert quickly and work offline; the cloud must give actionable, auditable recommendations. Cloud-only loses fast local alerts; edge-only loses procedure grounding. We link both with shared HTTP APIs used in tests, scheduling, and fault injection.

---

### Q6. 系统分几层？各层做什么？

**参考幻灯片**：Slide 7 · System architecture — Four layers ①–④；HTTP contracts；Service ports

**中文**  
**问**：架构怎么分层？  
**答**：四层——数据层（仿真 + split + version.txt）；边缘层（`/predict` → Alert）；云端（`/recommend` → Recommendation + knowledge_sources）；操作层（调度器 + 看板，JSONL 事件与故障注入）。边缘 8000、智能体 8001、Ollama 11434。

**English**  
**Q**: Describe the four layers.  
**A**: Data (simulation, splits, `version.txt`); edge (`/predict` → Alert); cloud (`/recommend` → Recommendation with citations); operations (scheduler + dashboard, JSONL, fault inject). Ports: edge 8000, agent 8001, Ollama 11434.

---

### Q7. 告警什么情况下才调用智能体？

**参考幻灯片**：Slide 11 · Fault taxonomy — severity 三档说明；Slide 21 · Alert → Recommendation — 数据流；Slide 30 · C7 表 — Scenario 4 monitor / Scenario 5 skip_agent

**中文**  
**问**：是不是每个告警都会调 LLM？  
**答**：不是。只有 **warning** 和 **critical** 调云端；**monitor**（如 PV_Normal）只在边缘处理，省约 8.5 s 和算力。C7 场景 4 证明 monitor 跳过 agent；场景 5 用 skip_agent 演示 LLM 不可用仍有 critical Alert。

**English**  
**Q**: Does every alert call the LLM?  
**A**: No. Only **warning** and **critical** call the cloud; **monitor** stays edge-only—saving ~8.5 s and compute. C7 scenario 4 skips the agent; scenario 5 shows a critical alert with `skip_agent=True`.

---

### Q8. 数据从传感器到看板完整怎么走？

**参考幻灯片**：Slide 21 · Alert → Recommendation data flow — 流程 bullets；Slide 7 · ④ Ops UI

**中文**  
**问**：端到端数据流？  
**答**：SensorWindow → 边缘 Alert →（若 warning/critical）→ 智能体 Recommendation → 写入 JSONL → 看板时间线与详情 Tab。monitor 在边缘终止，不进入 agent。

**English**  
**Q**: Walk through the end-to-end data flow.  
**A**: SensorWindow → edge Alert → (if warning/critical) → agent Recommendation → JSONL → dashboard timeline and detail tabs. Monitor alerts stop at the edge.

---

## 三、数据（C1）

### Q9. 为什么用仿真数据，不用真实电站数据？

**参考幻灯片**：Slide 9 · C1 — Why synthetic data? — Course constraint；simulators + seed 42；Slide 36 · Q&A backup (1/2) 第一问

**中文**  
**问**：不用真实数据是否缺乏说服力？  
**答**：课程禁止直接用 Kaggle/HF 等现成集，真实电站数据通常涉密。我们用物理启发仿真 + 固定 seed 42 + `data/version.txt` 保证可复现；六轴鲁棒性和真 HTTP 集成用来检验「离开训练分布」和「整条链路」而不仅是离线 F1。

**English**  
**Q**: Why synthetic data instead of real plant data?  
**A**: The course forbids off-the-shelf public datasets; real plant data is often private. We use physics-inspired simulation with seed 42 and `data/version.txt` for reproducibility. Six stress axes and live HTTP tests check behaviour beyond clean offline F1.

---

### Q10. 50500 样本怎么构成？划分会不会泄漏？

**参考幻灯片**：Slide 10 · Dataset scale & splits — 表格 Metric/Value；Train/Val/Test 35126/7768/7606；Split policy 行

**中文**  
**问**：50,500 怎么来的？  
**答**：PV 28,000 + BESS 22,500，共 50,500；60 s × 8 通道 @ 1 Hz。按 **system_id 分层**划分约 70/15/15，同一模拟资产不会同时出现在 train 和 test，避免虚高 F1。详见 `docs/data_card.md` 与 `data/version.txt`。

**English**  
**Q**: How is 50,500 built? Is there split leakage?  
**A**: 28,000 PV + 22,500 BESS; 60 s × 8 channels at 1 Hz. **Stratified by `system_id`** (~70/15/15)—no asset appears in both train and test. See `docs/data_card.md` and `data/version.txt`.

---

### Q11. 为什么是 11 类故障？PV 和 BESS 用一个模型吗？

**参考幻灯片**：Slide 11 · Fault taxonomy — 11 classes 列表；Slide 6 · C1 行

**中文**  
**问**：11 类怎么对应作业要求？  
**答**：作业要求 PV≥7、BESS≥5 类；我们 PV 7 + BESS 5 = 11。PV 与 BESS **共享 CNN 结构、不共享权重**，各训一套；云端统一收 Alert JSON。

**English**  
**Q**: Why eleven classes? One model for both systems?  
**A**: Assignment requires ≥7 PV and ≥5 BESS labels. We use **shared architecture, separate weights** per system; the cloud sees a unified Alert schema.

---

### Q12. 三种工况与八通道特征是什么？

**参考幻灯片**：Slide 12 · Operating conditions — Condition 表 5:3:2；八通道 bullet；Slide 19 · condition heatmap 图

**中文**  
**问**：工况和特征通道？  
**答**：高辐照 / 低辐照 / 高温，采样权重 5:3:2；每类故障在每种工况都有样本。八通道顺序固定在 `SensorWindow` 与 NPZ，鲁棒性缺通道实验按索引 mask。工况热力图（Slide 19）显示 macro-F1 跨工况仍高。

**English**  
**Q**: Operating conditions and eight channels?  
**A**: High/low irradiance and high temperature—weights 5:3:2; every fault appears in every condition. Eight channels are order-locked in `SensorWindow` and NPZ. The condition heatmap (Slide 19) shows stable macro-F1 across regimes.

---

### Q13. （附录）每一类样本多少？

**参考幻灯片**：Appendix A1 · Full class sample counts — 完整 11 类 Count 表

**中文**  
**问**：某类样本会不会太少？  
**答**：翻 Appendix A1：总计 50,500，Normal 略多，故障类约 3333 或 4375，比例差约 2.4×；训练用加权损失。可交叉 `data/splits/meta_*.csv`。

**English**  
**Q**: Per-class sample counts?  
**A**: See Appendix A1—50,500 total; Normal classes oversampled; faults balanced with weighted loss. Cross-check `data/splits/meta_*.csv`.

---

## 四、边缘模型与评测（C2/C3）

### Q14. 为什么用 1D-CNN，不用 Transformer / LSTM？

**参考幻灯片**：Slide 13 · CNN-1D edge architecture — 参数量 ~48k；ONNX P95；Q&A 提示

**中文**  
**问**：为何选小 CNN？  
**答**：作业强调部署预算：约 4.8 万参数，ONNX FP32 P95 约 **0.15 ms**，远小于 100 ms 上限，且可导出、可 INT8。Transformer/LSTM 更重，边缘网关不友好。

**English**  
**Q**: Why 1D-CNN instead of Transformer or LSTM?  
**A**: Deployment budgets: ~48k parameters, ONNX FP32 P95 ~**0.15 ms**, well under 100 ms, exportable and quantisable. Transformers/LSTMs are heavier for edge gateways.

---

### Q15. 三种后端对比结论？PV 和 BESS INT8 一样吗？

**参考幻灯片**：Slide 14 · Three-backend comparison — 6 行 System/Variant 表；Slide 15 · 第三张 BESS INT8 混淆矩阵

**中文**  
**问**：PyTorch / ONNX / INT8 结论？  
**答**：PV 三线 macro-F1 均 **0.9994**，INT8 几乎无损，体积 0.184→0.058 MiB。BESS FP32 **0.9980**，INT8 仅 **0.7058**——窄特征带 + per-tensor MinMax 导致类间混淆（Slide 15）。生产建议 BESS 用 **FP32 ONNX**。

**English**  
**Q**: Three backends—headline conclusions?  
**A**: PV: 0.9994 macro-F1 on all backends; INT8 nearly lossless. BESS FP32: 0.9980; INT8: 0.7058—narrow bands plus per-tensor MinMax (Slide 15). Deploy BESS on **FP32 ONNX**.

---

### Q16. 为什么要报告 BESS INT8 失败？

**参考幻灯片**：Slide 14 · BESS onnx_int8 行 ⚠；Slide 15 · INT8 confusion；Slide 5 · contribution 2「honest」

**中文**  
**问**：INT8 失败是否说明项目不完整？  
**答**：相反——作业要求压缩 **trade-off 分析**。我们如实报告失败边界和修复方向（按通道量化、熵校准），比只报 PV 成功更有可信度。PV INT8 可上线；BESS INT8 暂不建议。

**English**  
**Q**: Why report BESS INT8 failure?  
**A**: The assignment asks for honest compression trade-offs. We document failure modes and fixes (per-channel quant, entropy calibration). That strengthens credibility. PV INT8 is fine; BESS INT8 is not recommended yet.

---

### Q17. Macro-F1 和 Accuracy 看哪个？

**参考幻灯片**：Slide 14 · Macro-F1 列；Slide 36 · Q&A backup — Macro-F1 vs accuracy

**中文**  
**问**：为什么强调 Macro-F1？  
**答**：类别略不平衡，Macro-F1 对每类同等权重更公平；per-class 报告在 `reports/pv/onnx_fp32/classification_report.md` 等路径。

**English**  
**Q**: Macro-F1 vs accuracy?  
**A**: Mild imbalance—macro-F1 is fairer. Per-class reports under `reports/pv/onnx_fp32/classification_report.md` and BESS equivalents.

---

## 五、鲁棒性与选择性预测

### Q18. 六轴鲁棒性测了什么？

**参考幻灯片**：Slide 16 · deployment realism — 六 bullet stress axes；Slide 18 · 成功/失败边界

**中文**  
**问**：干净 F1 很高，还要鲁棒性干什么？  
**答**：应导师 deployment realism 反馈：六轴——工况偏移、缺通道、噪声、漂移、FGSM、跨系统 OOD。Slide 18 明确哪些能扛（轻噪声/轻漂移 F1>0.9）、哪些不能（缺 10% 通道掉 40 点且 overconfident）。

**English**  
**Q**: What do the six robustness axes test?  
**A**: Condition shift, missing channels, noise, drift, FGSM, cross-system OOD. Slide 18 states success (mild noise/drift) and failure (10% missing channels, strong drift/attacks).

---

### Q19. 选择性预测怎么用？95% coverage 什么意思？

**参考幻灯片**：Slide 17 · selective prediction 表 — Selective acc @95% cov；Slide 16 · energy-based 机制 bullet

**中文**  
**问**：选择性预测解决什么？  
**答**：用 energy score：不确定就拒绝分类、交人工。阈值在 validation 上校准到 **95% coverage**——约 95% 正常样本仍自动处理，最不确定的 5% 拒绝。分布内 selective accuracy 近 1（Slide 17 表）。

**English**  
**Q**: How does selective prediction work?  
**A**: Energy score—reject uncertain inputs for human review. Calibrated to **95% validation coverage**—~95% of normal cases auto-handled. In-distribution selective accuracy ≈1 (Slide 17 table).

---

### Q20. OOD 能量分数方向反了怎么办？

**参考幻灯片**：Slide 17 · Score direction inverted 行；Slide 19 · OOD energy histogram；Slide 33 · Future ② Auto-flip

**中文**  
**问**：OOD AUROC 还行但方向反了？  
**答**：跨系统 OOD 时高 energy 反而像 in-distribution（Slide 19  histogram）。不能盲用默认阈值——需 **flip 规则** 或 **Mahalanobis** 补充；我们在报告和 Slide 33 future work 里写了，不是隐藏问题。

**English**  
**Q**: Inverted OOD energy direction?  
**A**: Cross-system OOD inverts the score (Slide 19 histogram). Do not use the default threshold blindly—use a flip rule or Mahalanobis fallback; documented in the report and Slide 33 future work.

---

### Q21. （附录）某条鲁棒性曲线在哪？

**参考幻灯片**：Appendix A3 · PV figure index；Appendix A4 · BESS figure index — 九图文件名 grid

**中文**  
**问**：噪声曲线 / FGSM 曲线在哪？  
**答**：PV 在 `reports/robustness/pv/figures/`，BESS 对称路径；Appendix A3/A4 缩略图索引。主文 Slide 19 已引 OOD histogram 与 condition heatmap。

**English**  
**Q**: Where is a specific robustness plot?  
**A**: `reports/robustness/pv/figures/` or `bess/figures/`; see Appendix A3/A4 thumbnails.

---

## 六、智能体（C4/C5）

### Q22. ReAct 智能体有哪些工具？RAG 文档多少？

**参考幻灯片**：Slide 20 · ReAct architecture — 五阶段 + 四 Tool 表；Slide 6 · C4 行

**中文**  
**问**：智能体能力边界？  
**答**：ReAct 五阶段 + 四工具：`retrieve_knowledge`（30 篇 playbook + Chroma）、`system_history`、`estimate_rul`、`escalate_alert`。Grounding 核心是 retrieve_knowledge；Citation 带 chunk_id 可审计。

**English**  
**Q**: What tools does the ReAct agent have?  
**A**: Five ReAct stages plus four tools—`retrieve_knowledge` (30 playbooks, Chroma), `system_history`, `estimate_rul`, `escalate_alert`. Grounding hinges on retrieval; citations include chunk_id.

---

### Q23. 为什么用 llama3.2？JSON 解析失败怎么办？

**参考幻灯片**：Slide 20 · ollama_plan_fallback_mock bullet；Slide 37 · Q&A backup — Why llama3.2?

**中文**  
**问**：模型选型与稳定性？  
**答**：本地 Ollama **llama3.2**（约 2 GiB），`dev.yaml` 默认。Planner JSON 不稳定时走 **`ollama_plan_fallback_mock`** 确定性计划，工具仍执行、HTTP 仍 200——与 C6/C7 graceful degradation 一致。

**English**  
**Q**: Why llama3.2? Bad JSON?  
**A**: Local Ollama llama3.2 (~2 GiB). Invalid planner JSON triggers **`ollama_plan_fallback_mock`**—tools still run, HTTP 200—consistent with graceful degradation.

---

### Q24. 33 场景 benchmark 怎么设计？如何打分？

**参考幻灯片**：Slide 22 · Agent benchmark design — 33 scenarios；dual scoring；三 ablation 名

**中文**  
**问**：Benchmark 可信度？  
**答**：33 个 oracle JSON 场景（10 个故意模糊）；runner **真 edge 分类 in the loop**，非手写 fault_class。双评分：规则 rubric 0–1 + LLM judge 1–5（本轮 99/99，均值 **4.10**）。三消融：full / no_retrieve_knowledge / no_reasoning_trace，共 99 条记录。

**English**  
**Q**: How is the 33-scenario benchmark designed and scored?  
**A**: Oracle JSON scenarios (10 ambiguous); real edge classifier in the loop. Rule rubric plus LLM judge (99/99 scored, mean **4.10**). Three ablations—99 records total.

---

### Q25. No-RAG 消融说明什么？（最重要）

**参考幻灯片**：Slide 24 · No-RAG grounding collapse — mean vs KB/scenario；Slide 23 · Ablation 表 No-RAG 行；Slide 25 · kb_sources 图（若有）

**中文**  
**问**：关掉 RAG 差别大吗？  
**答**：**表面分几乎不变**（0.93→0.92），但 **KB 引用从 2.61/场景、91% 有引用 → 全 0**。话术仍合理，**无法审计规程依据**——工业场景不可接受。RAG 价值是 **traceability**，不是刷 rubric 均值。

**English**  
**Q**: What does the No-RAG ablation prove?  
**A**: Surface score barely changes (0.93→0.92), but citations drop from ~2.61/scenario and 91% grounded to **zero**. Text still sounds fine but is **not auditable**. RAG’s value is **traceability**, not the composite mean.

---

### Q26. keywords 为什么不是 100%？

**参考幻灯片**：Slide 23 · keywords 72.7%；Slide 25 · 分布图 sub-1.0；Appendix A5 · keyword-drift 场景 ID

**中文**  
**问**：哪些槽位会丢分？  
**答**：主要是 **keywords** 措辞与 oracle 短语不完全匹配（ambiguous 场景）；**urgency / forbidden / knowledge 硬槽位 100%**。Appendix A5 可举 `scn_ambiguous_*` 场景 ID，不必念全文 recommendation。

**English**  
**Q**: Why not 100% on keywords?  
**A**: Lexical drift on ambiguous scenarios—LLM wording vs oracle phrases. **Urgency, forbidden, and knowledge stay at 100%**. See scenario IDs in Appendix A5.

---

### Q27. No-Trace 和 No-RAG 应优先砍哪个？

**参考幻灯片**：Slide 23 · No-Trace 行；Slide 24 · 末段 No-Trace 对照；Slide 26 · ReAct depth 图

**中文**  
**问**：带宽不够先关什么？  
**答**：**先关 trace，不先关 RAG**。No-Trace mean 略降但 KB 仍在；No-RAG 则 grounding 全灭。Slide 26 显示 No-Trace 步数压到 1，适合带宽受限降级。

**English**  
**Q**: Under bandwidth pressure, drop trace or RAG first?  
**A**: **Drop trace before RAG**. No-Trace lowers mean slightly but keeps citations; No-RAG zeroes grounding. Slide 26 shows one-step degradation.

---

## 七、系统集成（C6）

### Q28. 三种 integration mode 延迟各多少？瓶颈在哪？

**参考幻灯片**：Slide 27 · Three-mode latency — Mode/P50/P95/P99 表；decomposition bullets；`01_latency_bars.png`

**中文**  
**问**：E2E 是否满足 10 s？瓶颈？  
**答**：真 HTTP，每模式 50 次（弃 3 次 warmup）。**edge_only P95 5.69 ms**；**full P95 9803 ms**；**cloud_only P95 9941 ms**——均 <10 s。分解：edge **26 ms（<1%）**，agent **~9785 ms（>99%）**——优化应优先 LLM，不是砍 CNN。

**English**  
**Q**: Latency by mode? Bottleneck?  
**A**: Live HTTP, 50 runs/mode. **edge_only P95 5.69 ms**; **full P95 9803 ms**; **cloud_only P95 9941 ms**—all under 10 s. Edge ~26 ms (<1%); agent ~9785 ms (>99%)—optimise the LLM path first.

---

### Q29. 为什么推荐 edge always-on，不推荐 cloud_only？

**参考幻灯片**：Slide 28 · Mode interpretation — 三条 numbered mode；edge_only / cloud_only 对比

**中文**  
**问**：能否只要云端？  
**答**：**cloud_only** 与 full 差不多慢，却**没有边缘 ML 给出的 fault_class**，agent 工具选择与 grounding 变弱。部署建议：**边缘常开**；cloud_only 仅作对比实验，不作生产默认。

**English**  
**Q**: Why not cloud_only in production?  
**A**: Similar latency to full but **loses edge fault labels** and weakens agent grounding. Keep **edge always on**; cloud_only is for comparison only.

---

### Q30. 十节点并发 144 events / 3 failures 说明什么？

**参考幻灯片**：Slide 29 · 10-node orchestrator — 144 events；Error kind 表 agent_recommend_failed=3；`04_node_fanout.png`

**中文**  
**问**：3 次 agent 失败算项目失败吗？  
**答**：不算——这是**真实 tail latency** 下 httpx 超时，非 mock。关键：**调度器未崩溃**，各节点独立 tick，边缘仍出 Alert，部分节点仍收到 Recommendation——**graceful degradation**。证据是 failure count + 系统存活，不是零失败。

**English**  
**Q**: Are three agent failures acceptable?  
**A**: Yes—as **documented tail behaviour** under concurrency, not mocks. The orchestrator survived; edges still alerted; some recommendations still arrived. Evidence is bounded failures plus system survival.

---

## 八、C7 看板与演示

### Q31. C7 交付了什么？看板是不是 mock？

**参考幻灯片**：Slide 30 · C7 — Operator UI 四 Tab + inject.py 同路径；五场景 OK 表；Scenario 3 highlight

**中文**  
**问**：C7 如何验证？  
**答**：两层——`dashboard/app.py` 双语四 Tab，inject 走 **`dashboard/inject.py`** 与编排器 **同 HTTP 路径**（11 条 inject 单测）；`demo_fault_injection.py` → **`fault_injection_demo.md` 五场景全 ✅**，事件进 `events_c7_demo.jsonl`。Slide 30 表可口述；界面由组员 **Streamlit 现场演示**。

**English**  
**Q**: What does C7 deliver? Is the dashboard real?  
**A**: Bilingual four-tab UI; inject via **`dashboard/inject.py`**—same HTTP path as the orchestrator (11 unit tests). Scripted **`fault_injection_demo.md`**—all five scenarios pass. Slide 30 table for oral backup; live Streamlit demo by a teammate.

---

### Q32. 五场景故障注入各代表什么？

**参考幻灯片**：Slide 30 · 五场景表 — #1–5 Severity/Urgency/Edge/Agent/KB 列

**中文**  
**问**：五场景含义？  
**答**：1–3 全链路 + 3 KB（PV critical/warning、BESS thermal）；**4** monitor 不触发 agent；**5** skip_agent 仍 critical Alert（LLM 不可用降级）。场景 3 引用 `kb_thermal_anomaly_bess.md` 体现云边价值。

**English**  
**Q**: What do the five fault-injection scenarios show?  
**A**: 1–3: full pipeline with three KB hits; **4**: monitor skips agent; **5**: critical alert with `skip_agent`. Scenario 3 cites `kb_thermal_anomaly_bess.md`—cloud value over a bare label.

---

## 九、工程质量、局限与复现（附录幻灯片）

### Q33. 测试覆盖与代码质量？

**参考幻灯片**：Slide 31 · Engineering quality — 284 tests；test_dashboard_inject 11 tests；Ruff 0

**中文**  
**问**：怎么保证工程质量？  
**答**：**284** pytest 全绿，含 dashboard inject MockTransport **11** 例；终稿声明 ruff 0 warning。集成/agent/C7 需本机 edge+agent+Ollama，步骤见 `复现指南.md` §4。

**English**  
**Q**: Test coverage and code quality?  
**A**: **284** passing tests including **11** inject mocks; ruff zero warnings in the final report. Integration/agent/C7 need local services—see `复现指南.md` §4.

---

### Q34. 项目主要局限与未来工作？

**参考幻灯片**：Slide 33 · Limitations | Future 两栏；Slide 32 · what worked ✅ 对照

**中文**  
**问**：如果重做会先改什么？  
**答**：局限：BESS INT8、OOD 方向、缺通道无上游 gate、full P95 近 10 s、planner JSON、看板手动刷新 JSONL、仅合成数据。优先 future：**BESS 量化策略 + 更快 7–8B LLM**，再 adversarial training。主动讲局限通常比等提问得分高。

**English**  
**Q**: Main limitations and what you would fix first?  
**A**: BESS INT8, inverted OOD, missing-channel gap, tail latency near 10 s, planner JSON, manual JSONL refresh, synthetic-only data. First fixes: **BESS quant policy + faster local LLM**, then adversarial training. Proactive limitations help.

---

### Q35. 如何复现？必须重跑全链路吗？

**参考幻灯片**：Slide 35 · Reproducibility — 命令链 + artefact 表；Appendix A6 · PowerShell 块；Slide 36–37 · when to rerun

**中文**  
**问**：答辩要不要 overnight 重跑？  
**答**：文档与 2026-06-03 产物对齐则**不必**重跑全链路；改代码/权重/benchmark 才重跑对应步。一键块在 **`复现指南.md` §4** 与 **`final_report.md` §11.3**（含 judge 环境变量、orchestrator `--http-timeout 120`）。Appendix A6 可投屏复制。

**English**  
**Q**: Must we rerun everything to reproduce?  
**A**: Not if docs match the 2026-06-03 artefacts—rerun only what changed. One-shot blocks in **`复现指南.md` §4** and **`final_report.md` §11.3**. Appendix A6 for copy-paste.

---

### Q36. 数字权威来源在哪？

**参考幻灯片**：Slide 35 · artefact 表 — integration_eval_meta.json、last_run_three_ablations_with_judge.json 等；Slide 1 footer · final_report.pdf

**中文**  
**问**：某个数字从哪来？  
**答**：分类 → `reports/pv|bess/comparison.md`；鲁棒性 → `reports/robustness_eval.md`；智能体 → `agent_eval/results/last_run_three_ablations_with_judge.json`；集成 → `reports/integration_eval.md`、`integration_eval_meta.json`；C7 → `reports/integration/fault_injection_demo.md`；总述 → **`reports/final_report.pdf`**。

**English**  
**Q**: Where is the authoritative source for a number?  
**A**: Classification—comparison tables; robustness—`robustness_eval.md`; agent—`last_run_three_ablations_with_judge.json`; integration—`integration_eval.md` / meta JSON; C7—`fault_injection_demo.md`; narrative—**`final_report.pdf`**.

---

## 十、快速索引：问题 → 幻灯片

| 主题 | 首选幻灯片 | 备份 / 附录 |
|------|------------|-------------|
| 项目意义 / 团队 | Slide 1–2 | — |
| 云边动机 / HTTP | Slide 3, 7, 21 | Slide 36 |
| 数据规模 / 11 类 | Slide 10–12 | A1, A2 |
| CNN / 三后端 / INT8 | Slide 13–15 | Slide 36 |
| 六轴鲁棒 / OOD | Slide 16–19 | A3, A4 |
| ReAct / RAG / No-RAG | Slide 20–26 | A5 |
| 三模式延迟 / 十节点 | Slide 27–29 | Slide 37 |
| C7 看板 / 五场景 | Slide 30 | `docs/网页演示指南.md` |
| 测试 / 局限 / 复现 | Slide 31–35 | A6 |
| 致谢 | Slide 38 | — |

---

*文档版本：与 `ppt旁白.md`（主汇报 Slide 1–30）及 `ppt制作指南.md` 2026-06-03 产物对齐。*
