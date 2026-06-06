# AgentPV 答辩旁白（Speaker Scripts）

> 从 `[ppt制作指南.md](ppt制作指南.md)` 提取 · **44 页**（正文 38 + 附录 6）· 生成日期 2026-06-06

## 怎么用

- **幻灯片上只有英文**；本文件是口头讲解稿，不要贴到 PPT 里。
- **主汇报止于 Slide 30**（C7 看板）；Slide 31 起及附录 A1–A6 供 Q&A 环节速查，见 [`Q&A.md`](Q&A.md)。
- 每页先 **中文旁白**（答辩主用），再 **English Narration**（双语答辩或练习用）。
- 部分页含 **预设 Q&A** 或 **Q&A 提示**，追问时可扫一眼。
- Slide 30 之后可请组员现场演示 Streamlit 看板，步骤见 [`docs/网页演示指南.md`](docs/网页演示指南.md)。

---

## Slide 1 · Title

### 中文旁白

各位老师好，今天我们汇报的项目叫**AgentPV**，是一套面向光伏和储能电池的云边协同故障诊断系统。我们三个组员分别是夏梓峻、董学语和蔡明仕。

这个项目的核心是解决光伏储能电站的实际运维痛点：一方面电站和储能站有大量传感器数据，边缘设备算力有限但必须在毫秒到百毫秒级给出告警；另一方面运维人员不满足于只知道故障名称，还需要知道该怎么处理、有多紧急、依据的是哪条运维规程。

所以我们做了分层的设计：边缘端做轻量化的故障分类，云端做带知识库的智能推理，最后还有可视化的操作看板。接下来我们会把整个项目的 7 大模块，从数据到系统集成，完整地讲述。

### English Narration

Good morning. Today we present **AgentPV**—a cloud–edge fault diagnosis system for photovoltaic plants and battery energy storage. Our team members are **Zijun Xia**, **Xueyu Dong**, and **Mingshi Cai**.

The core problem is real day-to-day operations at PV and BESS sites. Sites produce a lot of sensor data, but edge devices have limited compute and must still raise alerts within milliseconds to hundreds of milliseconds. Operators also need more than a fault name—they need to know what to do, how urgent it is, and which maintenance procedure applies.

So we use a layered design: lightweight fault classification at the edge, knowledge-grounded reasoning in the cloud, and a visual operator dashboard. Next we walk through all **seven major parts** of the project, from data generation to system integration.

### 预设 Q&A

*What does “AgentPV” mean?* — “Agent” = cloud ReAct LLM tier; “PV” = primary domain, extended to BESS.

---

## Slide 2 · Agenda

### 中文旁白

这是我们今天的汇报路线。

### English Narration

This is our roadmap for today’s presentation.

---

## Slide 3 · Problem — Why cloud–edge co-design?

### 中文旁白

光伏和储能站点都是典型的信息物理系统：每秒都有电压、电流、温度、辐照度或 SOC 等遥测进来。如果故障发现晚，轻则发电损失，重则热失控或设备不可逆损伤。边缘网关通常只有 CPU、内存都很有限，而且现场经常断网——所以第一层必须本地、快速、可靠地给出告警。  
但运维一线反馈是：光告诉「Partial_shading」或「Thermal_anomaly」不够，他们还要知道是先降功率、先隔离、还是先叫人，以及依据哪条 SOP。这就必须有一层云端推理，把结构化告警变成自然语言建议，并且引用知识库 chunk，方便事后审计。  
AgentPV 是用明确的 HTTP 契约连接：`SensorWindow` 进 edge，`Alert` 进 agent，输出 `Recommendation`。这样不管是测试、多节点调度还是故障注入，都用同一套接口。

### English Narration

PV and BESS sites are classic cyber-physical systems: continuous telemetry on voltage, current, temperature, irradiance or state-of-charge. Late detection costs money and can create safety incidents. Edge gateways have tight resource budgets and often lose cloud connectivity—so the first tier must classify locally, quickly, and reliably.  
Operators consistently ask for more than a fault name. They need an imperative action, an urgency level, and a traceable link to the playbook—not a bare softmax label. That motivates a cloud tier that turns structured alerts into recommendations with knowledge-base citations.  
AgentPV ties the tiers together with clear HTTP contracts: `SensorWindow` goes to the edge, `Alert` goes to the agent, and the output is a `Recommendation`. Tests, multi-node scheduling, and fault injection all use the **same APIs**—so what we measure is what we deploy.

---

## Slide 4 · Related work & our differentiation

### 中文旁白

相关工作可以分四块。第一，光伏故障深度学习文献很多停在准确率或混淆矩阵，很少会关心模型能不能部署，比如能不能导出成 ONNX、压缩之后精度掉多少、CPU 上跑有多快、模型体积够不够小，我们这些都在 C3 模块里做了完整的对比。。第二，储能领域常见的是 RUL 或 SOH 估计；在本项目里 RUL 是智能体可调用的工具之一，而不是边缘主分类任务，这样分工更清晰。第三，ReAct 和 RAG 在 NLP 里很常见，但很多 demo 是 isolated prompt；我们是真 edge 服务 in the loop，对 33 个场景跑消融。第四，OOD 检测我们采用 energy score，但不止报 AUROC，还做六轴 stress，并明确写「什么时候策略有效、什么时候无效」。  

### English Narration

Related work falls into four areas. First, many PV fault-detection papers stop at accuracy or a confusion matrix. They rarely ask whether a model can actually be deployed—ONNX export, INT8 compression loss, CPU speed, and model size. We cover all of that in **module C3**. Second, BESS work often focuses on RUL or SOH; here RUL is a **tool the agent can call**, not the main edge classifier—clearer division of roles. Third, ReAct and RAG are common in NLP, but many demos use isolated prompts; we connect a **real edge service in the loop** and run thirty-three scenarios with three ablations. Fourth, for OOD we use energy scores, but we also run **six stress tests** and document when the strategy works and when it does not.

---

## Slide 5 · Six contributions (from abstract)

### 中文旁白

这页是我们整个项目的核心点，我给大家简单过一下： 

第一，我们做了全链路可复现的代码库，从数据生成到最后的故障注入，所有环节都有脚本，还写了 284 个单元测试，保证每个模块都能正常工作，代码也做了规范检查，没有警告。 

第二，我们给光伏和储能各做了一套模型，每个模型都有三种部署版本：原生的 PyTorch、ONNX 的 FP32、还有 INT8 量化版，每个版本的精度、速度、体积我们都做了完整的对比。 

第三，我们额外做了部署场景的鲁棒性测试，模拟了真实环境里的 6 种问题，还加了选择性预测的机制，遇到不确定的情况就交给人工，不会乱输出。 

第四，我们的智能体用了本地的大模型，做了 33 个测试场景，还有三组对比实验，最后用大模型当裁判打分，平均分到了 4.1 分，满足要求。

 第五，系统集成我们做了真实的 HTTP 测试，三种运行模式各测了 50 次，还模拟了 10 个设备同时运行的并发场景。

第六，最后我们做了双语的可视化看板，还有脚本化的故障注入测试，所有场景都跑通了。

### English Narration

These six points summarise the whole project.  
**First**, we built a **reproducible end-to-end codebase**—from data generation to fault injection—with **284 unit tests** and clean linting.  
**Second**, we trained separate models for PV and BESS, each in **three deployment forms**: PyTorch, ONNX FP32, and INT8—with full comparisons of accuracy, speed, and size.  
**Third**, we added **deployment realism tests**: six real-world stress scenarios plus **selective prediction**—when the model is unsure, it defers to a human instead of guessing.  
**Fourth**, the agent uses a **local LLM** on **33 test scenarios** with three ablation runs; an LLM-as-judge score averaged **4.1**, meeting our target.  
**Fifth**, integration was tested over **real HTTP**—fifty runs per mode plus a **ten-node concurrent** scenario.  
**Sixth**, we built a **bilingual dashboard** and **scripted fault-injection tests**—all scenarios passed.

---

## Slide 6 · Assignment compliance C1–C8

### 中文旁白

这是我们七部分组件的内容概览。

### English Narration

This slide is an overview of our **seven component areas** and how each maps to the assignment requirements.

---

## Slide 7 · System architecture (four layers)

### 中文旁白

接下来给大家看我们整个系统的四层架构： 最底层是数据层，我们用仿真生成数据，然后划分好训练验证测试集，固定了随机种子，保证每次生成的数据都一样。 然后是边缘服务层，接收传感器的数据，调用边缘模型做预测，输出标准化的告警 JSON。 再往上是云端的智能体服务，接收边缘的告警，调用大模型和知识库，生成运维建议。 最上层是操作层，有调度器和可视化看板，把所有的事件存下来，还支持操作员手动注入故障来测试。 我们还定了统一的接口和端口，边缘服务跑在 8000 端口，智能体在 8001，大模型在 11434，所有服务都能独立部署，也能一起用 Docker 跑起来。

### English Narration

Here is our **four-layer architecture**.  
At the bottom, the **data layer** uses simulation, fixed train/val/test splits, and a fixed random seed so runs are repeatable.  
The **edge service** takes sensor windows, runs the edge model, and outputs standard **Alert JSON**.  
The **cloud agent service** takes those alerts, calls the LLM and knowledge base, and returns **maintenance recommendations**.  
At the top, the **operations layer** includes a scheduler and dashboard: events are stored, and operators can **inject faults manually** for testing.  
We use fixed ports—edge **8000**, agent **8001**, Ollama **11434**—so each service can run alone or together (including via Docker Compose for local setup).

---

## Slide 8 · Repository map *(已从汇报中移除)*

### 中文旁白

_（本页不再讲解。）_

### English Narration

_This slide has been removed from the live presentation._

---

## Slide 9 · C1 — Why synthetic data?

### 中文旁白

我们做了两个物理仿真器：PV 侧有辐照度、组件温度、DC 电压电流功率等简化物理关系；BESS 侧是 RC 等效电路加 SOC、内阻、循环老化项；`fault_injector` 对干净窗口施加确定性扰动，每种故障一个纯函数，RNG 可 seed。  
固定 seed 42 后，任何人重跑 `generate_dataset` 应得到相同 NPZ 和 split，元数据写在 `data/version.txt`。三类 operating condition（高辐照、低辐照、高温）按 5:3:2 采样，保证每个 fault class 在每种工况下都有样本——这是后续 distribution shift 测试的基础。

### English Narration

We built **two physics-inspired simulators**. On the PV side: irradiance, module temperature, DC voltage, current, and power follow simplified physical relationships. On the BESS side: an RC equivalent circuit with SOC, internal resistance, and ageing terms. `fault_injector` applies **deterministic, seedable** perturbations—one pure function per fault type.  
With **seed 42**, anyone who reruns `generate_dataset` should get the same NPZ files and splits; metadata is in `data/version.txt`. Three operating conditions—high irradiance, low irradiance, high temperature—are sampled **5:3:2**, and every fault class appears in every condition. That foundation supports our later distribution-shift tests.

---

## Slide 10 · C1 — Dataset scale & splits

### 中文旁白

我们最终生成的数据集，总共有 50500 个样本，其中光伏的 28000 个，电池的 22500 个，每个样本是 60 秒、8 通道窗口，采样率 1 Hz，与边缘推理输入一致。划分不是简单 random row split，而是按 `system_id` 分层：同一个模拟资产的所有窗口只出现在 train、val 或 test 之一，避免「同一条串」泄漏到测试集导致 F1 虚高。比例大约 70/15/15。

### English Narration

Our final dataset has **50,500 samples**—**28,000 PV** and **22,500 BESS**. Each sample is a **60-second, 8-channel** window at **1 Hz**, matching edge inference input.  
Splits are **stratified by `system_id`**: all windows from the same simulated asset stay in train, val, or test only—avoiding leakage that would inflate F1. The split is roughly **70 / 15 / 15**.

---

## Slide 11 · C1 — Fault taxonomy (11 classes)

### 中文旁白

故障的分类中光伏做了 7 类：正常、局部阴影、积灰、旁路二极管故障、组串断开、逆变器故障、组件老化； 电池做了 5 类：正常、容量衰减、内阻上升、热异常、电芯不均衡，加起来一共 11 类故障。 边缘模型输出这些分类之后，我们还会根据置信度把告警分成三个等级：monitor、warning、critical。如果只是 monitor 级别的，就不用调用云端的智能体了，只有 warning 和 critical 的，才会传给云端，让它出详细的运维建议，节省云端的资源。  

### English Narration

We define **eleven fault classes** in total. **PV—seven**: Normal, partial shading, soiling, bypass diode fault, string disconnection, inverter fault, and degradation. **BESS—five**: Normal, capacity fade, internal resistance increase, thermal anomaly, and cell imbalance.  
After classification, we map confidence to three **severity levels**: monitor, warning, and critical. **Monitor** alerts stay at the edge only—they **do not** call the cloud agent. Only **warning** and **critical** go to the cloud for detailed recommendations, saving cloud resources.

### 预设 Q&A

答辩时若被问「为什么 eleven classes」，答：作业要求 PV≥7、BESS≥5，我们按系统拆开训练两个 CNN head，但运维视图统一成 Alert JSON。

*Are PV and BESS labels merged in one model?* — No; separate weights and heads, shared architecture only.

---

## Slide 12 · C1 — Operating conditions & feature channels

### 中文旁白

我们还做了三种不同的运行工况，样本的权重是 5:3:2，覆盖不同的场景。 每个样本的特征也都是标准化的，光伏的有电压、电流、功率、温度、辐照度这些 8 个特征，电池的有端电压、电流、SOC、温度这些 8 个特征，八个通道的顺序写死在 Pydantic `SensorWindow` 和训练 NPZ 里，训练和 ONNX 推理必须同一顺序；鲁棒性实验里的 missing-feature mask 也是按通道索引施加的。PV 侧偏电气与热环境，BESS 侧强调内阻估计、电压散布和循环老化代理量。

### English Narration

We use **three operating conditions** with sample weights **5:3:2** to cover different scenarios.  
Each sample has **eight standardised channels**. PV includes voltage, current, power, temperature, irradiance, and related signals; BESS includes terminal voltage, current, SOC, temperature, and related signals. Channel order is fixed in Pydantic `SensorWindow` and training NPZ—training and ONNX inference must use the **same order**; missing-feature tests in robustness also mask by channel index. PV channels emphasise electrical and thermal environment; BESS channels emphasise resistance estimates, voltage spread, and ageing proxies.

---

## Slide 13 · C2 — CNN-1D edge architecture

### 中文旁白

边缘模型是刻意小的 1D-CNN：两层卷积加 BN、ReLU，全局平均池化后接全连接，每系统约四万八千参数，非常轻量。PV 与 BESS **共享结构、不共享权重**，避免跨系统 shortcut。训练目标用 validation macro-F1 做 early stop，比裸 accuracy 更适合轻度不平衡。  
导出链把标准化 (μ,σ) 烘焙进 ONNX 图前端，保证 edge 服务只做一次 forward训练完之后，然后再做 INT8 的静态量化，用 1024 个样本做校准。

Q&A：

老师若问「为何不用 Transformer/LSTM」，答：作业强调部署预算；我们实测 ONNX FP32 P95 约 0.15 ms，满足亚百毫秒约束，且可解释、可量化。

### English Narration

The edge model is a **small 1D-CNN**: two conv layers with batch norm and ReLU, global average pooling, then a classifier—about **48,000 parameters**, very lightweight. PV and BESS **share the same structure but not the same weights**. We early-stop on validation **macro-F1**, which handles mild class imbalance better than raw accuracy.  
After training we export to ONNX with normalisation baked into the graph, then apply **static INT8 quantisation** calibrated on 1,024 samples.  
If asked why not Transformer or LSTM: the assignment stresses **deployment budgets**; our ONNX FP32 P95 is about **0.15 ms**, meeting the sub-100 ms edge target with a model that is easy to explain and quantise.

---

## Slide 14 · C2/C3 — Three-backend comparison (headline numbers)

### 中文旁白

这张表是 C3 交付的核心数字墙。PV 三条线 macro-F1 都是 0.9994，INT8 甚至无损，体积从 0.184 MiB 压到 0.058 MiB。BESS FP32 同样 0.9980，但 INT8 掉到 0.7058——不是训练失败，而是量化与窄特征带冲突，下一页用混淆矩阵证明。延迟上 ONNX FP32 相对 PyTorch 约六倍加速，INT8 再减半；BESS 默认 FP32 ONNX，仍只有 0.175 MiB。

### English Narration

This table is the **core number wall for C3**. PV macro-F1 stays **0.9994** across all three backends; INT8 is effectively lossless, and size drops from **0.184 MiB to 0.058 MiB**. BESS FP32 is **0.9980**, but INT8 falls to **0.7058**—not a training failure, but quantisation hitting narrow feature bands; the next slide shows this in the confusion matrix.  
ONNX FP32 is roughly **six times faster** than PyTorch on CPU; INT8 roughly halves latency again. BESS production defaults to **FP32 ONNX** at only **0.175 MiB**.

### 预设 Q&A

*Why report BESS INT8 if it fails?* — Assignment requires trade-off analysis; honest failure boundaries strengthen the defence.

---

## Slide 15 · C3 — Confusion matrices (FP32 clean + INT8 failure)

### 中文旁白

这是混淆矩阵的结果，左两图说明 FP32 部署形态下 PV、BESS 对角线干净，满足「macro-F1 ≥0.99」叙事。第三张是 BESS INT8 的 canonical failure：Normal、热异常、内阻上升三类互混，三类在估计内阻和电压散布上数值带很窄，per-tensor MinMax 把尺度压扁，INT8 的量化把这些不同类的特征给压缩到一起了，导致模型分不出来了，最后宏 F1 掉了快 30 个百分点。   

Q&A  
若老师问「INT8 是否完全不可用」，答：PV 可用；BESS 需换量化策略前不建议上线 INT8，不过我们也记录了修复的方法，比如用按通道的量化，或者熵校准，以后可以优化

### English Narration

These confusion matrices tell the story. The first two show **clean FP32 diagonals** for PV and BESS—supporting macro-F1 at or above **0.99**. The third is the **BESS INT8 failure case**: Normal, thermal anomaly, and internal-resistance increase get mixed up. Those classes have **very narrow numeric bands** in estimated resistance and voltage spread; **per-tensor MinMax** flattens the scale, INT8 quantisation merges features across classes, and macro-F1 drops by nearly **30 percentage points**.  
If asked whether INT8 is completely unusable: **PV INT8 works**; for **BESS**, we would not deploy INT8 until we change the quantisation strategy—for example per-channel quantisation or entropy calibration—as we document in the report.

---

## Slide 16 · Extended evaluation — deployment realism motivation

### 中文旁白

不过，干净数据下的好结果还不够，真实部署的时候，会遇到各种各样的问题，所以我们额外做了一系列的鲁棒性测试。 我们模拟了 6 种真实场景的压力： 第一，不同运行工况的分布偏移； 第二，传感器丢数据，最多丢一半的特征； 第三，传感器的噪声； 第四，传感器的校准漂移，最多飘 20%； 第五，对抗性的 FGSM 攻击； 第六，跨系统的异常数据，比如把电池的数据传给光伏的模型。 然后我们加了一个能量 - based 的选择性预测机制，简单说就是，模型遇到不确定的输入，就会告诉我们 “这个我拿不准”，然后交给人工处理，不会乱输出结果，这个阈值我们校准到了 95% 的覆盖率，保证大部分正常的情况都能自动处理。

Q&A

分布内切片用 meta CSV 的 operating_condition；缺通道、噪声、漂移、FGSM、跨系统 OOD 分别对应不同 failure mode，后面一页会写「何时成功、何时失败」。选择性预测是 training-free 的 energy score，在 validation 上校准到 95% coverage，低于阈值就拒绝给出自信但错误的类。

### English Narration

Good results on clean data are not enough—**real deployment** faces many issues, so we added a full **robustness suite**. We simulate **six stress scenarios**: (1) distribution shift across operating conditions; (2) missing sensors—up to half the channels; (3) sensor noise; (4) calibration drift—up to 20%; (5) FGSM adversarial perturbations; (6) cross-system OOD—e.g. feeding BESS data to the PV model.  
We also added **energy-based selective prediction**: when the model is uncertain, it says “I am not sure” and defers to a human instead of guessing. The threshold is calibrated to **95% coverage** on validation so most normal cases still run automatically.

---

## Slide 17 · Robustness — selective prediction headline numbers

### 中文旁白

这是选择性预测的结果，分布内：PV/BESS 在 95% coverage 下 selective accuracy 接近 1，risk-coverage 曲线贴近左上角，说明「拒绝最不确定的 5%」策略有效。  
跨系统 OOD 时 energy 分数方向 inverted——高能量反而更像 in-distribution，但 AUROC 仍高于随机（PV 0.6037，BESS 1.0），所以部署不能盲用默认阈值，需要 flip 或 Mahalanobis 补充。

### English Narration

Here are the **selective prediction** results. In-distribution, at **95% coverage**, selective accuracy for PV and BESS is close to **1**; the risk–coverage curve sits near the top-left corner—rejecting the most uncertain 5% works well.  
Under **cross-system OOD**, the energy score direction is **inverted**—high energy looks more in-distribution—but AUROC is still above random (PV **0.6037**, BESS **1.0**). So we cannot use the default threshold blindly; we need a flip rule or a Mahalanobis fallback at deploy time.

---

## Slide 18 · Robustness — when it succeeds vs fails

### 中文旁白

成功边界：分布内 selective prediction 几乎完美；轻噪声和轻漂移下 macro-F1 仍高于 0.9，说明仿真器生成的干净信号在小幅扰动下仍可分。失败边界同样重要——缺 10% 通道时准确率掉四十个百分点，而 energy confidence 反而上升，说明 rejection policy **挡不住** 坏输入，必须上游做 NaN/缺测检查。  
跨系统 OOD 与 INT8 脆弱性形成呼应：BESS 对 FGSM 小 ε 更敏感，单靠事后拒绝不够，需要对抗训练或更强特征正则。

Q&A

这里给大家总结一下我们的鲁棒性测试的结果： 哪些情况能搞定？ 正常的输入没问题，轻微的噪声和漂移，比如噪声不超过 10%、漂移在 5% 以内，模型的 F1 都能保持在 0.9 以上，没问题。 哪些情况搞不定？ 如果丢了 10% 的传感器数据，准确率会掉 40%，而且模型自己还不知道，信心还很高，这个我们后面要加传感器的健康检查； 然后跨系统的异常，刚才说的能量方向反了，调整一下就好； 如果漂移太大，超过 20%，或者对抗攻击的强度太高，模型就扛不住了，这个以后可以加对抗训练来优化。

### English Narration

Let me summarise what worked and what did not.  
**What works:** normal inputs are fine; with **mild noise and drift**—noise up to about 10%, drift up to about 5%—macro-F1 stays **above 0.9**.  
**What fails:** if **10% of channels are missing**, accuracy drops about **40 points**, and the model stays overconfident—we need upstream sensor health checks. **Cross-system OOD** needs a threshold flip, as noted. With **drift beyond 20%** or **strong adversarial attacks**, the model breaks; adversarial training would be a future improvement.

---

## Slide 19 · Robustness — representative figures (OOD & conditions)

### 中文旁白

这些是我们鲁棒性测试的一些图表，上图是光伏的异常检测的能量直方图，能看到正常和异常的分布，虽然方向反了，但是还是能分开的；下图是不同工况下的 F1 热力图，能看到三种工况下，模型的效果都很好，分布偏移的影响不大。 

### English Narration

These are sample **robustness plots**. The top figure is the PV **OOD energy histogram**—normal and abnormal distributions overlap, and the direction is inverted, but classes are still separable. The bottom figure is the **macro-F1 heatmap across operating conditions**—F1 stays high in all three regimes, so distribution shift has limited impact here.

---

## Slide 20 · C4 — ReAct cloud agent architecture

### 中文旁白

C4 云端不是裸 ChatGPT，而是带工具调用的 ReAct：`observe` 读 Alert，`reason` 规划，`act` 调工具，`reflect` 校验，`report` 输出 Recommendation。四个工具中 `retrieve_knowledge` 是 grounding 核心，连接 30 篇 playbook 与 Chroma 向量库。  
LLM 用本机 Ollama llama3.2，计划阶段若 JSON 解析失败，会走 `ollama_plan_fallback_mock` 确定性计划，保证工具仍执行、最终仍有 recommendation——这在 agent_eval 和 C7 HTTP 日志里都能观察到。  
Citation 带 chunk_id，是为了运维审计，也是 No-RAG 消融时唯一能证明「依据规程」的指标。

### English Narration

Module C4 is not a bare chatbot—it is a **ReAct agent with tools**: observe the alert, reason, act on tools, reflect, and report a recommendation. Among four tools, **`retrieve_knowledge`** is the grounding core—it connects **30 playbooks** to a Chroma vector store.  
We use **Ollama llama3.2** locally. If planner JSON parsing fails, **`ollama_plan_fallback_mock`** supplies a deterministic plan so tools still run and a recommendation is still returned—as seen in agent_eval and C7 logs.  
Citations include **chunk_id** for operator audit—and for measuring grounding loss in the No-RAG ablation.

---

## Slide 21 · C4 — Alert → Recommendation data flow

### 中文旁白

整个数据的流程是这样的： 首先边缘服务接收传感器的数据，输出告警，里面有故障类型、等级、置信度这些信息； 如果告警的等级是 warning 或者 critical，就会传给云端的智能体服务； 智能体处理之后，输出运维建议，包括要做什么、紧急程度、置信度、要不要升级上报，还有引用了哪些知识库的内容； 最后这些事件都会存下来，同步到可视化看板上。 如果只是 monitor 级别的告警，就不用调用智能体了，边缘自己处理就好，节省资源。  


Q&A

数据流是答辩的云边故事线：结构化 `SensorWindow` 进 edge，出 `Alert`；只有 warning/critical 才进 agent 出 `Recommendation`。monitor 类（如 PV_Normal）故意不调用 agent，节省约 8.5 s 的 LLM 时间，也符合运维「正常不需 LLM」的预期。  
JSONL 事件被 orchestrator 和 dashboard 共享，保证被动监控与主动 inject 看同一份时间线。右图强调 full 模式延迟几乎全是 agent，不是 edge 慢——部署优化应优先 LLM，而不是砍掉 CNN。  
`knowledge_sources[]` 是审计字段，No-RAG 消融会把它清零，这是后面最重要的发现。

### English Narration

Here is the **end-to-end data flow**. The edge service receives sensor data and outputs an **Alert**—fault type, severity, confidence, and related fields. If severity is **warning** or **critical**, the alert goes to the cloud agent. The agent returns a **Recommendation**: what to do, urgency, confidence, whether to escalate, and which knowledge-base chunks were cited. All events are stored and synced to the **dashboard**.  
**Monitor** alerts never call the agent—the edge handles them alone, saving resources.

---

## Slide 22 · C5 — Agent benchmark design

### 中文旁白

接下来是智能体的评测，我们自己做了 33 个测试用例，其中 23 个是明确的故障场景，还有 10 个是故意做的模糊的疑难场景，覆盖所有的故障类型。 每个用例我们都定了标准的结果，比如应该是什么紧急等级，必须包含哪些关键词，不能有哪些违规的内容，最少要引用几个知识库的文档。 然后我们跑这些用例，用两种方式打分：一种是我们自己定的规则打分，另一种是用大模型当裁判，按 1 到 5 分打分。 而且我们做了三组对比实验：完整的系统、关掉 RAG 的、关掉 ReAct 推理步骤的，来验证每个模块的作用。

Q&A

Benchmark 不是手写几个 prompt，而是 33 个带 oracle 的 JSON 场景，其中 10 个故意模糊，逼 agent 在不确定时仍要给 urgency 和合规措辞。Runner 每场景走真 edge 分类再调 agent，保证「in the loop」而不是离线编造 fault_class。  
双通道评分：启发式 rubric 可复现、可自动化；LLM-as-judge 提供人类可读质量，本轮 99 条全有分，均值 4.10。三消融对应 full、关 RAG、关 trace，共 99 条记录，源文件 `last_run_three_ablations_with_judge.json`。  
设计意图是让 provenance（KB 数）与 surface score 可分离，为 No-RAG 幻灯片埋伏笔。

### English Narration

Next is **agent evaluation**. We built **33 test cases**—**23 clear-cut** fault scenarios and **10 deliberately ambiguous** hard cases—covering all fault types. Each case has an oracle: expected urgency, required keywords, forbidden phrases, and minimum knowledge-base citations.  
We score runs two ways: a **rule-based rubric** and an **LLM-as-judge** on a 1–5 scale. We also run **three ablations**: full system, RAG disabled, and ReAct trace disabled—to show what each module contributes.

---

## Slide 23 · C5 — Ablation results summary

### 中文旁白

Full 配置 heuristic mean 0.9318，LLM judge 4.10；urgency、forbidden、knowledge 三个硬槽位都是 100%，说明安全相关维度没崩。keywords 只有 72.7% perfect，是 lexical drift，不是 urgency 错。  
No-RAG 行最关键：mean 只降到 0.9242，但 KB/scenario 从 2.61→0，% with KB 从 91%→0%——**表面分几乎不变，grounding 全灭**。No-Trace mean 0.9015，但 KB 行为与 full 相近，说明带宽紧时可以丢 trace 不丢 RAG。  


### English Narration

Under the **full** config, heuristic mean is **0.9318** and LLM judge mean is **4.10**. Hard slots—urgency, forbidden phrases, knowledge—are all **100%**, so safety-related dimensions hold. Keywords are only **72.7%** perfect—that is **wording drift**, not wrong urgency.  
The **No-RAG** row is key: mean drops only slightly to **0.9242**, but KB per scenario goes from **2.61 to 0** and **% with KB from 91% to 0%**—the surface score barely moves, but **grounding is gone**. **No-Trace** mean is **0.9015**, but KB behaviour stays close to full—under bandwidth pressure you can drop trace before you drop RAG.

---

## Slide 24 · C5 — No-RAG: grounding collapse (headline finding)

### 中文旁白

这里我们重点说一下关掉 RAG 的对比实验的结果，这个很有意思。 我们把 RAG 关掉之后，发现规则打分的结果几乎没差，从 0.93 降到 0.92，看起来好像没影响？ 但是！原来完整的系统里，每个回答平均会引用 2.61 个知识库的文档，91% 的回答都有引用；关掉 RAG 之后，引用数直接变成 0 了，所有的回答都没有依据了。 这说明什么？说明关掉 RAG 之后，智能体说的话听起来还是很合理，表面的质量没差，但是它完全没有引用我们的运维规程，运维人员根本不知道这个建议是哪来的，没法审计，也不知道符不符合我们的内部规定。 这也告诉我们，RAG 的核心价值不是提高回答的表面质量，而是给回答加可追溯性，这对工业场景来说太重要了。 那关掉推理步骤的话，结果会掉一点，但是知识库的引用还在，所以如果带宽不够的话，其实可以把推理步骤关掉，节省点时间，不影响核心的功能。

Q&A

这是智能体章节最重要的单页。关掉 RAG 后，LLM 仍能生成读起来合理的建议，启发式总分几乎不掉，但 `% with KB` 归零，运维无法追溯规程依据——在真实电站这等于不可接受。  
因此我们主张：agent 评测必须同时报 provenance 指标（KB/scenario、% with KB），不能只看 composite mean。右图 kb_sources_per_ablation 是答辩必放图，一眼显示 No-RAG 柱为零。  
No-Trace 对照说明 trace 影响可解释性审计表，但不影响 grounding；与 full 比 mean 略低主要伤在 keywords 槽位。

### English Narration

This slide focuses on the **No-RAG ablation**—and it is striking. With RAG off, the rule score barely changes—from **0.93 to 0.92**—so it looks harmless. But in the full system, each answer cites about **2.61 knowledge-base documents on average**; **91%** of answers have citations. With RAG off, citations drop to **zero**—every answer is unsupported.  
So the agent still **sounds reasonable**, but operators **cannot audit** which procedure it followed or whether it matches internal rules. **RAG’s main value is traceability**, not polishing the surface score—critical in industrial settings. Turning off the reasoning trace lowers scores a bit, but **citations remain**—if bandwidth is tight, you can drop trace first without losing grounding.

### 预设 Q&A

*Why not 100% keywords?* — Real LLM wording drifts from oracle phrases; urgency/forbidden/knowledge remain 100%.

---

## Slide 25 · C5 — Score distribution & ablation delta

### 中文旁白

这是打分的分布，大家可以看到，三个版本的打分大部分都在 0.9 以上，红色的部分是去掉某个模块之后，质量掉的部分，能看到不同的场景下，两个模块的影响不一样，有的场景 RAG 影响大，有的推理步骤影响大。 那些低于 1 分的，大部分都是关键词没匹配上，不是紧急等级或者违规的问题，影响不大。

Q&A

直方图回答「整体有没有过 0.9 线」：三消融分布重叠较多，说明 LLM 质量整体合格，但仍有长尾低于 1.0。delta 图按场景展示 full 减去消融的差，红色表示去掉 RAG 或 trace 后变差——可指出哪些 scenario 对 grounding 最敏感。  
32 条 sub-1.0 记录几乎全败在 keywords，不是 urgency 或 forbidden，这对安全叙事是好事：我们没有在「该不该立即停机」上翻车。答辩时可挑一个 ambiguous scenario 口述，细节放附录 A5。

### English Narration

This is the **score distribution**. All three ablation runs mostly sit **above 0.9**; red areas show where quality drops when a module is removed—impact varies by scenario: sometimes RAG matters more, sometimes the reasoning trace.  
Scores **below 1.0** are mostly **keyword mismatches**, not urgency or forbidden-phrase failures—limited impact on safety.

---

## Slide 26 · C5 — Tool calls, ReAct depth & failure taxonomy

### 中文旁白

我们还统计了工具调用的情况，完整的系统里，每个场景平均会调用 1.82 次工具，走 5.82 步 ReAct 的流程；关掉 RAG 之后，工具调用就少了，因为不用检索知识库了；关掉推理步骤的话，就直接输出结果，一步就完了。 那些没拿到满分的情况，大部分都是关键词没匹配上，很少有紧急等级或者引用的问题，而且就算大模型输出的 JSON 格式不对，我们也做了降级的处理，整个服务还是能正常返回，不会崩掉。

Q&A

工具图证明 full 配置确实在调用 RAG 等工具，不是空转 ReAct。No-RAG 工具数下降符合预期；No-Trace 把步数压到 1，只剩最终 recommendation，适合解释「带宽受限时的降级选项」。  
failure_taxonomy 把 sub-1.0 的丢分槽位可视化，答辩强调「软失败在措辞，硬失败在安全槽位很少」。Planner fallback 说明 Ollama JSON 不稳定时系统仍 200 OK，与 C6/C7 graceful degradation 叙事一致。  
若老师问 ReAct 是否 over-engineered，答：步数和工具数是可观测的，且 No-Trace 消融量化了 trace 价值。

### English Narration

We also tracked **tool usage**. In the full system, each scenario averages **1.82 tool calls** and **5.82 ReAct steps**. With RAG off, tool calls drop because retrieval is disabled. With trace off, the agent outputs in **one step**.  
Most sub-perfect scores are **keyword mismatches**—rarely urgency or citation failures. Even when the LLM returns bad JSON, we **degrade gracefully**; the service still returns a response and does not crash.

---

## Slide 27 · C6 — Three-mode integration latency

### 中文旁白

接下来是系统集成的部分，我们做了真实的 HTTP 时延测试，三个模式各跑了 50 次，去掉热身的 3 次。 大家可以看，完整的系统，95% 的请求都能在 9.8 秒内完成，刚好满足课程要求的 10 秒的上限。 我们拆分了一下时间，边缘的预测只花了 26 毫秒，几乎可以忽略，大部分的时间都花在云端的大模型推理上，这个是大模型本身的速度限制。

Q&A

C6 用真 HTTP 而非 mock：三种 integration mode 各跑 50 次，warmup 丢弃 3 次后统计 P50/P95/P99。edge_only P95 5.69 ms，是 graceful degradation 的地板；full P95 9803 ms，cloud_only 9941 ms，都低于 10 s 作业预算。  
分解图是关键证据：full 模式里 edge 只占不到 1%，agent/LLM 占 99% 以上——优化 E2E 应换更快本地模型、缓存或 speculative decode，而不是去掉 CNN。数字与 `integration_eval_meta.json` 一致，2026-06-03 生成。  
三张 latency 图（bars、violin、split）建议都放进 deck 或附录，答辩时至少讲 bars + split。

### English Narration

Next is **system integration**. We ran **real HTTP latency tests**—**50 runs per mode**, discarding **3 warm-up** runs. In **full** mode, **95% of requests finish within 9.8 seconds**—just under the **10-second** course limit.  
When we split the time, **edge prediction is about 26 ms**—negligible. Almost all delay is **cloud LLM inference**—a limit of the model speed itself.

---

## Slide 28 · C6 — Mode interpretation & deployment guidance

### 中文旁白

我们来解释一下这三个模式： 第一个是 edge_only，也就是只有边缘服务，这个时候 P95 的时延只有 5.69 毫秒，非常快。这个是我们的兜底方案，就算云端的服务挂了，断网了，边缘还是能正常出告警，不会影响基本的功能，这就是优雅降级。 第二个是 cloud_only，也就是只有云端的服务，它的时延和完整的系统差不多，但是它没有边缘的故障分类的结果，相当于少了一层 grounding，所以我们的部署建议是，边缘的服务一定要一直开着，不能只靠云端。 第三个就是完整的云边协同的模式，就是我们最终的系统，时延刚好卡在 10 秒以内，刚好满足要求，以后我们可以换更快的大模型，或者加缓存，把这个时间再压一压。

Q&A

三种模式的**比较意义**比单点延迟更重要。edge_only 定义了「LLM 挂了还能运维什么」——仍有 fault_class 和 severity。cloud_only 与 full 几乎一样慢，却失去边缘 ML 标签对 agent tool 选择的 grounding，因此部署建议永远是 edge always-on。  
full P99 碰到 10.0 s，说明在并发或 tail 场景下仍可能触预算上限；编排器默认 10 s HTTP timeout 会在 fan-out 下产生 `agent_recommend_failed`，下一页用 144 events / 3 failures 证明系统不崩。  
timeout 调到 120 s 是复现建议，不是掩盖问题，而是让 demo 完整跑完同时保留失败计数作为证据。

### English Narration

Here is what the **three modes** mean.  
**edge_only**—edge service only. P95 is **5.69 ms**, very fast. This is our **fallback**: if the cloud is down or offline, the edge still raises typed alerts—**graceful degradation**.  
**cloud_only**—cloud only. Latency is similar to full mode, but you **lose the edge fault label**—one layer of grounding is missing. We recommend **keeping the edge always on**, not cloud-only.  
**full**—cloud–edge together. End-to-end latency sits **just under 10 seconds**. Later we can use a faster LLM or caching to reduce it further.

---

## Slide 29 · C6 — 10-node orchestrator & graceful degradation

### 中文旁白

然后我们还做了 10 个节点的并发测试，模拟 6 个光伏设备和 4 个电池设备同时运行，跑 60 秒。 最后 60 秒里我们处理了 144 个事件，中间有 3 次云端的智能体服务因为并发太高超时了，但是我们的调度器完全没受影响，每个节点的任务都是独立的，边缘还是正常出告警，整个系统没有崩，还是能继续跑，这就是我们说的优雅降级，保证系统的可靠性。

Q&A

`pv6_bess4` 目录十节点、跑 60 秒，共 144 条 JSONL 事件，其中 3 次 agent 推荐因 httpx 超时失败——并发 tail latency 超过编排器 HTTP 上限时的真实行为，不是单元测试 mock。关键是 orchestrator **没有崩溃**，各节点 task 继续 tick，部分节点仍收到 Recommendation。  
fanout 图展示每节点 events/alerts/recommendations/errors；severity mix 展示 monitor/warning/critical 比例，与 AGENT_TRIGGER 规则一致。这页直接支撑 Deliverable #7「empirical graceful degradation」。  
Q&A：为何 120 s timeout 仍有 3 次失败？——并发下 agent 队列仍可能触顶；证据是 failure count + 系统存活，而非零失败吹牛。

### English Narration

We also ran a **ten-node concurrent test**—six PV and four BESS assets—for **60 seconds**. We processed **144 events**. The cloud agent **timed out three times** under concurrency, but the **scheduler kept running**—each node’s task is independent, the edge still raised alerts, and the **system did not crash**. That is **graceful degradation** and **reliability under load**.

### 预设 Q&A

*Is three failures acceptable?* — Yes as documented tail behaviour; the system degrades without crashing.

---

## Slide 30 · C7 — Operator UI & scripted fault injection

### 中文旁白

最后是第七个模块，可视化的操作看板。 我们做了一个 Streamlit 的双语看板，请我们的成员演示给大家看。

Q&A

C7 要求「可交互的原型」，我们交付两层证据。第一层是 `dashboard/app.py`：双语操作员界面，四个 Tab 分别看节点概览、时间线、单条事件详情和全局统计；数据来自 orchestrator 写的 JSONL，注入走 `dashboard/inject.py`，与 C6 编排器 `NodeRunner` 完全同路径——不是 mock 分支。  
第二层是脚本化复现：`python scripts/demo_fault_injection.py` 生成 `fault_injection_demo.md`，五场景全部 ✅。场景 1–3 走全链路，edge 个位数到二十多 ms，agent 约 8.5–8.7 s，各有 3 条 KB 引用。场景 4 证明 monitor 不触发 agent（与 `AGENT_TRIGGER_SEVERITIES` 一致）。场景 5 用 `skip_agent=True` 演示 LLM 不可用时仍有 critical Alert——graceful degradation。  
场景 3 最能说明云边价值：边缘给出 Thermal_anomaly + critical；云端返回 immediate urgency，并引用 `kb_thermal_anomaly_bess.md` 的 containment 步骤，而不是重复标签。Recommendation 全文在 demo 报告里可查，knowledge_sources 可追溯到向量库 chunk。  
**本 PPT 不做浏览器逐步演示**；若老师要看界面，说明另场按 `docs/网页演示指南.md` 进行。答辩口述此表即可证明 C7 可审计、可重复。

### English Narration

Finally, **module seven**—the **visual operator dashboard**. We built a **bilingual Streamlit dashboard**; our teammate will **walk you through it live** now.

### 预设 Q&A

*Is the UI the real system?* — Yes: `inject.py` shares the node-runner HTTP path; eleven unit tests in `test_dashboard_inject.py`.

---

## Slide 31 · Engineering quality & tests *(Q&A 附录)*

### 中文旁白

工程质量与模型分数同等重要：284 条单元测试覆盖 simulation 到 dashboard inject，其中 inject 路径用 MockTransport 测 11 例，保证 C7 注入逻辑在答辩前已被验证。ruff 零 warning 在终稿声明，体现可维护性。  
集成评测、智能体 benchmark 和 C7 脚本都需要本机起 edge/agent 与 Ollama——复现步骤见 `复现指南.md` §4 与终稿 §11.3。RAG 30 文档需 `rag.ingest` 后才可检索。  
老师问「有没有 CI」：可指 pytest 全绿 + 报告 JSON 时间戳；GitHub Actions 非本作业硬性要求。

### English Narration

Two hundred eighty-four unit tests span simulation through dashboard inject—including eleven mocked inject tests so C7 logic is verified before defence. Zero ruff warnings are claimed in the final report.  
Integration benches, agent evaluation, and C7 scripts require local edge and agent services plus Ollama—documented in `复现指南.md` section four and final report section eleven point three. Thirty playbooks require `rag.ingest` before retrieval works.  
Point to pytest green and timestamped JSON artefacts if asked about automation.

---

## Slide 32 · Discussion — what worked

### 中文旁白

讨论章先讲 worked：双系统 FP32 高精度与亚毫秒边缘延迟；INT8 失败被诚实记录反而增强可信度；编排器与 mock plan fallback 展示 graceful degradation；真 LLM 下 rubric 与 judge 双通道都过线；inject 与集成同路径；鲁棒性六轴回应导师 deployment realism。  
这页为下一页局限做铺垫，体现「我们知道哪里好、哪里不好」。不要只报喜——评委更信 balanced narrative。  
可口头补一句：19+ 报告图、6 份子报告、24 页 PDF 都是同一晚 refresh 产物，时间戳 2026-06-03。

### English Narration

What worked: dual-system FP32 accuracy and sub-millisecond ONNX edge latency; honest BESS INT8 failure strengthens credibility; orchestrator and mock-plan fallback show graceful degradation; real LLM passes rubric and judge with safety slots at one hundred percent; shared HTTP inject path; six-axis robustness beyond baseline F1.  
This slide sets up limitations—balanced narrative beats cheerleading. Mention nineteen plus figures and six sub-reports refreshed 2026-06-03.

---

## Slide 33 · Discussion — limitations & future work

### 中文旁白

局限与终稿 §9 一一对应：BESS INT8、OOD 方向、缺通道、E2E 贴近 10 s、planner JSON、Dashboard 需手动刷新 JSONL、训练数据为合成。每条局限都绑了优先 future work，说明不是「不会做」而是 scope 内诚实交付。  
答辩时主动讲 limitation 通常比等提问得分更高，尤其 INT8 和 No-RAG 已在正文展开，这里做收束。未来工作避免空泛，全部可追溯到 `开发记录.md` 或 final report §9.3。  
若老师问「若重做会改什么」，答：先 BESS 量化策略 + 7B LLM，再 adversarial training。

### English Narration

Limitations mirror final report section nine—INT8, inverted OOD, missing channels, tail latency, planner JSON, manual JSONL refresh, synthetic-only data—each paired with prioritised future work, not hand-waving. Proactive limitation discussion scores well after INT8 and No-RAG already in the body.  
If asked what to change first: BESS quantisation policy and a faster local LLM, then adversarial training.

---

## Slide 34 · Conclusion

### 中文旁白

结论句：AgentPV 不是单一高 F1 分类器，而是可运维、可降级、可审计的云边系统。数值上满足作业四条预算；方法上回应 deployment realism——鲁棒六轴、选择性预测、No-RAG provenance、编排器失败吸收。  
284 测试、6 份子报告、终稿 PDF 均可按 §11.3 一键复现。保持语调自信但克制，数字已在前文证明，这里做 synthesis 即可。追问时可打开 `final_report.pdf` 任意章节 supplement。

### English Narration

AgentPV is not just a high-F1 classifier—it is an operable, degradable, auditable cloud–edge stack meeting numeric budgets and deployment-realism questions: six stress axes, selective prediction, No-RAG provenance collapse, and orchestrator failure absorption.  
Two hundred eighty-four tests, six reports, and a twenty-four-page PDF reproduce from section eleven point three. Close confidently but modestly—numbers already earned; this slide synthesises. Offer the PDF for follow-up detail.

---

## Slide 35 · Reproducibility & artefact index

### 中文旁白

复现不需要猜命令：`复现指南.md` §4 与终稿 §11.3 提供 PowerShell 复制块，含 judge 环境变量与 `--http-timeout 120`。表中路径是答辩追问时的「导航页」——数据看 version.txt，模型看 artifacts，智能体看 with_judge JSON，集成看 integration_eval.md，C7 看 fault_injection_demo.md。  
强调：文档对齐的答辩不必重跑全链路；只有改代码、权重或 benchmark 才需重跑对应步骤。PDF 可用 `render_final_report.py` 再生。  
若时间紧，附录 A6 放完整命令块，本页只讲索引逻辑。

### English Narration

Reproduction is copy-paste from `复现指南.md` section four and final report section eleven point three—including judge env vars and orchestrator timeout one hundred twenty. The table navigates artefacts for follow-up questions.  
Aligned documentation defence need not rerun the full pipeline—only changed code or weights require partial reruns. Regenerate PDF via `render_final_report.py`. Full commands live in appendix A6.

---

## Slide 36 · Q&A backup (1/2) — data & models

### 中文旁白

备份页不念，供追问扫读。数据合法性：合成 + seed 42 + version.txt。50500 构成与 split 35126/7768/7606 指 data card。INT8 答窄带 + MinMax + 生产 FP32。延迟答 evaluation 脚本千次推理。指标答 macro-F1 与 per-class report 并用。  
若被问「合成是否过于简单」，答：六轴 stress 正是为了检验离开训练分布时的行为；且 full pipeline 在真 HTTP 下已验证。  
保持回答短句，引用路径即可。

### English Narration

Backup—do not read verbatim. Synthetic seed-forty-two data with `version.txt`; fifty thousand five hundred composition in the data card; BESS INT8 narrow bands and FP32 production default; edge latency via thousand-iteration CPU bench; macro-F1 plus per-class reports.  
If synthetic seems too easy: six stress axes and live HTTP integration test departure from training assumptions. Answer in short sentences with paths.

---

## Slide 37 · Q&A backup (2/2) — agent & integration

### 中文旁白

智能体与集成高频题：模型选型、No-RAG 含义、cloud_only 不推荐、三次 timeout 仍算成功降解、C7 路径一致性、何时重跑。答句都指向 JSON/MD 证据，避免口头新数字。  
可补充：judge mean 4.10 来自 99/99 scored；144 events 来自 orchestrator meta。网页 UI 演示另场进行，本 deck 用 fault_injection_demo 表作 C7 证据。  
与 slide 36 一样，现场扫一眼即可，不要占用主讲时间。

### English Narration

High-frequency agent and integration questions—model choice, No-RAG meaning, cloud_only rejection, three timeouts as absorbed failures, C7 inject path parity, when to rerun—all point to JSON and markdown evidence without new oral numbers.  
Add judge four point one zero on ninety-nine scores and one hundred forty-four orchestrator events if needed. The browser UI demo is off-deck; this deck cites the five-scenario fault-injection report for C7. Scan, do not monopolise main talk time.

---

## Slide 38 · Thank you

### 中文旁白

致谢页保持简洁。说明可在第二屏打开终稿 PDF 或附录 A6 复现命令，愿意针对任意 C1–C8 组件 deep dive。网页系统演示若尚未进行，可说明将按 `docs/网页演示指南.md` 单独展示。  
中英文都可收尾；语气温和、邀请提问。不要在此页展开新数字。  
记录评委问题，事后可更新 FAQ 进附录。

### English Narration

Keep thanks minimal. Offer a second screen with `final_report.pdf` or appendix A6 for reproduction questions and any C1–C8 deep dive. If the browser demo has not run yet, note it follows `docs/网页演示指南.md` as a separate segment. Invite questions warmly—no new numbers here. Note follow-up questions for FAQ updates.

---

## Appendix A1 · Full class sample counts

### 中文旁白

附录 A1 回答「某一类有多少样本」。总计 50500，Normal 类最多，故障类各 3333 或 4375，比例差约 2.4 倍，训练用加权损失缓解。若评委质疑某类过少，指本表与 `data/splits/meta_*.csv` 交叉验证。  
不必主讲背诵，追问时翻页即可。数字与 2026-06-03 data card 完全一致。

### English Narration

Appendix A1 lists per-class counts totaling fifty thousand five hundred—Normals oversampled, faults balanced via weighted loss. Point to split CSVs if challenged. Numbers match the 2026-06-03 data card—flip to this slide on demand, do not memorise in the main talk.

---

## Appendix A2 · Simulation formulas (summary)

### 中文旁白

A2 给关心「仿真物理是否靠谱」的老师。PV 用简化 NOCT 与 DC/AC 功率关系；BESS 用 RC 等效与 SOC/内阻/循环计数代理量。故障注入对每个 label 独立纯函数，seed 可控。  
强调：仿真不是为了追真实电站每一伏特，而是为了可复现、可应力测试、可写 data card。真数据禁止时，这是合规且可辩护的路径。

### English Narration

Appendix A2 summarises PV NOCT-style thermal and electrical relationships and BESS RC-flavoured SOC/resistance trajectories—seedable per-fault injectors. The goal is reproducible stress testing and a complete data card, not perfect plant cloning—appropriate when real data is forbidden.

---

## Appendix A3 · Robustness figure index (PV)

### 中文旁白

A3 是 PV 九图导航，答辩若被问「某轴曲线在哪」直接翻页指缩略图。主文已用 heatmap、histogram、risk-coverage，其余可在 Q&A 展开。路径均在 `reports/robustness/pv/figures/`。

### English Narration

Appendix A3 indexes nine PV robustness figures—use thumbnails to answer “where is the noise curve?” Main slides already cite key plots; others are one page away under `reports/robustness/pv/figures/`.

---

## Appendix A4 · Robustness figure index (BESS)

### 中文旁白

A4 与 A3 对称，BESS 侧 FGSM 与 missing features 通常更「难看」，适合解释 INT8 与 adversarial 脆弱性同一根源。指图时强调 2026-06-03 同批生成。

### English Narration

Appendix A4 mirrors A3 for BESS—FGSM and missing-feature plots often look worse, supporting the INT8 and adversarial narrative. All generated in the same 2026-06-03 batch.

---

## Appendix A5 · Agent eval keyword-drift examples

### 中文旁白

A5 解释「为什么 keywords 不是 100%」。ambiguous 场景故意模糊，LLM 措辞与 oracle 短语不完全匹配，启发式扣在 keywords 槽。安全槽 urgency/forbidden/knowledge 仍稳。可对比 No-RAG 同场景 kb=0 仍 0.75，说明 surface 分误导。  
追问「举一条失败例」时读场景 ID 即可，不必展开全文 recommendation。

### English Narration

Appendix A5 explains keyword drift—ambiguous scenarios where LLM wording misses oracle phrases while safety slots hold. Contrast No-RAG: score may stay zero point seven five with zero KB—provenance failure masked by rubric. Cite scenario IDs, not full text.

---

## Appendix A6 · Full reproduction commands (PowerShell)

### 中文旁白

A6 放完整 PowerShell 块给「怎么复现」追问。幻灯片只列骨架，完整可复制段落在 `复现指南.md` §4，含 judge 环境变量与 http-timeout 120。强调 Ollama 需先 serve，agent_eval 耗时长属正常。  
若评委要 overnight 复现，指 §4 顺序与 §5 自检清单（pytest + Test-Path 产物）。

### English Narration

Appendix A6 holds the reproduction skeleton; the full copy-paste block is `复现指南.md` section four with judge variables and orchestrator timeout one hundred twenty. Note Ollama must be running—agent_eval is slow with a real LLM. Point to section five checklists for verification.

---

