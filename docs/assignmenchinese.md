这份文档是 **Wenzhou Kean University (温州肯恩大学)** 2026年春季学期 **CPS 5802 — Machine Learning and Innovations** 课程的期末大作业说明。

为了方便你直接阅读和复制，我已将文档内容整理为标准的 Markdown 格式。

---

### **CPS 5802 — Machine Learning and Innovations**

**Spring 2026 Course Project: AgentPV** **An Agentic AI System for Intelligent Monitoring of Solar Panels and Battery Energy Storage**

**Instructor:** Dr. Omar Dib

---

#### **1. 项目背景与动机 (Background and Motivation)**

**1.1 现有问题 (The Problem)** 当前的光伏 (PV) 和电池储能系统 (BESS) 监控系统主要依赖人工，存在以下痛点：

- **被动检测**：故障通常在发生后才被发现。
- **静态维护**：维护计划不随系统状态自适应调整。
- **人工解读**：操作员需手动解释传感器数据。
- **人力短缺**：分布式部署区域缺乏现场技术人员。 这导致了计划外停机、设备加速老化和运营成本增加。

**1.2 技术机遇 (The Opportunity)** 两项技术进步使得软件解决方案成为可能：

1. **边缘部署机器学习**：紧凑、量化的深度学习模型可在软件模拟的边缘节点上进行实时多类故障分类。
2. **Agentic AI (智能体 AI)**：配备检索增强生成 (RAG) 和工具使用能力的大语言模型 (LLM)，可以解释结构化传感器警报，检索领域知识，并生成可解释的维护建议。

---

#### **2. 项目概述 (Project Overview)**

**2.1 系统概念 (AgentPV)** 这是一个云-边协同的智能监控与决策支持系统，包含三个功能层：

1. **模拟层 (Simulation Layer)**：基于物理环境的软件，生成包含正常运行和多种故障条件的标记时间序列数据。
2. **边缘 AI 层 (Edge AI Layer)**：多类故障分类模型，经过压缩和优化以实现低延迟推理，并发出结构化安全警报。
3. **云端智能体层 (Cloud Agent Layer)**：检索增强的 LLM 智能体，接收边缘警报，检索相关领域知识，推理故障上下文，并生成可操作的维护建议。

**2.2 项目限制**

- **非硬件项目**：所有边缘组件在软件中实现。
- **非单一任务**：训练分类器只是 7 个组件之一。
- **非聊天机器人**：LLM 智能体必须基于 RAG 和工具调用，具备可评估性。

**2.3 团队结构**

- 3-4 人一组。
- 每个成员必须能够解释系统的任何部分。

---

#### **3. 系统架构 (System Architecture)**

**3.1 警报接口契约 (Alert Interface Contract)** 边缘模块与云端智能体通过结构化的 JSON 警报通信。**所有团队必须遵守以下 Schema：**

```json
{
  "timestamp": "<ISO 8601>",
  "system_id": "<string>",
  "system_type": "PV" | "BESS",
  "fault_class": "<string>", // 预测标签
  "severity": "monitor" | "warning" | "critical",
  "confidence": <float 0-1>,
  "sensor_snapshot": { ... } // 关键传感器读数
}

```

---

#### **4. 项目组件 (Project Components)**

项目分解为 7 个组件：

**4.1 组件 1 — 数据生成 (Data Generation)**

- **目标**：生成包含正常运行和多种故障类别的标记时间序列数据集。
- **要求**：
  - **光伏 (PV)**：至少包含 7 类故障（正常、局部遮挡、积尘、旁路二极管故障、组串断开、逆变器故障、老化）。
  - **电池 (BESS)**：至少包含 5 类故障（正常、容量衰减、内阻增加、热异常、电芯不平衡）。
  - **规模**：总样本量 $\ge$ 50,000，包含至少 3 种运行条件。
  - **交付**：数据卡片 (Data Card) 描述数据集模式和限制。

**4.2 组件 2 — 模型构建 (Model Build)**

- **目标**：训练多类故障分类模型并优化以适应资源受限环境。
- **要求**：
  - 必须是多类分类器（非二分类）。
  - 采用适合时间序列的架构（如 1D CNN, LSTM）。
  - **必须应用至少一种压缩技术**：结构化剪枝、INT8 量化或知识蒸馏。
  - **导出为 ONNX 格式**。
  - **约束**：模型大小 $\le$ 50MB，推理延迟 $\le$ 100ms (CPU-only)。

**4.3 组件 3 — 模型评估 (Model Evaluation)**

- **目标**：严格评估故障分类模型。
- **指标**：
  - 宏平均 F1 分数 (目标 $\ge$ 90%)。
  - 混淆矩阵 (Confusion Matrix)。
  - 推理延迟 (P95)。
  - 压缩权衡分析 (精度损失 vs. 速度/大小增益)。
- **要求**：比较至少两个模型变体（例如全精度 vs. 量化）。

**4.4 组件 4 — LLM 智能体构建 (LLM Agent Build)**

- **目标**：实现基于 RAG 的云端 LLM 智能体。
- **架构**：ReAct 循环 (Reason + Act)。
- **知识库**：构建包含 $\ge$ 30 个文档的领域特定知识库（故障描述、维护建议等）。
- **工具**：必须包含检索知识、获取系统历史、估计剩余寿命 (RUL) 等工具。
- **输出**：必须包含推荐操作、紧迫性、推理轨迹 (Reasoning Trace) 和置信度。

**4.5 组件 5 — LLM 智能体评估 (LLM Agent Evaluation)**

- **目标**：评估智能体在运维场景中的决策质量。
- **基准**：构建包含 $\ge$ 30 个场景的基准测试（含模糊故障场景）。
- **评分维度 (1-5分)**：
  1. **正确性 (Correctness)**：是否解决了实际故障？
  2. **可操作性 (Actionability)**：操作员能否直接执行？
  3. **可解释性 (Interpretability)**：推理轨迹是否清晰？
  4. **安全性 (Safety)**：是否避免了危险操作？
- **消融研究 (Ablation Study)**：评估无 RAG 和无推理轨迹情况下的性能下降。

**4.6 组件 6 — 决策与系统集成 (Decision Making and System Integration)**

- **目标**：将边缘模块和云端智能体集成。
- **要求**：
  - 支持至少 10 个并发模拟节点。
  - **端到端延迟**：从故障事件到智能体推荐 $\le$ 10 秒 (P95)。
  - **Docker 化**：使用 Docker Compose 打包（至少包含边缘服务和智能体服务）。

**4.7 组件 7 — 原型与演示 (Prototype and Demonstration)**

- **目标**：构建面向操作员的仪表盘。
- **要求**：
  - Web 界面（Streamlit, Flask 等）。
  - 显示实时警报、严重性颜色编码、推理轨迹。
  - 支持交互式场景演示。
  - 可通过 `docker compose up` 启动。

---

#### **5. 交付物 (Deliverables)**


| 编号  | 交付物                         | 描述                       |
| --- | --------------------------- | ------------------------ |
| 1   | **DataCard**                | 数据集描述、方法论、局限性            |
| 2   | **Dataset**                 | 可复现的标记数据集 ($\ge$ 50k 样本) |
| 3   | **Edge Model**              | 训练好的、压缩的、ONNX 格式的分类器     |
| 4   | **Model Evaluation Report** | 分类报告、混淆矩阵、压缩权衡分析         |
| 5   | **LLM Agent**               | 基于 RAG 的 ReAct 智能体       |
| 6   | **Agent Evaluation Report** | 基准测试结果、消融研究、评分           |
| 7   | **Integrated System**       | Docker 化的端到端系统           |
| 8   | **Operator Dashboard**      | 交互式原型界面                  |
| 9   | **Final Report**            | 技术报告                     |
| 10  | **Final Presentation**      | 现场演示和答辩                  |


---

#### **6. 最终报告结构 (Final Report Structure)**

1. **Introduction**：问题陈述与方案。
2. **Related Work**：至少 8 篇相关论文。
3. **Data Generation**：模拟方法与统计。
4. **Edge AI Module**：架构、训练与压缩。
5. **LLM Agent**：设计、知识库与工具。
6. **Evaluation**：模型与智能体评估。
7. **System Integration**：架构与演示。
8. **Discussion**：局限性与未来方向。
9. **Conclusion**。
10. **References**。
11. **Appendix**：数据卡片、Schema、Docker 指令。

---

#### **7. 评分标准 (Grading Rubric)**


| 组件                      | 权重       |
| ----------------------- | -------- |
| Component 1 (数据生成)      | 15%      |
| Component 2 (模型构建)      | 15%      |
| Component 3 (模型评估)      | 10%      |
| Component 4 (LLM 智能体构建) | 15%      |
| Component 5 (LLM 智能体评估) | 10%      |
| Component 6 (系统集成)      | 15%      |
| Component 7 (原型演示)      | 10%      |
| Final Report Quality    | 10%      |
| **总计**                  | **100%** |


---

#### **8. 技术约束 (Technical Constraints)**

- **语言**：Python 3.10+ (仪表盘前端可用 JS/HTML/CSS)。
- **可复现性**：必须固定随机种子，使用版本控制。
- **无硬件要求**：边缘约束在软件中模拟 (CPU-only, 内存限制)。
- **LLM API**：可使用 DeepSeek, Qwen, OpenAI 或本地模型 (Ollama)。
- **禁止行为**：
  - 禁止直接使用 Kaggle/HuggingFace 的现成数据集（必须自己生成）。
  - 禁止仅提交提示工程的聊天机器人（必须有 RAG 和工具调用）。
  - 禁止仅提交 Notebook 截图（必须是运行中的 Web 应用）。

---

#### **9. 推荐工具 (Recommended Tools)**

- **光伏模拟**：`pvlib-python`
- **电池模拟**：简单 RC 等效电路模型
- **时间序列**：`tsai` 或 PyTorch 1D CNN
- **模型压缩**：`torch.quantization`, `torch.nn.utils.prune`
- **ONNX**：`torch.onnx.export`, `onnxruntime`
- **RAG/Agent**：`LangChain` / `LlamaIndex`, `chromadb` / `faiss`
- **仪表盘**：`Streamlit` (最快), `Dash`, `Flask`
- **容器化**：`Docker Compose`

---

#### **10. 学术诚信 (Academic Integrity)**

- 允许使用 AI 编码助手 (Copilot, Claude 等)。
- **核心要求**：提交的每一行代码都必须被团队成员理解。
- **违规判定**：在答辩中无法解释 AI 生成的代码将被视为学术不端。
- 报告必须由团队撰写，AI 仅用于语法检查。

