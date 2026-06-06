```markdown
# CPS 5802 — Machine Learning and Innovations Spring 2026

# Course Project: AgentPV  
## An Agentic AI System for Intelligent Monitoring of Solar Panels and Battery Energy Storage

**CPS 5802 — Machine Learning and Innovations**  
**Wenzhou Kean University — Spring 2026**  
**Instructor: Dr. Omar Dib**

---

# Why this project?

Most machine learning courses end at model training. This project starts there.

You will build a full intelligent system that integrates:

- multi-class fault detection
- model compression for real-world deployment
- a retrieval-augmented LLM agent
- an end-to-end prototype

using nothing but software.

The domain is renewable energy, one of the most data-rich and socially impactful fields in which AI is actively deployed today.

---

# 1. Background and Motivation

The global transition to renewable energy has led to massive deployment of photovoltaic (PV) solar systems and battery energy storage systems (BESS).

As of 2024:

- worldwide PV capacity additions exceeded 400 GW in a single year
- battery storage has become one of the fastest-growing segments of the energy sector

Managing these systems reliably at scale is a critical challenge.

---

# 1.1 The Problem

Current monitoring systems for PV and battery installations are largely reactive and human-dependent:

- Faults are often detected only after failure has occurred
- Maintenance schedules are static and do not adapt to system condition
- Operators must manually interpret sensor data to decide on corrective action
- In geographically distributed deployments, skilled operators are not always available on site

These limitations lead to:

- unplanned downtime
- accelerated equipment degradation
- increased operational costs

all of which are incompatible with the scale and diversity of modern energy infrastructure.

---

# 1.2 The Opportunity

Two converging technological advances make it possible to address these problems with software.

---

## 1. Edge-deployable machine learning

Compact, quantized deep learning models can now perform real-time multi-class fault classification directly on software-emulated edge nodes.

No specialized hardware is required for development.

---

## 2. Agentic AI with large language models (LLMs)

LLMs equipped with:

- retrieval-augmented generation (RAG)
- tool-use capabilities

can reason over structured sensor alerts, retrieve domain knowledge, and generate interpretable, actionable maintenance recommendations.

The LLM acts as an intelligent operational assistant rather than just a predictive tool.

---

Together, these technologies enable a new paradigm:

```text
AI not as a model, but as an intelligent system.

```

---

# 2. Project Overview

---

# 2.1 System Name and Concept

AgentPV is a cloud-edge intelligent monitoring and decision-support system for:

- PV solar operations
- battery energy storage operations

The system is organized into three functional layers.

---

## 1. Simulation Layer

A physics-based software environment that generates realistic labeled time-series data covering:

- normal operation
- multiple fault conditions

---

## 2. Edge AI Layer

A multi-class fault classification model:

- compressed
- optimized for low-latency inference

The model continuously monitors sensor streams and issues structured safety alerts.

---

## 3. Cloud Agent Layer

A retrieval-augmented LLM agent that:

- receives alerts from the edge
- retrieves relevant domain knowledge
- reasons through fault context
- generates interpretable and traceable maintenance recommendations for operator review

---

# 2.2 What This Project Is Not

- It is not a hardware project. All edge components are implemented in software and benchmarked on CPU-constrained virtual environments.
- It is not a single-task ML project. Training a classifier is one component of seven.
- It is not a chatbot wrapper. The LLM agent must be grounded, structured, and evaluated against defined criteria.

---

# 2.3 Team Structure

Projects are completed in teams of 3 to 4 students.

Each team is responsible for all seven components described later in the document.

Teams may divide labor internally, but every member must be able to explain and defend any part of the system during the final presentation.

---

# 3. System Architecture

The AgentPV system follows a three-layer cloud-edge design.

Architecture flow:

```text
[Simulation Environment]
↓ labeled time-series data

[Edge AI Module]
Multi-class fault classifier
(compressed, CPU-constrained)

↓ structured JSON alert

[Cloud LLM Agent]
RAG → Reasoning → Tool use → Recommendation

↓ interpretable output

[Operator Dashboard / Prototype Interface]

```

---

The edge layer is responsible for:

- speed-critical tasks
- low-context tasks

The cloud layer performs:

- context-rich reasoning
- knowledge-intensive operations

The two layers communicate through a structured alert interface.

---

# 3.1 Alert Interface Contract

The edge module communicates with the cloud agent via a structured JSON alert.

All teams must conform to the following schema:

```json
{
  "timestamp": "<ISO 8601>",
  "system_id": "<string>",
  "system_type": "PV" | "BESS",
  "fault_class": "<string>",
  "severity": "monitor" | "warning" | "critical",
  "confidence": <float 0-1>,
  "sensor_snapshot": {
    "...": "..."
  }
}

```

This schema is fixed.

- Your edge module must produce it
- Your cloud agent must consume it

---

# 4. Project Components

The project is decomposed into seven components.

They are not strictly sequential.

Components:

- 2
- 3
- 4
- 5

can proceed in parallel once Component 1 is complete.

---

# 4.1 Component 1 — Data Generation

---

## 4.1.0 Objective

Produce a labeled, reproducible time-series dataset covering:

- normal operation
- multiple fault classes

for both:

- PV systems
- battery systems

---

## 4.1.0 Context

Real industrial data from PV and battery deployments is:

- proprietary
- expensive to obtain
- difficult to share

Physics-based simulation is the standard academic approach because it allows:

- controlled fault injection
- full labeling
- reproducible splits

Your simulation does not need to be a perfect physical model.

It needs to generate data with statistically distinguishable signatures across fault classes.

You are building a data pipeline, not a physics research project.

---

# 4.1.0 PV Fault Classes

Your dataset must include at least the following seven PV fault types as distinct class labels:

1. Normal operation
2. Partial shading
3. Soiling / dust accumulation
4. Bypass diode fault
5. String disconnection
6. Inverter fault
7. Degradation (long-term efficiency loss)

---

# 4.1.0 Battery Fault Classes

Your dataset must include at least the following five battery fault types:

1. Normal operation
2. Capacity fade (gradual)
3. Internal resistance increase
4. Thermal anomaly
5. Cell imbalance

---

# 4.1.0 Requirements

- Minimum 50,000 labeled samples total across all classes
- Class distribution must be documented
- Severe imbalance must be addressed
- At least three operating conditions
  - high irradiance
  - low irradiance
  - high temperature
- Dataset must be split into:
  - train
  - validation
  - test
- Example split:
  - 70 / 15 / 15
- Splits must be fixed and reproducible via a random seed

---

## Deliver a Data Card

A short document describing:

- dataset schema
- class distribution
- generation parameters
- known limitations

---

# 4.1.0 Suggested Tools

```text
numpy
pandas
scipy
pvlib

```

For battery simulation:

```text
Custom battery ECM (equivalent circuit model)
implemented in Python

```

---

# 4.2 Component 2 — Model Build

---

## 4.2.0 Objective

Train a multi-class fault classification model on the simulated dataset and optimize it for:

- low-latency
- resource-constrained inference

---

## 4.2.0 Context

The edge module is the first line of defense.

It must classify incoming sensor streams into one of the fault classes defined previously:

- in real time
- under strict resource limits

In a production system this would run on an embedded device.

In this project, constraints are simulated in software by imposing:

- CPU-only inference
- memory caps

---

# 4.2.0 Requirements

- The model must be multi-class, not binary
- All fault classes are candidate outputs

---

## Architecture

Architecture must suit time-series data.

Examples:

- 1D CNN
- LSTM
- Transformer
- lightweight ensemble

You must justify your choice.

---

## Compression

Apply at least one compression technique:

- structured pruning
- INT8 quantization
- knowledge distillation

---

## Export

Export final model to:

```text
ONNX format

```

for portable deployment.

---

## Constraints

- Compressed model ≤ 50MB
- Inference latency ≤ 100ms
- CPU-only benchmarking
- No GPU inference

---

## Alert Severity Output

The model must implement:

```text
monitor
warning
critical

```

mapped from:

- predicted class
- confidence score

---

# 4.2.0 Suggested Tools

```text
PyTorch
TensorFlow
torch.onnx
tf2onnx
onnxruntime

```

---

# 4.3 Component 3 — Model Evaluation

---

## 4.3.0 Objective

Rigorously evaluate the fault classification model across:

- all classes
- deployment constraints

---

## 4.3.0 Context

In safety-critical systems, high overall accuracy is not sufficient.

Example:

A model achieving 95% accuracy by predicting mostly Normal class while failing on rare faults is dangerous.

You must evaluate:

- per-class behavior
- engineering reliability

---

# 4.3.0 Required Metrics

---

## Classification Metrics

- Macro-average F1-score across all classes
  - target ≥ 90%
- Per-class precision
- Per-class recall
- Per-class F1

All reported in tables.

---

## Visualization

- Confusion matrix heatmap

---

## Deployment Metrics

- Mean inference latency
- 95th percentile latency
- 1000 CPU runs
- Model size before compression
- Model size after compression

---

## Compression Trade-off

Analyze:

- accuracy loss
- speed improvement
- size reduction

---

# 4.3.0 Additional Requirements

- Compare at least two model variants
- Example:
  - full vs quantized
  - CNN vs Transformer

---

## Error Analysis

Characterize:

- which classes are confused
- why confusion occurs

---

## Unified Model Reporting

If your model handles both PV and BESS:

- report results separately

---

# 4.4 Component 4 — LLM Agent Build

---

## 4.4.0 Objective

Implement a cloud-based LLM agent that:

- receives structured alerts
- produces actionable maintenance recommendations

---

## 4.4.0 Context

A vanilla LLM prompt is not sufficient.

General-purpose LLMs do not know:

- your fault taxonomy
- your alert schema
- maintenance procedures
- battery safety logic

You must ground the agent with:

- a domain-specific knowledge base
- transparent reasoning
- auditable decision flow

---

# 4.4.0 Architecture — ReAct Loop

The agent must implement a ReAct loop.

---

## 1. Observe

Receive the structured JSON alert.

---

## 2. Reason

Analyze:

- fault class
- severity
- sensor snapshot

Retrieve relevant knowledge.

---

## 3. Act

Invoke one or more tools.

---

## 4. Reflect

Assess whether retrieved information is sufficient.

---

## 5. Report

Generate a structured recommendation with explicit reasoning trace.

---

# 4.4.0 Knowledge Base

Construct a domain-specific knowledge base containing at least:

```text
30 curated documents

```

Possible contents:

- fault descriptions
- common causes
- maintenance actions
- safety standards
- operating thresholds
- normal sensor ranges

---

## RAG Requirement

The knowledge base must be indexed for retrieval-augmented generation.

The agent retrieves relevant chunks at query time instead of loading all documents into prompts.

---

# 4.4.0 Required Agent Tools

Your agent must provide at least:

```python
retrieve_knowledge(query)
get_system_history(system_id, window)
estimate_rul(system_id)
escalate(system_id, reason)

```

---

# 4.4.0 Output Format

Every agent response must contain:

- recommended_action
- urgency
- reasoning_trace
- knowledge_sources
- confidence

---

## Urgency Values

```text
immediate
scheduled
monitor

```

---

## Confidence Values

```text
low
medium
high

```

---

# 4.4.0 Suggested Tools

```text
DeepSeek
Qwen
OpenAI
Ollama
LangChain
LlamaIndex
FAISS
ChromaDB

```

---

# 4.5 Component 5 — LLM Agent Evaluation

---

## 4.5.0 Objective

Evaluate the LLM agent’s decision quality using a curated benchmark.

---

## 4.5.0 Context

LLM evaluation is different from classifier evaluation.

There is no single ground-truth label.

You must define:

- evaluation criteria
- operational scenarios
- scoring methodology

Possible approaches:

- human judgment
- rubric scoring
- LLM-as-judge

You must discuss tradeoffs.

---

# 4.5.0 Benchmark Construction

Build at least:

```text
30 operational scenarios

```

Each scenario includes:

- synthetic JSON alert
- expected expert outcome
- severity annotation

---

## Severity Levels

```text
low
medium
high stakes

```

---

## Coverage Requirements

Scenarios must cover:

- all fault classes
- PV systems
- BESS systems

At least five scenarios must include:

```text
ambiguous or overlapping fault signatures

```

---

# 4.5.0 Evaluation Criteria

Each response scored 1–5.


| Dimension        | Definition                                        |
| ---------------- | ------------------------------------------------- |
| Correctness      | Does the recommendation address the actual fault? |
| Actionability    | Can an operator act without clarification?        |
| Interpretability | Is reasoning clear and supported?                 |
| Safety           | Does the recommendation avoid harmful actions?    |


---

## Target

```text
Mean score ≥ 4.0

```

across all dimensions and scenarios.

---

# 4.5.0 Ablation Study

Evaluate under degraded conditions.

---

## 1. No RAG

The agent receives only alerts.

No knowledge retrieval.

---

## 2. No Reasoning Trace

The agent produces direct recommendations only.

---

## Goal

Show how RAG and ReAct affect:

- correctness
- safety
- interpretability
- actionability

---

# 4.6 Component 6 — Decision Making and System Integration

---

## 4.6.0 Objective

Integrate edge module and cloud agent into one unified system.

Demonstrate:

```text
fault occurrence → recommendation output

```

---

## 4.6.0 Context

Components built separately often fail when connected.

Integration introduces:

- latency accumulation
- schema mismatches
- failure handling

This component focuses on:

- reliability
- system thinking

not adding features.

---

# 4.6.0 Requirements

- Edge module and cloud agent must communicate through the fixed JSON schema

---

## Multi-node Simulation

Support at least:

```text
10 simulated nodes

```

representing multiple installations.

---

## Latency Requirement

End-to-end latency:

```text
≤ 10 seconds P95

```

measured across:

```text
50 runs

```

---

## Graceful Degradation

If the cloud agent fails or times out:

- edge alerts must still function

---

## Docker Requirement

Package the full system using:

```text
Docker Compose

```

Minimum services:

- edge-service
- agent-service

---

# 4.6.0 Ablation Study

Run and compare:

---

## 1. Edge Only

Fault alerts without cloud agent.

---

## 2. Cloud Only

Raw sensor data directly sent to agent.

No edge classifier.

---

## 3. Full System

Edge classification feeds structured alert into cloud agent.

---

## Compare

Measure:

- latency
- interpretability
- decision quality

---

# 4.7 Component 7 — Prototype and Demonstration

---

## 4.7.0 Objective

Build an operator-facing interface showing:

- system status
- fault alerts
- agent recommendations

---

## 4.7.0 Context

A system that works but cannot be understood by non-technical users has limited value.

The dashboard is the human interface of AgentPV.

It does not need to be production-grade.

It must:

- function correctly
- clearly demonstrate value

---

# 4.7.0 Requirements

---

## Web Dashboard

Framework examples:

- Streamlit
- Flask + HTML
- Dash

---

## Dashboard Features

Display:

- live or replayed alerts
- simulated nodes
- severity color coding
- reasoning trace on demand

---

## Interactive Scenario

User selects a fault type.

System injects the fault into simulation.

Dashboard shows:

```text
full pipeline response end-to-end

```

---

## Launch Requirement

Entire demonstration runnable using:

```bash
docker compose up

```

---

# 5. Deliverables


| #   | Deliverable             | Description                                                     |
| --- | ----------------------- | --------------------------------------------------------------- |
| 1   | Data Card               | Schema, class distribution, generation methodology, limitations |
| 2   | Dataset                 | Reproducible labeled dataset with fixed splits                  |
| 3   | Edge Model              | Trained, compressed, ONNX-exported classifier                   |
| 4   | Model Evaluation Report | Metrics, confusion matrix, compression analysis                 |
| 5   | LLM Agent               | Working RAG-based ReAct agent                                   |
| 6   | Agent Evaluation Report | Benchmark results and ablations                                 |
| 7   | Integrated System       | Dockerized end-to-end system                                    |
| 8   | Operator Dashboard      | Interactive prototype                                           |
| 9   | Final Report            | Technical documentation                                         |
| 10  | Final Presentation      | Live demo and Q&A                                               |


---

# 6. Final Report Structure

Your report must include:

1. Introduction
2. Related Work
3. Data Generation
4. Edge AI Module
5. LLM Agent
6. Evaluation
7. System Integration and Prototype
8. Discussion
9. Conclusion
10. References
11. Appendix

---

# 7. Grading Rubric


| Component                          | Points | Weight |
| ---------------------------------- | ------ | ------ |
| Component 1 — Data Generation      | 15     | 15%    |
| Component 2 — Model Build          | 15     | 15%    |
| Component 3 — Model Evaluation     | 10     | 10%    |
| Component 4 — LLM Agent Build      | 15     | 15%    |
| Component 5 — LLM Agent Evaluation | 10     | 10%    |
| Component 6 — Integration          | 15     | 15%    |
| Component 7 — Prototype & Demo     | 10     | 10%    |
| Final Report Quality               | 10     | 10%    |
| Total                              | 100    | 100%   |


---

## Note on Individual Accountability

During the final presentation:

- each member may be questioned on any component
- not only the component they personally built

Failure to explain teammate-built components may affect individual grading.

---

# 8. Technical Expectations and Constraints

---

# 8.1 Language and Runtime

All implementation must use:

```text
Python 3.10+

```

Frontend-only exceptions:

```text
JavaScript / HTML / CSS

```

---

# 8.2 Reproducibility

All experiments must be reproducible.

Requirements:

- fixed random seeds
- dataset versioning
- requirements.txt or pyproject.toml

The system must run on machines other than your own.

---

# 8.3 No Hardware Required

All edge constraints are simulated in software.

You benchmark using:

```text
onnxruntime CPU-only mode

```

Do not buy or borrow embedded hardware.

---

# 8.4 LLM API Usage

You may use any LLM API.

Examples:

- DeepSeek
- Qwen
- Groq
- OpenAI

Free-tier APIs are acceptable.

You may also run local models using:

```text
Ollama

```

Examples:

- llama3
- mistral

---

# 8.5 Forbidden Shortcuts

---

## 1. No Prebuilt Dataset

Do not use:

- Kaggle datasets
- HuggingFace datasets

as the primary dataset.

You must generate your own.

---

## 2. No Fake Agent

Do not submit:

```text
prompt-engineered chatbot wrappers

```

The following are required:

- RAG
- tool calls
- ReAct loop

---

## 3. No Notebook Screenshots

Do not present notebook screenshots as the prototype.

The dashboard must be:

```text
a running web application

```

---

# 9. Recommended Starting Points

---

## PV Simulation

```text
pvlib-python

```

well-documented PV modeling library.

---

## Battery Model

Implement:

```text
RC equivalent circuit model

```

using published degradation parameters.

---

## Time-series Classification

Possible options:

- tsai library
- custom 1D CNN in PyTorch

---

## Model Compression

```text
torch.quantization
torch.nn.utils.prune

```

---

## ONNX Deployment

```text
torch.onnx.export
onnxruntime.InferenceSession

```

---

## RAG Pipeline

```text
LangChain RetrievalQA
LlamaIndex VectorStoreIndex

```

---

## Vector Database

```text
chromadb
faiss-cpu

```

---

## Dashboard

```text
Streamlit
Dash
Flask + HTML

```

---

## Containerization

```text
Docker Compose

```

with:

- edge-service
- agent-service

---

# 10. Academic Integrity

You are expected to use AI coding assistants such as:

- GitHub Copilot
- Claude
- ChatGPT

as tools.

However:

- every submitted line of code must be understood
- inability to explain AI-generated code may violate academic integrity
- your report must be written by your team
- AI may assist grammar checking only
- clearly disclose AI tool usage in the appendix

---

Questions about the project should be directed to:

```text
Dr. Omar Dib

```

during office hours or via email:

```text
odib@kean.edu

```

```

```

