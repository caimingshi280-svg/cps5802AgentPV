# docs — 文档索引

与运行时代码并列的**说明性资产**：作业对照、复现步骤、数据卡片、答辩演示等。

---

## 当前文件一览（`docs/`）

| 文件 | 说明 |
|------|------|
| [`Reproducibility Guide.md`](Reproducibility%20Guide.md) | **主复现文档**：作业 C1–C8 对照、PowerShell 命令、自检与 FAQ |
| [`Document Interpretation.md`](Document%20Interpretation.md) | 仓库目录与主要源文件用途索引 |
| [`Dashboard Demo Guide.md`](Dashboard%20Demo%20Guide.md) | Streamlit 答辩现场演示（约 5 分钟） |
| [`data_card.md`](data_card.md) | Component 1 数据卡片 |
| [`alert_schema.json`](alert_schema.json) | 边缘 → 云端告警 JSON Schema |
| [`AgentPV-项目方案.md`](AgentPV-%E9%A1%B9%E7%9B%AE%E6%96%B9%E6%A1%88.md) | 模块工程方案与规划 |
| [`CPS-5802-Project-SP26.pdf`](CPS-5802-Project-SP26.pdf) | 课程作业原文 PDF |

---

## 交付物（`reports/`）

| 文件 | 组件 |
|------|------|
| `model_eval.md` + `pv/`、`bess/` | C3 模型评测 |
| `robustness_eval.md` | 鲁棒性 / OOD 扩展 |
| `agent_eval.md` | C5 智能体评测 |
| `integration_eval.md` + `integration/` | C6 集成与延迟 |
| `integration/fault_injection_demo.md` | C7 脚本化五场景 |
| `final_report.md` / `final_report.pdf` | C8 学术终稿 |
| `AgentPV_Final_Presentation.pptx` | Final Presentation 幻灯片 |

子报告由 `scripts/render_*` 或 `python -m evaluation` 生成后，**手工同步**终稿 `final_report.md` 与 PDF / PPTX。

---

## 推荐阅读顺序

**答辩前自检**

1. `Reproducibility Guide.md` §5 自检清单  
2. `Dashboard Demo Guide.md` 彩排 Streamlit  
3. 打开 `reports/final_report.pdf` 与 `AgentPV_Final_Presentation.pptx` 核对数字一致  

**读代码**

1. `Document Interpretation.md` → 定位源文件  
2. 对应子包 `README.md`（如 `agent/README.md`）  

---

## 维护提示

- 改仿真规模或类别 → 更新 `data_card.md`  
- 重跑评测脚本 → 同步 `reports/final_report.md` 与 PDF  
- 答辩演示默认 **本机 uvicorn + Streamlit**，勿与 `docker compose` 同时占用 8000 / 8001 / 8501  
