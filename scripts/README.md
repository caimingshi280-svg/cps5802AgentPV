# scripts — 命令行辅助脚本

本目录为 **一次性或可重复执行的分析/报告/压测脚本**，不作为被其他包 `import` 的稳定库 API（内部工具函数除外）。

## 脚本索引

| 脚本 | 作用 | 典型前置条件 |
|------|------|----------------|
| `run_robustness_eval.py` | 鲁棒性、OOD、选择性预测全流程评估 | 已训练模型与 `evaluation` 依赖数据 |
| `run_dev_first_artifacts.py` | 开发期首批 JSON/Markdown 产物 | `APP_ENV=dev` |
| `bootstrap_kb_documents.py` | 知识库 Markdown 初始化/补全 | 无 |
| `e2e_latency_bench.py` | Edge + Agent HTTP 延迟（C6） | 本机已起 `edge_service` / `agent_service` |
| `render_integration_eval_report.py` | 根据 JSON/JSONL 生成 `reports/integration_eval.md` 与图 | 先有 bench 与 orchestrator 输出 |
| `render_agent_eval_report.py` | 根据 `agent_eval` 结果 JSON 生成报告与图 | 先 `python -m agent_eval ...` |
| `demo_fault_injection.py` | C7 多场景故障注入演示与报告 | Edge/Agent 可选；见脚本 `--help` |
| `_count_agent_eval_signals.py` | 从日志统计遥测（UTF-8 / UTF-16） | 日志文件路径参数 |

## 推荐顺序（与 `docs/Reproducibility Guide.md` 一致）

1. 数据 → 训练 → 量化 → `python -m evaluation --compare`  
2. `run_robustness_eval.py`（可选）  
3. `agent_eval` + `render_agent_eval_report.py`  
4. 起服务 + `e2e_latency_bench.py` + orchestrator + `render_integration_eval_report.py`  
5. `demo_fault_injection.py`（可选）  

终稿 PDF 与答辩 PPT 在 `reports/` 中维护，见 `reports/final_report.md` 与 `reports/AgentPV_Final_Presentation.pptx`。

## 运行方式

均在仓库根目录执行，例如：

```powershell
python scripts/run_robustness_eval.py
python scripts/render_agent_eval_report.py --help
```
