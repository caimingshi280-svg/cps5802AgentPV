# AgentPV 网页演示指南（C6 / C7）

> **用途**：Final Presentation 现场 Web 演示（接 PPT C7 页之后）。  
> **地址**：http://localhost:8501  
> **建议时长**：约 **5 分钟**（2 次自动一键 + 1 次侧栏手动）。  
> **服务启动**：见 [`Reproducibility Guide.md`](Reproducibility%20Guide.md) §3.7、§3.11，在演示前完成，不计入 5 分钟。

---

## 1. 上场前检查

在仓库根目录、已激活 `.venv`、`$env:APP_ENV = "dev"`：

| 终端 | 命令 |
|------|------|
| A · Edge :8000 | `python -m uvicorn api.edge_service:app --host 0.0.0.0 --port 8000` |
| B · Agent :8001 | `python -m uvicorn api.agent_service:app --host 0.0.0.0 --port 8001` |
| C · Streamlit | `streamlit run dashboard/app.py` |

另需 **Ollama** 运行且已 `ollama pull llama3.2`（与 `configs/dev.yaml` 一致）。

侧栏 **Event log path** 建议：`data/orchestrator/events_c7_demo.jsonl`（与 C7 脚本一致）。

确认网页顶部 **Edge service / Agent service** 均为 **Online**。

---

## 2. 五分钟演示流程

| 时间 | 操作 | 讲解要点 |
|------|------|----------|
| 0:00 | 承接 PPT，全屏浏览器 | 与 `fault_injection_demo.md` 同路径，非 mock 分支 |
| 0:20 | 扫过五个一键按钮 + 四步流水线表（不点） | 五场景对应 C7；Normal 不触发 Agent；Degrade 跳过 Agent |
| 0:40 | 点击 **PV Inverter Fault**，等 spinner | 四格：仿真 → 边缘 &lt;30 ms → Agent 8～20 s → JSONL；建议 + RAG 引用 |
| 2:50 | 点击 **Edge-only Degrade** | 第三步 **Skipped**（非 Failed），对应 C6 优雅降级 |
| 3:20 | 侧栏 **Refresh** → Event timeline → Event detail 的 trace 表 | 审计视图与 ReAct 复盘 |
| 4:00 | 侧栏 **Manual fault injection**：改 seed 或故障类 → Run | 与一键按钮、`inject.py`、C7 脚本同后端 |
| 4:50 | 收尾 | 边缘 P95 &lt;100 ms；全链路 P95 &lt;10 s；详见 `final_report.pdf` |

**等待规则**：全链路只等 **1 次** Agent spinner；两次注入间隔 ≥10 s。Agent 若 ❌，改演示 Edge-only + 手动 Edge only，勿连点重试。

---

## 3. 五个一键场景

| 按钮 | 严重度 | Agent |
|------|--------|-------|
| PV Inverter Fault | critical | 调用 |
| PV Partial Shading | warning | 调用 |
| BESS Thermal Anomaly | critical | 调用 |
| PV Normal | monitor | **跳过** |
| Edge-only Degrade | critical | **跳过** |

---

## 4. 常见问题

| 现象 | 处理 |
|------|------|
| Agent 第三步 ❌，约 120 s | Ollama / 8001 未就绪；先演示 Edge-only |
| 注入成功但 Tab 无新行 | 点 **Refresh**；核对 Event log path |
| 服务 Offline | 启动 uvicorn；端口占用则复用已有进程 |
| 与 Docker 冲突 | 答辩用本机三终端；`docker compose down` 后再起 |

---

## 5. 与脚本报告的关系

```powershell
python scripts/demo_fault_injection.py --events-path data/orchestrator/events_c7_demo.jsonl
```

产出 `reports/integration/fault_injection_demo.md`——幻灯片上的五场景表；浏览器是同一管线的 live 操作员视图。
