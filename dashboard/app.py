"""Streamlit operator dashboard for AgentPV (Component 6 MVP).

This module is the **rendering layer** only. All data loading, filtering,
and aggregation lives in :mod:`dashboard.data` so that pure logic stays
unit-testable without Streamlit.

Run locally:

    streamlit run dashboard/app.py

The dashboard reads ``data/orchestrator/events.jsonl`` (the JSONL file
produced by ``python -m orchestrator``). Refresh is **manual** — click
the "🔄 Refresh" button to re-read the file. Auto-refresh is left for
the polish phase to avoid race conditions on Windows file locks.

Layout (rule §27 minimal-viable):

* Sidebar — controls (event log path, filters, refresh).
* Tab 1 — Node overview (per-node table + status chips).
* Tab 2 — Event timeline (filtered table + activity chart).
* Tab 3 — Event detail (drill-down by event_id).
* Tab 4 — Global stats (severity / fault-class / latency).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import streamlit as st

from api.schemas import OperatingCondition, Severity, SystemType
from dashboard.data import (
    events_to_dataframe,
    fault_class_counts,
    filter_events,
    get_event_by_id,
    latency_stats,
    load_events,
    per_node_summary,
    severity_counts,
    severity_over_time,
)
from dashboard.i18n import (
    PRESET_COPY,
    button_help,
    en_zh_html,
    tab_label,
    zh_sub,
)
from dashboard.inject import (
    InjectionResult,
    fault_choices_for,
    inject_fault_demo,
)
from utils.paths import ORCHESTRATOR_DIR

SEVERITY_COLOR = {
    "monitor": "#7BB661",  # green-ish
    "warning": "#E0A800",  # amber
    "critical": "#D9534F",  # red
}

# One-click presets aligned with ``scripts/demo_fault_injection.py``.
DEMO_PRESETS: tuple[dict[str, Any], ...] = (
    {
        "id": "pv_inv",
        "system": "PV",
        "fault_class": "Inverter_fault",
        "operating_condition": "high_irradiance",
        "system_id": "DEMO-PV-INV-001",
        "seed": 4242,
        "skip_agent": False,
    },
    {
        "id": "pv_shade",
        "system": "PV",
        "fault_class": "Partial_shading",
        "operating_condition": "high_irradiance",
        "system_id": "DEMO-PV-SHADE-001",
        "seed": 13,
        "skip_agent": False,
    },
    {
        "id": "bess_thermal",
        "system": "BESS",
        "fault_class": "Thermal_anomaly",
        "operating_condition": "high_temperature",
        "system_id": "DEMO-BESS-THERMAL-001",
        "seed": 99,
        "skip_agent": False,
    },
    {
        "id": "pv_normal",
        "system": "PV",
        "fault_class": "PV_Normal",
        "operating_condition": "low_irradiance",
        "system_id": "DEMO-PV-NORMAL-001",
        "seed": 7,
        "skip_agent": False,
    },
    {
        "id": "edge_only",
        "system": "PV",
        "fault_class": "String_disconnection",
        "operating_condition": "high_irradiance",
        "system_id": "DEMO-PV-DEGRADE-001",
        "seed": 21,
        "skip_agent": True,
    },
)


def _severity_badge(severity: str | None) -> str:
    if not severity:
        return "—"
    color = SEVERITY_COLOR.get(severity, "#6c757d")
    return (
        f'<span style="background:{color};color:#fff;padding:4px 10px;'
        f'border-radius:6px;font-weight:600;">{severity.upper()}</span>'
    )


def _probe_health(base_url: str, *, timeout_s: float = 2.5) -> tuple[bool, str]:
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/healthz", timeout=timeout_s)
        if resp.status_code == 200:
            return True, "online"
        return False, f"HTTP {resp.status_code}"
    except Exception as exc:  # noqa: BLE001 — UI probe only
        return False, type(exc).__name__


def _run_preset_injection(events_path: Path, preset: dict[str, Any]) -> InjectionResult:
    edge_url = st.session_state.get("inject_edge_url", "http://127.0.0.1:8000")
    agent_url = st.session_state.get("inject_agent_url", "http://127.0.0.1:8001")
    return inject_fault_demo(
        system_type=SystemType(preset["system"]),
        fault_class=preset["fault_class"],
        operating_condition=OperatingCondition(preset["operating_condition"]),
        system_id=preset["system_id"],
        edge_url=edge_url,
        agent_url=agent_url,
        events_path=events_path,
        seed=int(preset["seed"]),
        skip_agent=bool(preset["skip_agent"]),
    )


# ---------------------------------------------------------------------------
# Page setup + sidebar
# ---------------------------------------------------------------------------


def _setup_page() -> None:
    st.set_page_config(
        page_title="AgentPV Dashboard",
        page_icon="⚡",
        layout="wide",
    )
    st.markdown(
        """
<style>
.agentpv-zh-sub {
  color: #5c6370;
  font-size: 0.88rem;
  font-weight: 400;
  line-height: 1.35;
}
.agentpv-pipeline th { font-size: 0.9rem; }
.agentpv-pipeline td { vertical-align: top; }
</style>
""",
        unsafe_allow_html=True,
    )
    st.title("AgentPV — Operator Dashboard")
    st.markdown(zh_sub("运维仪表盘 · 组件 7 交互原型"), unsafe_allow_html=True)
    st.caption(
        "Reads orchestrator JSONL events (edge → agent pipeline). "
        "Start Edge :8000, Agent :8001, and Ollama before full-pipeline demos. "
        "· 读取编排器 JSONL；全链路演示前请启动 Edge、Agent 与 Ollama。"
    )


def _sidebar(default_path: Path) -> tuple[Path, dict]:
    """Render sidebar; return (events_path, filter_dict)."""

    with st.sidebar:
        st.header("Controls")
        st.caption("控制 · 数据路径与筛选")
        path_str = st.text_input(
            "Event log path · 事件日志路径",
            value=str(default_path),
            help="Path to orchestrator JSONL · 编排器 JSONL 文件路径",
        )
        st.button(
            "🔄 Refresh · 刷新",
            width="stretch",
            help="Re-read the JSONL file · 重新读取日志",
        )
        st.divider()
        st.subheader("Filters")
        st.caption("筛选")
        sel_systems = st.multiselect(
            "System type · 系统类型",
            options=[s.value for s in SystemType],
            default=[s.value for s in SystemType],
        )
        sel_severities = st.multiselect(
            "Severity (alert) · 严重度",
            options=[s.value for s in Severity],
            default=[s.value for s in Severity],
        )
        only_with_rec = st.checkbox(
            "Only with recommendation · 仅有维护建议",
            value=False,
        )

        st.divider()
        _sidebar_fault_injection(Path(path_str))
    filters = {
        "system_types": [SystemType(s) for s in sel_systems] if sel_systems else None,
        "severities": [Severity(s) for s in sel_severities] if sel_severities else None,
        "only_with_recommendation": only_with_rec,
    }
    return Path(path_str), filters


# ---------------------------------------------------------------------------
# Fault-injection demo (Component 7 deliverable — interactive trigger)
# ---------------------------------------------------------------------------


def _render_demo_command_center(events_path: Path) -> None:
    """Top-of-page demo panel: service health, one-click presets, pipeline legend."""

    st.markdown(
        en_zh_html("🎯 Interactive Demo Console (C7)", "交互演示控制台"),
        unsafe_allow_html=True,
    )
    st.caption(
        "One-click presets → Edge ONNX classify → (optional) cloud ReAct agent → JSONL. "
        "· 一键场景 → 边缘推理 → 智能体建议 → 写入事件日志。"
    )

    edge_url = st.session_state.get("inject_edge_url", "http://127.0.0.1:8000")
    agent_url = st.session_state.get("inject_agent_url", "http://127.0.0.1:8001")
    h1, h2, h3 = st.columns(3)
    edge_ok, edge_msg = _probe_health(edge_url)
    agent_ok, agent_msg = _probe_health(agent_url)
    with h1:
        st.metric(
            "Edge service",
            "Online" if edge_ok else "Offline",
            delta=edge_url,
            delta_color="normal" if edge_ok else "inverse",
        )
        st.caption("边缘服务 · " + ("在线" if edge_ok else "离线"))
    with h2:
        st.metric(
            "Agent service",
            "Online" if agent_ok else "Offline",
            delta=agent_url,
            delta_color="normal" if agent_ok else "inverse",
        )
        st.caption("智能体服务 · " + ("在线" if agent_ok else "离线"))
    with h3:
        st.metric("Event log", events_path.name, delta=str(events_path.parent))
        st.caption("事件日志")

    if not edge_ok or not agent_ok:
        st.warning(
            f"Services not ready — Edge ({edge_msg}), Agent ({agent_msg}). "
            "Full pipeline needs both online; try **Edge-only Degrade** first. "
            "· 服务未就绪；全链路需 Edge 与 Agent 均在线，可先用「仅边缘降级」。"
        )

    st.markdown(
        en_zh_html("One-click demo scenarios", "一键演示场景（点击后自动注入）"),
        unsafe_allow_html=True,
    )
    preset_cols = st.columns(len(DEMO_PRESETS))
    for col, preset in zip(preset_cols, DEMO_PRESETS, strict=True):
        en, zh, sub_en, sub_zh = PRESET_COPY[preset["id"]]
        with col:
            if st.button(
                en,
                help=button_help(f"{sub_en} / {sub_zh}", f"{zh} · {sub_en}"),
                width="stretch",
                key=f"preset_{preset['system_id']}",
            ):
                try:
                    with st.spinner(f"Injecting {en}… / 正在注入 {zh}…"):
                        result = _run_preset_injection(events_path, preset)
                    st.session_state["inject_last_error"] = None
                    st.session_state["inject_last_result"] = result
                    st.session_state["highlight_event_id"] = result.event.event_id
                except ValueError as exc:
                    st.session_state["inject_last_error"] = str(exc)
                    st.session_state["inject_last_result"] = None

    st.markdown(
        """
<table class="agentpv-pipeline" style="width:100%; border-collapse:collapse;">
<thead><tr>
<th>Step · 步骤</th><th>Description · 说明</th>
</tr></thead>
<tbody>
<tr><td><b>1 Simulation</b><br><span class="agentpv-zh-sub">仿真</span></td>
<td>Build 60s window with injected fault label · 生成 60s 窗口并注入故障标签</td></tr>
<tr><td><b>2 Edge</b><br><span class="agentpv-zh-sub">边缘</span></td>
<td><code>POST /predict</code> → structured Alert (severity, fault_class) · 结构化告警</td></tr>
<tr><td><b>3 Agent</b><br><span class="agentpv-zh-sub">智能体</span></td>
<td><code>POST /recommend</code> → RAG + tools + playbook (warning/critical) · 维护建议</td></tr>
<tr><td><b>4 Persist</b><br><span class="agentpv-zh-sub">落盘</span></td>
<td>Append JSONL; inspect under <b>Event detail</b> tab · 追加日志，在「事件详情」查看</td></tr>
</tbody></table>
""",
        unsafe_allow_html=True,
    )


def _sidebar_fault_injection(events_path: Path) -> None:
    """Render the on-demand fault injection panel inside the sidebar."""

    with st.expander("🔥 Manual fault injection (advanced) · 手动注入", expanded=False):
        st.caption(
            "Custom system / fault / seed — same pipeline as one-click presets. "
            "· 自定义参数，与上方一键场景共用注入管线。"
        )
        sys_choice = st.radio(
            "System · 系统",
            options=[s.value for s in SystemType],
            horizontal=True,
            key="inject_system",
        )
        system_type = SystemType(sys_choice)
        fault_options = list(fault_choices_for(system_type))
        fault_class = st.selectbox(
            "Fault class · 故障类别",
            options=fault_options,
            index=min(1, len(fault_options) - 1),
            key="inject_fault_class",
            help="Use *_Normal* for healthy window · 选 Normal 表示无故障注入",
        )
        op_cond = st.selectbox(
            "Operating condition · 工况",
            options=[c.value for c in OperatingCondition],
            index=0,
            key="inject_op_cond",
        )
        default_sys_id = f"DEMO-{system_type.value}-001"
        system_id = st.text_input(
            "System ID · 系统编号",
            value=default_sys_id,
            key="inject_system_id",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            seed = st.number_input(
                "Seed · 随机种子",
                value=4242,
                min_value=0,
                max_value=2**31 - 1,
                step=1,
                key="inject_seed",
            )
        with col_b:
            skip_agent = st.checkbox(
                "Edge only · 仅边缘",
                value=False,
                key="inject_skip_agent",
                help="Skip agent — graceful degradation demo · 跳过 Agent，演示降级",
            )

        with st.expander("Service URLs · 服务地址", expanded=False):
            edge_url = st.text_input(
                "Edge URL", value="http://127.0.0.1:8000", key="inject_edge_url"
            )
            agent_url = st.text_input(
                "Agent URL", value="http://127.0.0.1:8001", key="inject_agent_url"
            )

        clicked = st.button(
            "🔥 Run injection · 执行注入",
            width="stretch",
            type="primary",
            key="inject_submit",
        )

    if not clicked:
        return

    try:
        with st.spinner(
            f"Injecting {fault_class} on {system_id} → "
            f"{'edge' if skip_agent else 'edge → agent'}…"
        ):
            result = inject_fault_demo(
                system_type=system_type,
                fault_class=fault_class,
                operating_condition=OperatingCondition(op_cond),
                system_id=system_id,
                edge_url=edge_url,
                agent_url=agent_url,
                events_path=events_path,
                seed=int(seed),
                skip_agent=bool(skip_agent),
            )
    except ValueError as exc:
        st.session_state["inject_last_error"] = str(exc)
        st.session_state["inject_last_result"] = None
        return

    st.session_state["inject_last_error"] = None
    st.session_state["inject_last_result"] = result
    st.session_state["highlight_event_id"] = result.event.event_id


def _pipeline_step_box(
    *,
    title_en: str,
    title_zh: str,
    status: str,
    detail: str,
    accent: str,
) -> None:
    st.markdown(
        f"""
<div style="border:2px solid {accent};border-radius:10px;padding:12px;
min-height:118px;background:#fafafa;">
  <div style="font-size:22px;font-weight:700;">{status}</div>
  <div style="font-size:14px;font-weight:600;margin-top:4px;">{title_en}</div>
  <div class="agentpv-zh-sub" style="margin-top:2px;">{title_zh}</div>
  <div style="font-size:12px;color:#555;margin-top:6px;">{detail}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_injection_banner() -> None:
    """Surface the latest injection with a visual end-to-end pipeline."""

    error = st.session_state.get("inject_last_error")
    result: InjectionResult | None = st.session_state.get("inject_last_result")
    if error:
        st.error(
            f"Injection rejected (validation): {error} · 注入被拒绝（参数校验失败）"
        )
        return
    if result is None:
        st.info(
            "No injection yet — use one-click scenarios above or manual injection "
            "in the sidebar. · 尚未执行注入，请使用上方一键场景或侧栏手动注入。"
        )
        return

    event = result.event
    st.markdown(
        en_zh_html("📋 Latest injection result", "最近一次注入结果"),
        unsafe_allow_html=True,
    )
    header_l, header_r = st.columns([3, 1])
    with header_l:
        sev = event.alert.severity.value if event.alert else None
        st.markdown(
            f"**System** `{event.system_id}` · **Ground truth** "
            f"`{event.ground_truth_label}` · **Predicted** "
            f"`{event.alert.fault_class if event.alert else '—'}` · "
            f"Severity {_severity_badge(sev)}  \n"
            f"{zh_sub('系统 · 真值 · 预测 · 严重度')}",
            unsafe_allow_html=True,
        )
    with header_r:
        st.caption(f"event_id · 事件编号\n`{event.event_id[:12]}…`")

    # Pipeline status boxes
    sim_ok = event.alert is not None or result.edge_error is None
    edge_ok = result.edge_error is None and event.alert is not None
    agent_skip = not result.agent_called
    agent_ok = (
        result.agent_called
        and result.agent_error is None
        and event.recommendation is not None
    )
    persist_ok = True

    p1, p2, p3, p4 = st.columns(4)
    with p1:
        _pipeline_step_box(
            title_en="1 · Simulation window",
            title_zh="仿真窗口",
            status="✅" if sim_ok else "❌",
            detail=(
                f"Window ready · {event.ground_truth_label} · 窗口已生成"
            ),
            accent="#2E86AB",
        )
    with p2:
        _pipeline_step_box(
            title_en="2 · Edge /predict",
            title_zh="边缘推理",
            status="✅" if edge_ok else "❌",
            detail=(
                f"{(result.edge_ms or 0):.1f} ms · "
                f"{event.alert.severity.value if event.alert else result.edge_error}"
            ),
            accent="#7BB661" if edge_ok else "#D9534F",
        )
    with p3:
        if agent_skip:
            _pipeline_step_box(
                title_en="3 · Agent /recommend",
                title_zh="智能体建议",
                status="⏭",
                detail="Skipped (edge-only or monitor) · 已跳过",
                accent="#6c757d",
            )
        elif agent_ok:
            _pipeline_step_box(
                title_en="3 · Agent /recommend",
                title_zh="智能体建议",
                status="✅",
                detail=(
                    f"{result.agent_ms:.0f} ms · urgency "
                    f"{event.recommendation.urgency.value}"
                ),
                accent="#7BB661",
            )
        else:
            _pipeline_step_box(
                title_en="3 · Agent /recommend",
                title_zh="智能体建议",
                status="❌",
                detail=(
                    (result.agent_error or "timeout / no response")[:80]
                    + " · 超时或无响应"
                ),
                accent="#D9534F",
            )
    with p4:
        _pipeline_step_box(
            title_en="4 · Persist JSONL",
            title_zh="写入日志",
            status="✅" if persist_ok else "❌",
            detail="Sidebar Refresh → timeline / detail · 刷新后查看",
            accent="#2E86AB",
        )

    if result.ok:
        st.success(
            "Pipeline OK — alert and recommendation (if applicable) generated. "
            "· 链路成功：边缘告警与智能体建议均已生成。"
        )
    else:
        st.error(
            "Partial failure — event still logged for audit. "
            f"Edge: {result.edge_error!r} · Agent: {result.agent_error!r} "
            "· 部分失败，事件已写入日志。"
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Edge latency", f"{(result.edge_ms or 0):.1f} ms")
    m1.caption("边缘延迟")
    m2.metric(
        "Agent latency",
        f"{result.agent_ms:.0f} ms" if result.agent_ms is not None else "—",
    )
    m2.caption("智能体延迟")
    m3.metric(
        "Confidence",
        f"{event.alert.confidence:.1%}" if event.alert else "—",
    )
    m3.caption("置信度")
    m4.metric(
        "KB citations",
        len(event.recommendation.knowledge_sources) if event.recommendation else 0,
    )
    m4.caption("知识库引用")

    rec_col, alert_col = st.columns(2)
    with rec_col:
        st.markdown(
            en_zh_html("Agent recommendation", "智能体维护建议"),
            unsafe_allow_html=True,
        )
        if event.recommendation is not None:
            st.markdown(
                f"**Urgency** `{event.recommendation.urgency.value}` · "
                f"**Confidence** `{event.recommendation.confidence.value}` · "
                f"{zh_sub('紧迫性 · 置信度')}",
                unsafe_allow_html=True,
            )
            st.info(event.recommendation.recommended_action)
            if event.recommendation.knowledge_sources:
                st.markdown("**RAG sources** · RAG 引用")
                for src in event.recommendation.knowledge_sources:
                    st.markdown(f"- `{src}`")
        elif result.agent_called:
            st.warning(
                "Agent called but no valid recommendation (often HTTP timeout). "
                "· Agent 已调用但未返回建议（常见：超时，默认 120s）。"
            )
        else:
            st.caption(
                "Agent not called (normal severity or edge-only). "
                "· 未调用 Agent。"
            )
    with alert_col:
        st.markdown(en_zh_html("Edge alert", "边缘告警"), unsafe_allow_html=True)
        if event.alert is not None:
            st.json(event.alert.model_dump(mode="json"))
        else:
            st.warning("No alert — edge call failed. · 无 Alert，Edge 失败。")

    with st.expander(
        "Full event JSON (OrchestratorEvent) · 完整事件 JSON",
        expanded=False,
    ):
        st.json(event.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------


def _render_node_overview(events: list) -> None:
    st.markdown(
        en_zh_html("Node overview", "节点总览"),
        unsafe_allow_html=True,
    )
    df = per_node_summary(events)
    if df.empty:
        st.info(
            "No events yet — run the orchestrator to populate the JSONL. "
            "· 暂无事件，请先运行编排器。"
        )
        return

    # 顶部 4 个 KPI metric
    n_nodes = df["node_id"].nunique()
    n_events = int(df["n_steps"].sum())
    n_alerts = int(df["n_alerts"].sum())
    n_recs = int(df["n_recommendations"].sum())
    n_errors = int(df["n_errors"].sum())
    cols = st.columns(5)
    cols[0].metric("Nodes", n_nodes)
    cols[1].metric("Events", n_events)
    cols[2].metric("Alerts", n_alerts)
    cols[3].metric("Recommendations", n_recs)
    cols[4].metric("Errors", n_errors)

    st.dataframe(df, width="stretch", hide_index=True)


def _render_timeline(filtered_events: list, all_events: list) -> None:
    st.markdown(
        en_zh_html("Event timeline", "事件时间线"),
        unsafe_allow_html=True,
    )
    df = events_to_dataframe(filtered_events)
    if df.empty:
        st.info(
            "No events match the current filters. · 当前筛选无匹配事件。"
        )
    else:
        df_view = df.sort_values("timestamp", ascending=False).copy()
        df_view["timestamp"] = df_view["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        display_cols = [
            "timestamp",
            "node_id",
            "system_type",
            "step_number",
            "ground_truth_label",
            "predicted_class",
            "severity",
            "confidence",
            "has_recommendation",
            "edge_elapsed_ms",
            "agent_elapsed_ms",
            "error",
        ]
        st.dataframe(
            df_view[display_cols],
            width="stretch",
            hide_index=True,
        )

    st.markdown(
        en_zh_html("Activity over time (5s buckets)", "活动趋势（5 秒分桶）"),
        unsafe_allow_html=True,
    )
    sot = severity_over_time(all_events, bucket_seconds=5.0)
    if sot.empty:
        st.caption(
            "Not enough alerts for the chart yet. · 告警不足，暂无法绘图。"
        )
    else:
        # st.bar_chart 需要 wide 形式
        wide = sot.pivot(index="bucket", columns="severity", values="count").fillna(0)
        # 保证三列出现，颜色稳定
        for col in ("monitor", "warning", "critical"):
            if col not in wide.columns:
                wide[col] = 0
        wide = wide[["monitor", "warning", "critical"]]
        st.bar_chart(wide, width="stretch")


def _render_event_detail(events: list) -> None:
    st.markdown(
        en_zh_html("Event detail", "事件详情"),
        unsafe_allow_html=True,
    )
    if not events:
        st.info("No events to inspect. · 暂无可查看事件。")
        return
    df = events_to_dataframe(events)
    df_view = df.sort_values("timestamp", ascending=False)
    options = list(df_view["event_id"])
    labels = [
        f"{eid[:8]}…  ·  {row['node_id']} step {row['step_number']}  ·  "
        f"{row['predicted_class'] or 'no-alert'} ({row['severity'] or 'n/a'})"
        for eid, (_, row) in zip(options, df_view.iterrows(), strict=True)
    ]
    highlight = st.session_state.get("highlight_event_id")
    default_index = 0
    if highlight and highlight in options:
        default_index = options.index(highlight)
    chosen = st.selectbox(
        "Select event (latest injection auto-selected) · 选择事件",
        options=options,
        index=default_index,
        format_func=lambda eid: labels[options.index(eid)],
    )
    event = get_event_by_id(events, chosen)
    if event is None:
        st.warning("Event not found.")
        return

    cols = st.columns(3)
    cols[0].metric("Severity", event.alert.severity.value if event.alert else "—")
    cols[1].metric(
        "Confidence (alert)",
        f"{event.alert.confidence:.2%}" if event.alert else "—",
    )
    cols[2].metric(
        "Urgency",
        event.recommendation.urgency.value if event.recommendation else "—",
    )

    with st.expander(
        "Alert + sensor snapshot · 告警与传感器快照",
        expanded=True,
    ):
        if event.alert is None:
            st.write(
                "No alert (edge failed or pending). · 无告警。"
            )
        else:
            st.json(event.alert.model_dump(mode="json"))
    with st.expander(
        "Recommendation · 维护建议",
        expanded=event.recommendation is not None,
    ):
        if event.recommendation is None:
            st.write(
                "No recommendation (below trigger or agent unavailable). "
                "· 无维护建议。"
            )
        else:
            st.write(f"**Action**: {event.recommendation.recommended_action}")
            st.write(
                f"**Confidence**: `{event.recommendation.confidence.value}` "
                f"·  **Urgency**: `{event.recommendation.urgency.value}` "
                f"·  **Sources**: {len(event.recommendation.knowledge_sources)}"
            )
            if event.recommendation.knowledge_sources:
                st.write("**Knowledge sources**")
                for src in event.recommendation.knowledge_sources:
                    st.write(f"- {src}")
            st.write("**Reasoning trace** · 推理轨迹")
            trace_rows = [
                {
                    "step": s.step,
                    "phase": s.phase,
                    "thought": s.thought,
                    "action": s.action or "—",
                    "result_summary": s.result_summary or "—",
                }
                for s in event.recommendation.reasoning_trace
            ]
            st.dataframe(
                pd.DataFrame(trace_rows),
                width="stretch",
                hide_index=True,
            )
    if event.error:
        st.error(f"Pipeline error: {event.error}")


def _render_global_stats(events: list) -> None:
    st.markdown(
        en_zh_html("Severity distribution", "严重度分布"),
        unsafe_allow_html=True,
    )
    sev = severity_counts(events)
    sev_chart = sev.set_index("severity")
    st.bar_chart(sev_chart, width="stretch")

    st.markdown(
        en_zh_html("Fault class distribution (alerts only)", "故障类别分布（仅告警）"),
        unsafe_allow_html=True,
    )
    fc = fault_class_counts(events)
    if fc.empty:
        st.caption("No alerts recorded yet. · 尚无告警记录。")
    else:
        st.bar_chart(fc.set_index("fault_class"), width="stretch")

    st.markdown(
        en_zh_html("Latency statistics", "延迟统计"),
        unsafe_allow_html=True,
    )
    stats = latency_stats(events)
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Edge `/predict`**")
        st.metric("samples", stats["n_edge"])
        st.metric(
            "p50 (ms)",
            f"{stats['edge_p50_ms']:.2f}" if stats["n_edge"] else "—",
        )
        st.metric(
            "p95 (ms)",
            f"{stats['edge_p95_ms']:.2f}" if stats["n_edge"] else "—",
        )
    with cols[1]:
        st.markdown("**Agent `/recommend`**")
        st.metric("samples", stats["n_agent"])
        st.metric(
            "p50 (ms)",
            f"{stats['agent_p50_ms']:.2f}" if stats["n_agent"] else "—",
        )
        st.metric(
            "p95 (ms)",
            f"{stats['agent_p95_ms']:.2f}" if stats["n_agent"] else "—",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _setup_page()
    default_path = ORCHESTRATOR_DIR / "events.jsonl"
    events_path, filters = _sidebar(default_path)

    load_result = load_events(events_path)
    if load_result.n_skipped:
        st.warning(
            f"Skipped {load_result.n_skipped} malformed line(s) while loading "
            f"`{events_path}`. First 3 reasons: "
            + " · ".join(load_result.skipped_reasons[:3])
        )

    all_events = load_result.events
    filtered = filter_events(
        all_events,
        node_ids=None,
        system_types=filters["system_types"],
        severities=filters["severities"],
        only_with_recommendation=filters["only_with_recommendation"],
    )

    st.caption(
        f"Loaded **{len(all_events)}** events from `{events_path}` "
        f"(**{len(filtered)}** after filters). Sidebar **Refresh** reloads the log. "
        f"· 已加载 {len(all_events)} 条，筛选后 {len(filtered)} 条。"
    )

    _render_demo_command_center(events_path)
    _render_injection_banner()

    tabs = st.tabs(
        [
            tab_label("📊 Node overview", "节点总览"),
            tab_label("📜 Event timeline", "时间线"),
            tab_label("🔍 Event detail", "详情"),
            tab_label("📈 Global stats", "统计"),
        ]
    )
    with tabs[0]:
        _render_node_overview(all_events)
    with tabs[1]:
        _render_timeline(filtered, all_events)
    with tabs[2]:
        _render_event_detail(filtered)
    with tabs[3]:
        _render_global_stats(all_events)


if __name__ == "__main__":
    main()
