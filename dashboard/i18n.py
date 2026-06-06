"""Bilingual UI copy for the Streamlit dashboard (EN primary, ZH secondary)."""

from __future__ import annotations


def zh_sub(zh: str) -> str:
    """Muted Chinese subtitle line (HTML), for use under English headings."""
    if not zh:
        return ""
    return f'<span class="agentpv-zh-sub">{zh}</span>'


def en_zh_html(en: str, zh: str = "") -> str:
    """English line + optional Chinese subtitle (markdown/HTML safe block)."""
    if not zh:
        return f"**{en}**"
    return f"**{en}**<br>{zh_sub(zh)}"


def tab_label(en: str, zh: str) -> str:
    return f"{en} · {zh}"


def button_help(en: str, zh: str) -> str:
    return f"{en} — {zh}"


# Demo preset copy (EN on button, ZH in help)
PRESET_COPY: dict[str, tuple[str, str, str, str]] = {
    "pv_inv": (
        "PV Inverter Fault",
        "PV 逆变器故障",
        "critical · full pipeline",
        "严重 · 全链路",
    ),
    "pv_shade": (
        "PV Partial Shading",
        "PV 局部遮挡",
        "warning · full pipeline",
        "警告 · 全链路",
    ),
    "bess_thermal": (
        "BESS Thermal Anomaly",
        "BESS 热异常",
        "critical · full pipeline",
        "严重 · 全链路",
    ),
    "pv_normal": (
        "PV Normal",
        "PV 正常",
        "monitor · agent skipped",
        "监视 · 不触发 Agent",
    ),
    "edge_only": (
        "Edge-only Degrade",
        "仅边缘降级",
        "critical · skip agent",
        "严重 · 跳过 Agent",
    ),
}
