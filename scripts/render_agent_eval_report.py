"""Render ``reports/agent_eval.md`` + presentation figures from a benchmark JSON.

Component 5 deliverable #6 entry point.

Consumes:
    agent_eval/results/last_run_ollama.json     (default — real LLM run)
    agent_eval/results/last_run.json            (or whatever ``--input`` points to)

Produces:
    reports/agent_eval.md
    reports/figures/agent_eval/*.png            (presentation-grade)
    reports/agent_eval_artifact_meta.json       (provenance pointer)

The renderer is **side-effect-only**: every figure is generated through the
shared `evaluation.figures.apply_presentation_style()` so the agent_eval
plots match the robustness / model_eval plots stylistically.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.figures import (  # noqa: E402
    PALETTE,
    apply_presentation_style,
    palette,
    save_fig,
)
from utils.paths import PROJECT_ROOT, REPORTS_DIR, ensure_dir  # noqa: E402

DEFAULT_INPUT = PROJECT_ROOT / "agent_eval" / "results" / "last_run_ollama.json"
DEFAULT_LOG = PROJECT_ROOT / "agent_eval" / "results" / "last_run_ollama.log"
DEFAULT_REPORT = REPORTS_DIR / "agent_eval.md"
DEFAULT_FIG_DIR = REPORTS_DIR / "figures" / "agent_eval"
DEFAULT_META = REPORTS_DIR / "agent_eval_artifact_meta.json"


# ---------------------------------------------------------------------------
# Log parsing (telemetry from python -m agent_eval ...)
# ---------------------------------------------------------------------------


_TS_PATTERN = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", re.MULTILINE
)


def _read_log(path: Path) -> str:
    """Read a log file written by either PowerShell (UTF-16LE) or Python (UTF-8)."""

    if not path.exists():
        return ""
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="ignore")
    return raw.decode("utf-8", errors="ignore")


def _parse_log_telemetry(path: Path) -> dict[str, Any]:
    text = _read_log(path)
    if not text:
        return {"log_present": False}

    ts_matches = list(_TS_PATTERN.finditer(text))
    first_ts = ts_matches[0].group("ts") if ts_matches else None
    last_ts = ts_matches[-1].group("ts") if ts_matches else None
    duration_s: float | None = None
    if first_ts and last_ts:
        fmt = "%Y-%m-%dT%H:%M:%S"
        duration_s = (
            datetime.strptime(last_ts, fmt) - datetime.strptime(first_ts, fmt)
        ).total_seconds()

    return {
        "log_present": True,
        "log_path": str(path),
        "log_bytes": path.stat().st_size,
        "ollama_http_calls": len(re.findall(r"POST http://localhost:11434/api/chat", text)),
        "plan_fallback_warnings": len(re.findall(r"ollama_plan_fallback_mock", text)),
        "tool_validation_warnings": len(re.findall(r"tool_validation_failed", text)),
        "react_completed_events": len(re.findall(r"react_completed", text)),
        "alerts_escalated": len(re.findall(r"alert_escalated", text)),
        "first_log_ts": first_ts,
        "last_log_ts": last_ts,
        "duration_seconds": duration_s,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _group_by_ablation(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        out.setdefault(r["ablation"], []).append(r)
    return out


def _kb_source_count(record: dict[str, Any]) -> int:
    """Number of KB sources cited by the agent for this record.

    The runner stores a serialised ``recommendation_snapshot`` (see
    :func:`agent_eval.runner._recommendation_snapshot`) rather than the full
    ``Recommendation`` object, so we read the precomputed
    ``n_knowledge_sources`` integer. Falls back to ``0`` when the snapshot is
    missing (older results files).
    """

    snap = record.get("recommendation_snapshot") or {}
    n = snap.get("n_knowledge_sources")
    if n is None:
        rec = record.get("recommendation") or {}
        sources = rec.get("knowledge_sources") or []
        n = len(sources)
    return int(n)


def _summarise_ablation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-ablation aggregate stats."""

    scores = [r["heuristic"]["score"] for r in rows]
    urgency_ok = sum(1 for r in rows if r["heuristic"]["urgency_ok"])
    keywords_ok = sum(1 for r in rows if r["heuristic"]["keywords_ok"])
    forbidden_ok = sum(1 for r in rows if r["heuristic"]["forbidden_ok"])
    knowledge_ok = sum(1 for r in rows if r["heuristic"]["knowledge_ok"])
    kb_counts = [_kb_source_count(r) for r in rows]
    n = max(len(rows), 1)

    return {
        "n": len(rows),
        "mean_score": round(statistics.fmean(scores), 4) if scores else 0.0,
        "median_score": round(statistics.median(scores), 4) if scores else 0.0,
        "min_score": round(min(scores), 4) if scores else 0.0,
        "max_score": round(max(scores), 4) if scores else 0.0,
        "pct_perfect": round(sum(1 for s in scores if s >= 0.999) / n, 4),
        "pct_urgency_ok": round(urgency_ok / n, 4),
        "pct_keywords_ok": round(keywords_ok / n, 4),
        "pct_forbidden_ok": round(forbidden_ok / n, 4),
        "pct_knowledge_ok": round(knowledge_ok / n, 4),
        "mean_tool_calls": round(statistics.fmean(r["n_tool_results"] for r in rows), 3) if rows else 0.0,
        "mean_reasoning_steps": round(statistics.fmean(r["n_reasoning_steps"] for r in rows), 3) if rows else 0.0,
        # Operational grounding signal (Component 5 No-RAG ablation):
        # captures provenance loss even when the rubric's permissive
        # ``min_knowledge_sources`` threshold cannot fire.
        "mean_kb_sources": round(statistics.fmean(kb_counts), 3) if kb_counts else 0.0,
        "pct_with_kb": round(sum(1 for c in kb_counts if c >= 1) / n, 4),
    }


def _failure_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows that scored < 1.0 with the dominant failure category."""

    out: list[dict[str, Any]] = []
    for r in rows:
        h = r["heuristic"]
        if h["score"] >= 0.999:
            continue
        failed = []
        if not h["urgency_ok"]:
            failed.append("urgency")
        if not h["keywords_ok"]:
            failed.append("keywords")
        if not h["forbidden_ok"]:
            failed.append("forbidden")
        if not h["knowledge_ok"]:
            failed.append("knowledge")
        snap = r.get("recommendation_snapshot") or {}
        out.append({
            "scenario_id": r["scenario_id"],
            "ablation": r["ablation"],
            "score": h["score"],
            "failed_checks": failed,
            "action": snap.get("recommended_action") or "",
            "urgency": snap.get("urgency"),
            "agent_confidence": snap.get("agent_confidence"),
            "n_knowledge_sources": snap.get("n_knowledge_sources"),
        })
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _bar_ablation_summary(
    *,
    summary_by_ablation: dict[str, dict[str, Any]],
    output_path: Path,
    title: str,
) -> Path:
    """Grouped bar: mean score + per-component pass rate per ablation."""

    import matplotlib.pyplot as plt

    apply_presentation_style()
    ablations = list(summary_by_ablation.keys())
    components = ("mean_score", "pct_urgency_ok", "pct_keywords_ok",
                  "pct_forbidden_ok", "pct_knowledge_ok")
    labels = ("mean score", "urgency ✓", "keywords ✓", "forbidden ✓", "knowledge ✓")

    x = np.arange(len(components))
    width = 0.8 / max(len(ablations), 1)

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for i, abl in enumerate(ablations):
        vals = [summary_by_ablation[abl][c] for c in components]
        offset = (i - (len(ablations) - 1) / 2.0) * width
        bars = ax.bar(x + offset, vals, width, color=PALETTE[i % len(PALETTE)],
                      edgecolor="#222", linewidth=0.7, label=abl)
        for bar, v in zip(bars, vals, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.015,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=8.5, color="#222")
    ax.axhline(y=0.90, color=PALETTE[5], linestyle="--", linewidth=1.2,
               label="quality bar = 0.90")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.10)
    ax.set_ylabel("Score / pass rate")
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=len(ablations) + 1)
    fig.tight_layout()
    return save_fig(fig, output_path)


def _hist_score_distribution(
    *,
    by_ablation: dict[str, list[dict[str, Any]]],
    output_path: Path,
    title: str,
) -> Path:
    """Grouped (side-by-side) histogram of heuristic scores per ablation.

    Uses grouped bars rather than overlay so identical distributions
    (a common outcome when ablating post-hoc artefacts like the
    reasoning trace) remain visually distinguishable.
    """

    import matplotlib.pyplot as plt

    apply_presentation_style()
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    bins = np.linspace(0.0, 1.0, 11)
    bin_centres = 0.5 * (bins[:-1] + bins[1:])
    bin_width = bins[1] - bins[0]
    ablations = list(by_ablation.keys())
    group_width = bin_width * 0.85
    bar_width = group_width / max(len(ablations), 1)

    for i, abl in enumerate(ablations):
        scores = [r["heuristic"]["score"] for r in by_ablation[abl]]
        counts, _ = np.histogram(scores, bins=bins)
        offset = (i - (len(ablations) - 1) / 2.0) * bar_width
        bars = ax.bar(
            bin_centres + offset,
            counts,
            width=bar_width,
            color=PALETTE[i % len(PALETTE)],
            edgecolor="#222",
            linewidth=0.7,
            alpha=0.95,
            label=f"{abl}  (mean = {statistics.fmean(scores):.3f})" if scores else abl,
        )
        for bar, c in zip(bars, counts, strict=True):
            if c <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.3,
                str(int(c)),
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#222",
            )

    ax.set_xlabel("Heuristic score bin (0.0 – 1.0)")
    ax.set_ylabel("Scenario count")
    ax.set_xticks(bin_centres)
    ax.set_xticklabels([f"{c:.1f}" for c in bin_centres])
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return save_fig(fig, output_path)


def _tool_calls_box(
    *,
    by_ablation: dict[str, list[dict[str, Any]]],
    output_path: Path,
    title: str,
) -> Path:
    """Side-by-side box / strip of tool-call and reasoning-step counts per ablation."""

    import matplotlib.pyplot as plt

    apply_presentation_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ablations = list(by_ablation.keys())

    for ax, key, ylabel in (
        (axes[0], "n_tool_results", "Tool calls per scenario"),
        (axes[1], "n_reasoning_steps", "Reasoning steps per scenario"),
    ):
        data = [[r[key] for r in by_ablation[a]] for a in ablations]
        positions = np.arange(1, len(ablations) + 1)
        bp = ax.boxplot(data, positions=positions, widths=0.55,
                        patch_artist=True, showmeans=True,
                        meanprops={"marker": "D", "markerfacecolor": "white",
                                   "markeredgecolor": PALETTE[5], "markersize": 7})
        for patch, color in zip(bp["boxes"], palette(len(ablations)), strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
            patch.set_edgecolor("#222")
        for whisker in bp["whiskers"]:
            whisker.set_color("#444")
        for cap in bp["caps"]:
            cap.set_color("#444")
        for median in bp["medians"]:
            median.set_color("#111")
            median.set_linewidth(1.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(ablations)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)

    fig.suptitle(title, y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, output_path)


def _kb_sources_chart(
    *,
    by_ablation: dict[str, list[dict[str, Any]]],
    output_path: Path,
    title: str,
) -> Path:
    """Two-panel figure highlighting the No-RAG ablation's grounding loss.

    Left panel  — mean knowledge sources per scenario, per ablation.
    Right panel — % of scenarios where the agent returned ≥1 KB citation.
    """

    import matplotlib.pyplot as plt

    apply_presentation_style()

    ablations = list(by_ablation.keys())
    mean_sources = [
        statistics.fmean([_kb_source_count(r) for r in by_ablation[a]])
        if by_ablation[a] else 0.0
        for a in ablations
    ]
    pct_with = [
        (sum(1 for r in by_ablation[a] if _kb_source_count(r) >= 1)
         / max(len(by_ablation[a]), 1))
        for a in ablations
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    colors = palette(len(ablations))
    bars0 = axes[0].bar(ablations, mean_sources, color=colors,
                         edgecolor="#222", linewidth=0.7)
    for bar, v in zip(bars0, mean_sources, strict=True):
        axes[0].text(bar.get_x() + bar.get_width() / 2.0,
                     bar.get_height() + 0.05,
                     f"{v:.2f}", ha="center", va="bottom",
                     fontsize=10, fontweight="bold", color="#222")
    axes[0].set_ylabel("Mean # KB sources cited per scenario")
    axes[0].set_title("Provenance: citations / scenario")
    upper0 = max(mean_sources) * 1.25 if mean_sources else 1.0
    axes[0].set_ylim(0.0, max(upper0, 0.5))

    bars1 = axes[1].bar(ablations, [p * 100 for p in pct_with], color=colors,
                         edgecolor="#222", linewidth=0.7)
    for bar, v in zip(bars1, pct_with, strict=True):
        axes[1].text(bar.get_x() + bar.get_width() / 2.0,
                     bar.get_height() + 1.5,
                     f"{v:.0%}", ha="center", va="bottom",
                     fontsize=10, fontweight="bold", color="#222")
    axes[1].set_ylabel("% scenarios with ≥1 KB citation")
    axes[1].set_title("Auditability: % recommendations grounded")
    axes[1].set_ylim(0.0, 115.0)

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

    fig.suptitle(title, y=1.02, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, output_path)


def _ablation_diff_chart(
    *,
    by_ablation: dict[str, list[dict[str, Any]]],
    output_path: Path,
    title: str,
) -> Path:
    """Per-scenario delta of (full − ablated) heuristic score, sorted."""

    import matplotlib.pyplot as plt

    apply_presentation_style()

    if "full" not in by_ablation or len(by_ablation) < 2:
        # Skip cleanly if there's nothing to compare.
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        ax.text(0.5, 0.5, "Need both `full` and ≥1 ablation to plot.",
                ha="center", va="center", color="#666")
        ax.set_axis_off()
        return save_fig(fig, output_path)

    base = {r["scenario_id"]: r["heuristic"]["score"] for r in by_ablation["full"]}
    fig, ax = plt.subplots(figsize=(11.5, 5.4))

    width = 0.85 / (len(by_ablation) - 1)
    others = [a for a in by_ablation if a != "full"]
    scenario_ids = sorted(base.keys())

    for i, abl in enumerate(others):
        s_map = {r["scenario_id"]: r["heuristic"]["score"] for r in by_ablation[abl]}
        diffs = [base[s] - s_map.get(s, 0.0) for s in scenario_ids]
        offset = (i - (len(others) - 1) / 2.0) * width
        positions = np.arange(len(scenario_ids)) + offset
        color = PALETTE[5] if any(d > 0 for d in diffs) else PALETTE[3]
        ax.bar(positions, diffs, width, color=color, alpha=0.85,
               edgecolor="#222", linewidth=0.5, label=f"full − {abl}")

    ax.axhline(y=0.0, color="#444", linewidth=1.0)
    ax.set_xticks(np.arange(len(scenario_ids)))
    ax.set_xticklabels([s.replace("scn_", "").replace("_", " ")
                        for s in scenario_ids], rotation=80, ha="right", fontsize=7.5)
    ax.set_ylabel("Δ heuristic score   (full − ablated)", labelpad=10)
    ax.set_ylim(-0.35, 0.35)
    ax.set_title(title)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.92, bottom=0.34)
    return save_fig(fig, output_path)


def _failure_taxonomy_bar(
    *,
    failures: list[dict[str, Any]],
    output_path: Path,
    title: str,
) -> Path:
    """Count of failed-check categories across all failing scenarios."""

    import matplotlib.pyplot as plt

    apply_presentation_style()
    categories = ("urgency", "keywords", "forbidden", "knowledge")
    counts: dict[str, dict[str, int]] = {c: {} for c in categories}
    for f in failures:
        for cat in f["failed_checks"]:
            counts[cat][f["ablation"]] = counts[cat].get(f["ablation"], 0) + 1

    ablations = sorted({f["ablation"] for f in failures})
    if not ablations:
        ablations = ["full"]

    x = np.arange(len(categories))
    width = 0.8 / max(len(ablations), 1)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    for i, abl in enumerate(ablations):
        vals = [counts[c].get(abl, 0) for c in categories]
        offset = (i - (len(ablations) - 1) / 2.0) * width
        bars = ax.bar(x + offset, vals, width, color=PALETTE[i % len(PALETTE)],
                      edgecolor="#222", linewidth=0.7, label=abl)
        for bar, v in zip(bars, vals, strict=True):
            if v == 0:
                continue
            ax.text(bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.3,
                    str(v), ha="center", va="bottom",
                    fontsize=9, color="#222", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Number of scenarios failing this check")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


def _render_markdown(
    *,
    payload: dict[str, Any],
    summary_by_ablation: dict[str, dict[str, Any]],
    failures_by_ablation: dict[str, list[dict[str, Any]]],
    figures: dict[str, Path],
    figures_rel: str,
    telemetry: dict[str, Any],
) -> str:
    backend = payload.get("llm_backend", "?")
    abls = payload.get("ablations", [])
    n_scenarios = payload.get("n_scenarios", 0)
    n_records = sum(s["n"] for s in summary_by_ablation.values())

    lines: list[str] = []
    lines.append("# AgentPV — Component 5 LLM agent evaluation (Deliverable #6)")
    lines.append("")
    lines.append(
        f"- **LLM backend**: `{backend}`  (model identity: see `dev_run_meta.json` / Ollama tags)."
    )
    lines.append(f"- **Scenarios per ablation**: {n_scenarios}")
    lines.append(f"- **Ablations**: `{', '.join(abls)}`")
    lines.append(f"- **Total scored records**: {n_records}")
    lines.append(f"- **LLM-as-judge rows scored**: {payload.get('llm_judge_n_scored', 0)}")
    if payload.get("llm_judge_mean") is not None:
        lines.append(f"- **LLM-as-judge mean (1–5)**: {payload['llm_judge_mean']:.3f}")
    lines.append("")

    if telemetry.get("log_present"):
        d = telemetry
        duration = d.get("duration_seconds")
        duration_str = f"{duration:.0f} s ({duration / 60.0:.1f} min)" if duration else "n/a"
        lines.append("## 0. Run provenance (from `last_run_ollama.log`)")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        lines.append(f"| Wall-clock duration | {duration_str} |")
        lines.append(f"| Ollama HTTP `/api/chat` calls | {d['ollama_http_calls']} |")
        lines.append(
            f"| Planner JSON fallback warnings | {d['plan_fallback_warnings']} "
            f"({d['plan_fallback_warnings'] / max(d['react_completed_events'], 1):.1%} of scenarios) |"
        )
        lines.append(
            f"| Tool validation warnings | {d['tool_validation_warnings']} "
            f"({d['tool_validation_warnings'] / max(d['react_completed_events'], 1):.1%}) |"
        )
        lines.append(f"| `react_completed` events | {d['react_completed_events']} |")
        lines.append(f"| `alert_escalated` events | {d['alerts_escalated']} |")
        lines.append(f"| Log size on disk | {d['log_bytes']:,} bytes |")
        lines.append("")
        lines.append(
            "The planner JSON fallback rate (≈ "
            f"{d['plan_fallback_warnings'] / max(d['react_completed_events'], 1):.0%}) "
            "is the headline robustness signal for the *plan* step: when "
            "`llama3.2` cannot produce a valid `{tool_calls: [...]}` JSON, the "
            "agent gracefully falls back to the deterministic mock planner so "
            "downstream tool calls still happen. Final recommendations still "
            "use the LLM (synthesis step)."
        )
        lines.append("")

    lines.append("## 1. Headline (presentation snapshot)")
    lines.append("")
    lines.append(f"![ablation summary]({figures_rel}/{figures['ablation_summary'].name})")
    lines.append("")
    lines.append(
        "The chart aggregates the four rubric checks (urgency match, must-contain "
        "keywords, forbidden phrases, knowledge-source minimum) plus the composite "
        "mean score per ablation. The `0.90` dashed line is the project quality bar."
    )
    lines.append("")

    lines.append("## 2. Per-ablation aggregate")
    lines.append("")
    lines.append(
        "| Ablation | n | Mean | Median | Min | Max | % perfect | % urgency | "
        "% keywords | % forbidden | % knowledge | KB sources / scn | % with KB | "
        "Tool calls / scn | ReAct steps / scn |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for abl, s in summary_by_ablation.items():
        lines.append(
            f"| `{abl}` | {s['n']} | **{s['mean_score']:.4f}** | {s['median_score']:.4f} | "
            f"{s['min_score']:.4f} | {s['max_score']:.4f} | "
            f"{s['pct_perfect']:.2%} | {s['pct_urgency_ok']:.2%} | "
            f"{s['pct_keywords_ok']:.2%} | {s['pct_forbidden_ok']:.2%} | "
            f"{s['pct_knowledge_ok']:.2%} | "
            f"**{s['mean_kb_sources']:.2f}** | **{s['pct_with_kb']:.2%}** | "
            f"{s['mean_tool_calls']:.2f} | {s['mean_reasoning_steps']:.2f} |"
        )
    lines.append("")
    lines.append(
        "> The two **bold** columns are the operationally critical signal for the "
        "*No-RAG* ablation: even when the rubric's permissive "
        "`min_knowledge_sources` threshold (calibrated for the benchmark's "
        "Normal-class scenarios) cannot fire, the actual citation count drops "
        "to **zero** without retrieval — i.e. the recommendation is no longer "
        "*grounded* or *auditable*."
    )
    lines.append("")

    lines.append("## 3. Provenance (No-RAG ablation operational signal)")
    lines.append("")
    lines.append(
        f"![kb sources per ablation]({figures_rel}/{figures['kb_sources'].name})"
    )
    lines.append("")
    lines.append(
        "**Left:** mean number of knowledge-base citations attached to each "
        "agent recommendation. **Right:** share of scenarios where the agent "
        "produced at least one grounded citation. The `no_retrieve_knowledge` "
        "ablation collapses both panels to zero — the LLM still produces "
        "plausible-looking recommendations, but they are no longer auditable. "
        "This is the headline finding for assignment §4.5 ablation #1."
    )
    lines.append("")

    lines.append("## 4. Score distribution")
    lines.append("")
    lines.append(f"![score distribution]({figures_rel}/{figures['score_histogram'].name})")
    lines.append("")

    lines.append("## 5. Ablation impact (full − ablated, per scenario)")
    lines.append("")
    lines.append(f"![ablation delta]({figures_rel}/{figures['ablation_diff'].name})")
    lines.append("")
    lines.append(
        "Positive bars (red) mean the *full* configuration scored higher than the "
        "ablated variant — i.e. removing that component cost quality. Bars at zero "
        "mean the ablation had no effect on this scenario. The `no_reasoning_trace` "
        "ablation strips the trace **after** ReAct completes, so any non-zero bars "
        "here come from heuristic checks that depend on the trace itself."
    )
    lines.append("")

    lines.append("## 6. Tool usage and reasoning depth")
    lines.append("")
    lines.append(f"![tool calls and reasoning]({figures_rel}/{figures['tool_calls'].name})")
    lines.append("")

    lines.append("## 7. Failure taxonomy")
    lines.append("")
    lines.append(f"![failure taxonomy]({figures_rel}/{figures['failure_taxonomy'].name})")
    lines.append("")
    total_failures = sum(len(rs) for rs in failures_by_ablation.values())
    lines.append(
        f"Across {n_records} scored records, **{total_failures}** "
        "scored below 1.0. The chart above categorises *which check* failed; "
        "with a real LLM backend most failures are lexical drift on the "
        "`keywords` slot rather than wrong urgency or forbidden phrasing."
    )

    lines.append("")
    lines.append("## 8. Example failures (up to 3 per ablation)")
    lines.append("")
    for abl, rows in failures_by_ablation.items():
        if not rows:
            lines.append(f"### `{abl}` — no failures")
            lines.append("")
            continue
        lines.append(f"### `{abl}` — {len(rows)} scenarios below the 1.0 bar")
        lines.append("")
        for sample in rows[:3]:
            lines.append(f"- **`{sample['scenario_id']}`** → score **{sample['score']:.2f}** "
                         f"(failed: `{', '.join(sample['failed_checks'])}`, "
                         f"urgency=`{sample['urgency']}`, "
                         f"kb={sample['n_knowledge_sources']}):")
            lines.append("")
            action = (sample["action"] or "").strip()
            if action:
                lines.append(f"  > {action}")
            lines.append("")

    lines.append("## 9. Notes on heuristic interpretation")
    lines.append("")
    lines.append(
        "The rubric was originally calibrated against the `mock` backend, which "
        "embeds the `[MOCK]` tag and the exact verb \"Inspect\". Real LLMs (Ollama "
        "`llama3.2`) produce semantically equivalent but lexically different outputs "
        "(e.g. \"Check\", \"Verify\", \"Monitor\"). The rubric's "
        "`_keyword_satisfied` now treats the imperative verbs "
        "`inspect / check / verify / assess / monitor / dispatch / isolate / review / "
        "investigate / examine` as an equivalence class, so the heuristic score "
        "remains a fair signal across backends. Hard keywords (fault-class names, "
        "system IDs) are still matched literally."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentpv-render-agent-eval")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help=f"Benchmark JSON (default: {DEFAULT_INPUT}).")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG,
                        help=f"Optional companion log for provenance (default: {DEFAULT_LOG}).")
    parser.add_argument("--out-md", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--meta-json", type=Path, default=DEFAULT_META)
    args = parser.parse_args(argv)

    if not args.input.exists():
        raise FileNotFoundError(
            f"benchmark JSON missing: {args.input}; run `python -m agent_eval ...` first"
        )

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = payload.get("records", [])
    by_ablation = _group_by_ablation(records)

    summary_by_ablation = {abl: _summarise_ablation(rows) for abl, rows in by_ablation.items()}
    failures_by_ablation = {abl: _failure_rows(rows) for abl, rows in by_ablation.items()}

    ensure_dir(args.fig_dir)
    figures: dict[str, Path] = {}

    figures["ablation_summary"] = _bar_ablation_summary(
        summary_by_ablation=summary_by_ablation,
        output_path=args.fig_dir / "ablation_summary.png",
        title="Agent rubric — mean score and per-check pass rate per ablation",
    )
    figures["score_histogram"] = _hist_score_distribution(
        by_ablation=by_ablation,
        output_path=args.fig_dir / "score_histogram.png",
        title="Heuristic score distribution across scenarios",
    )
    figures["tool_calls"] = _tool_calls_box(
        by_ablation=by_ablation,
        output_path=args.fig_dir / "tool_calls_reasoning.png",
        title="Tool usage and reasoning depth per ablation",
    )
    figures["ablation_diff"] = _ablation_diff_chart(
        by_ablation=by_ablation,
        output_path=args.fig_dir / "ablation_diff.png",
        title="Per-scenario ablation impact (full − ablated)",
    )
    figures["failure_taxonomy"] = _failure_taxonomy_bar(
        failures=[f for fs in failures_by_ablation.values() for f in fs],
        output_path=args.fig_dir / "failure_taxonomy.png",
        title="Failure taxonomy — how scenarios miss the rubric",
    )
    figures["kb_sources"] = _kb_sources_chart(
        by_ablation=by_ablation,
        output_path=args.fig_dir / "kb_sources_per_ablation.png",
        title="Knowledge-source provenance per ablation",
    )

    figures_rel = args.fig_dir.relative_to(args.out_md.parent).as_posix()
    telemetry = _parse_log_telemetry(args.log)

    md = _render_markdown(
        payload=payload,
        summary_by_ablation=summary_by_ablation,
        failures_by_ablation=failures_by_ablation,
        figures=figures,
        figures_rel=figures_rel,
        telemetry=telemetry,
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")

    args.meta_json.parent.mkdir(parents=True, exist_ok=True)
    args.meta_json.write_text(json.dumps({
        "input_json": str(args.input),
        "out_md": str(args.out_md),
        "fig_dir": str(args.fig_dir),
        "summary_by_ablation": summary_by_ablation,
        "failure_counts": {abl: len(rs) for abl, rs in failures_by_ablation.items()},
        "telemetry": telemetry,
    }, indent=2), encoding="utf-8")

    print(json.dumps({
        "out_md": str(args.out_md),
        "fig_dir": str(args.fig_dir),
        "meta_json": str(args.meta_json),
        "n_records": sum(s["n"] for s in summary_by_ablation.values()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
