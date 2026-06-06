"""Run the Component 5 benchmark matrix (scenarios × ablations × optional judge)."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from agent_eval.heuristic_rubric import HeuristicScore, score_heuristic
from agent_eval.llm_judge import LlmJudgeScores, maybe_judge_sync
from agent_eval.scenarios import BenchmarkScenario, load_benchmark_json
from agent_eval.wiring import build_benchmark_agent
from api.schemas import Recommendation
from configs.settings import get_settings
from utils.logging_config import get_logger
from utils.paths import PROJECT_ROOT, ensure_dir

log = get_logger(__name__)

RESULTS_DIR = PROJECT_ROOT / "agent_eval" / "results"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "agent_eval_last_run.md"

Ablations = Literal[
    "full",
    "no_retrieve_knowledge",
    "no_system_history",
    "no_estimate_rul",
    "no_escalate_alert",
    "no_reasoning_trace",
]

ABLATION_DISABLED_MAP: dict[Ablations, frozenset[str]] = {
    "full": frozenset(),
    "no_retrieve_knowledge": frozenset({"retrieve_knowledge"}),
    "no_system_history": frozenset({"system_history"}),
    "no_estimate_rul": frozenset({"estimate_rul"}),
    "no_escalate_alert": frozenset({"escalate_alert"}),
    "no_reasoning_trace": frozenset(),
}


@dataclass
class ScenarioRunRecord:
    scenario_id: str
    ablation: Ablations
    heuristic: HeuristicScore
    llm_judge: LlmJudgeScores | None = None
    judge_skip_reason: str | None = None
    recommendation: dict[str, Any] = field(default_factory=dict)
    n_tool_results: int = 0
    n_reasoning_steps: int = 0


@dataclass
class BenchmarkRunSummary:
    """Aggregate statistics for one benchmark invocation."""

    n_scenarios: int
    ablations: tuple[Ablations, ...]
    mean_heuristic: float
    records: list[ScenarioRunRecord]
    llm_judge_n_scored: int
    llm_judge_mean: float | None
    llm_backend: str = "mock"

    def to_json(self) -> dict[str, Any]:
        return {
            "n_scenarios": self.n_scenarios,
            "ablations": list(self.ablations),
            "llm_backend": self.llm_backend,
            "mean_heuristic": round(self.mean_heuristic, 4),
            "llm_judge_n_scored": self.llm_judge_n_scored,
            "llm_judge_mean": self.llm_judge_mean,
            "records": [
                {
                    "scenario_id": r.scenario_id,
                    "ablation": r.ablation,
                    "heuristic": r.heuristic.to_dict(),
                    "llm_judge": r.llm_judge.to_dict() if r.llm_judge else None,
                    "judge_skip_reason": r.judge_skip_reason,
                    "n_tool_results": r.n_tool_results,
                    "n_reasoning_steps": r.n_reasoning_steps,
                    "recommendation_snapshot": _recommendation_snapshot(r.recommendation),
                }
                for r in self.records
            ],
        }


def _recommendation_snapshot(rec: dict[str, Any]) -> dict[str, Any]:
    """Return a small slice of the Recommendation for downstream reporting.

    Full recommendations carry knowledge_sources + reasoning_trace which
    blow up the summary JSON. The robustness / agent_eval reports only
    need the headline action, urgency, agent confidence, and the count of
    cited sources.
    """

    if not rec:
        return {}
    action = str(rec.get("recommended_action") or "").strip()
    return {
        "recommended_action": action[:600] + ("…" if len(action) > 600 else ""),
        "urgency": rec.get("urgency"),
        "agent_confidence": rec.get("agent_confidence"),
        "n_knowledge_sources": len(rec.get("knowledge_sources") or []),
    }


async def _run_one(
    agent,
    scenario: BenchmarkScenario,
    ablation: Ablations,
    *,
    use_llm_judge: bool,
) -> ScenarioRunRecord:
    strip_trace = ablation == "no_reasoning_trace"
    rec: Recommendation = await agent.run(scenario.alert, strip_reasoning_trace=strip_trace)
    heur = score_heuristic(rec, scenario.expected)
    llm_scores: LlmJudgeScores | None = None
    skip: str | None = None
    if use_llm_judge:
        llm_scores, skip = maybe_judge_sync(scenario, rec)
    else:
        skip = "LLM judge disabled via CLI flag"

    n_tools = sum(
        1
        for step in rec.reasoning_trace
        if step.phase == "act"
        and (step.result_summary or "")
        and "skipped" not in (step.result_summary or "").lower()
    )
    return ScenarioRunRecord(
        scenario_id=scenario.id,
        ablation=ablation,
        heuristic=heur,
        llm_judge=llm_scores,
        judge_skip_reason=skip if llm_scores is None else None,
        recommendation=rec.model_dump(mode="json"),
        n_tool_results=n_tools,
        n_reasoning_steps=len(rec.reasoning_trace),
    )


async def run_benchmark(
    *,
    benchmark_path: Path | None = None,
    knowledge_base_dir: Path | None = None,
    ablations: tuple[Ablations, ...] = ("full",),
    use_llm_judge: bool = True,
    llm_backend: str | None = None,
) -> BenchmarkRunSummary:
    """Execute every scenario for each ablation mode.

    ``llm_backend`` falls back to ``settings.llm_backend`` when ``None`` so
    callers (CLI, tests) automatically pick up ``ollama`` in dev / prod
    deployments while staying ``mock`` in CI.
    """

    settings = get_settings()
    kb = knowledge_base_dir or settings.knowledge_base_dir
    if not kb.exists():
        raise FileNotFoundError(
            f"knowledge_base_dir missing: {kb}; agent_eval requires the same KB as agent_service"
        )

    effective_backend = (llm_backend or settings.llm_backend).strip().lower()
    log.info(
        "benchmark_start",
        extra={
            "ablations": list(ablations),
            "use_llm_judge": use_llm_judge,
            "llm_backend": effective_backend,
        },
    )

    scenarios = load_benchmark_json(benchmark_path)
    records: list[ScenarioRunRecord] = []
    for ablation in ablations:
        disabled = ABLATION_DISABLED_MAP[ablation]
        agent = build_benchmark_agent(
            knowledge_base_dir=kb,
            llm_backend=effective_backend,
            disabled_tools=disabled,
        )
        for sc in scenarios:
            records.append(await _run_one(agent, sc, ablation, use_llm_judge=use_llm_judge))

    mean_h = sum(r.heuristic.score for r in records) / max(len(records), 1)
    scored = [r for r in records if r.llm_judge is not None]
    llm_mean = (
        sum(r.llm_judge.mean() for r in scored) / len(scored) if scored else None  # type: ignore[union-attr]
    )
    log.info(
        "benchmark_done",
        extra={
            "n_records": len(records),
            "mean_heuristic": round(mean_h, 4),
            "llm_judge_n": len(scored),
        },
    )
    return BenchmarkRunSummary(
        n_scenarios=len(scenarios),
        ablations=ablations,
        mean_heuristic=mean_h,
        records=records,
        llm_judge_n_scored=len(scored),
        llm_judge_mean=llm_mean,
        llm_backend=effective_backend,
    )


def write_run_artifacts(summary: BenchmarkRunSummary, *, json_path: Path, md_path: Path) -> None:
    """Persist JSON + a short Markdown interpretation."""

    ensure_dir(json_path.parent)
    json_path.write_text(json.dumps(summary.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# AgentPV — Component 5 agent benchmark (last run)",
        "",
        f"- **Scenarios per ablation**: {summary.n_scenarios}",
        f"- **Ablations**: `{', '.join(summary.ablations)}`",
        f"- **Mean heuristic score (0–1)**: {summary.mean_heuristic:.4f}",
        f"- **LLM-as-judge scored rows**: {summary.llm_judge_n_scored} / {len(summary.records)}",
    ]
    if summary.llm_judge_mean is not None:
        lines.append(f"- **Mean LLM judge (1–5)**: {summary.llm_judge_mean:.3f}")
    lines.extend(["", "## Per-row heuristic failures (score < 1.0)", ""])
    bad = [r for r in summary.records if r.heuristic.score < 1.0]
    if not bad:
        lines.append("_None — all oracle checks passed._")
    else:
        for r in bad[:50]:
            h = r.heuristic
            lines.append(
                f"- `{r.scenario_id}` / `{r.ablation}` → score={h.score:.2f} "
                f"(urgency_ok={h.urgency_ok}, keywords_ok={h.keywords_ok}, "
                f"forbidden_ok={h.forbidden_ok}, knowledge_ok={h.knowledge_ok})"
            )
        if len(bad) > 50:
            lines.append(f"- … _{len(bad) - 50} more omitted_")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark_cli_main(argv: list[str] | None = None) -> None:
    """Blocking entry for ``python -m agent_eval``."""

    import argparse

    parser = argparse.ArgumentParser(prog="agentpv-agent-eval")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=None,
        help="Path to benchmark.json (default: agent_eval/benchmark.json if present).",
    )
    parser.add_argument(
        "--kb-dir",
        type=Path,
        default=None,
        help="Knowledge base directory (default: settings.knowledge_base_dir).",
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        choices=list(ABLATION_DISABLED_MAP.keys()),
        default=["full"],
        help="One or more ablation modes to run (default: full only).",
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Skip remote LLM-as-judge (heuristic-only, fast CI mode).",
    )
    parser.add_argument(
        "--llm-backend",
        choices=("mock", "ollama"),
        default=None,
        help=(
            "Override the agent's LLM backend (defaults to settings.llm_backend, "
            "i.e. dev.yaml + AGENTPV_LLM_BACKEND env var)."
        ),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=RESULTS_DIR / "last_run.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )
    parser.add_argument(
        "--write-default-benchmark",
        action="store_true",
        help="Write built-in ≥30 scenarios to agent_eval/benchmark.json and exit.",
    )
    args = parser.parse_args(argv)

    if args.write_default_benchmark:
        from agent_eval.scenarios import write_default_benchmark_json

        p = write_default_benchmark_json()
        print(json.dumps({"written": str(p)}, indent=2))
        return

    ablations_tuple = tuple(args.ablations)
    summary = asyncio.run(
        run_benchmark(
            benchmark_path=args.benchmark,
            knowledge_base_dir=args.kb_dir,
            ablations=ablations_tuple,
            use_llm_judge=not args.no_llm_judge,
            llm_backend=args.llm_backend,
        )
    )
    write_run_artifacts(summary, json_path=args.out_json, md_path=args.out_md)
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md)}, indent=2))
