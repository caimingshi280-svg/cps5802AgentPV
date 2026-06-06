"""Unit tests for ``agent_eval`` (Component 5).

All tests are **offline**: no LLM API keys, no network. They validate scenario
counts, heuristic scoring, wiring validation, and a tiny async benchmark run
against the real placeholder knowledge base under ``rag/knowledge_base/documents``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_eval.heuristic_rubric import score_heuristic
from agent_eval.llm_judge import _local_ollama_openai_base
from agent_eval.runner import ABLATION_DISABLED_MAP, run_benchmark
from agent_eval.scenarios import ExpectedOutcome, default_benchmark_scenarios, load_benchmark_json
from agent_eval.wiring import ALL_TOOL_NAMES, build_benchmark_agent
from api.schemas import (
    AgentConfidence,
    ReasoningStep,
    Recommendation,
    Urgency,
)
from configs.settings import get_settings


def test_default_benchmark_has_at_least_30_scenarios_and_5_ambiguous() -> None:
    scenarios = default_benchmark_scenarios()
    assert len(scenarios) >= 30
    ambiguous = sum(1 for s in scenarios if "ambiguous" in s.tags)
    assert ambiguous >= 5


def test_load_benchmark_json_round_trips(tmp_path: Path) -> None:
    scenarios = default_benchmark_scenarios()[:3]
    path = tmp_path / "mini.json"
    path.write_text(
        json.dumps([s.model_dump(mode="json") for s in scenarios], ensure_ascii=False),
        encoding="utf-8",
    )
    loaded = load_benchmark_json(path)
    assert len(loaded) == 3
    assert loaded[0].id == scenarios[0].id


def test_score_heuristic_soft_mock_keyword_accepts_real_llm_style() -> None:
    """Oracle ``[MOCK]`` slot passes when action is operational without the mock tag."""

    rec = Recommendation(
        recommended_action="Verify combiner currents and schedule an IV sweep within 48h.",
        urgency=Urgency.SCHEDULED,
        reasoning_trace=[ReasoningStep(step=0, phase="observe", thought="seen alert")],
        knowledge_sources=["kb"],
        confidence=AgentConfidence.MEDIUM,
    )
    exp = ExpectedOutcome(
        expected_urgency=Urgency.SCHEDULED,
        must_contain_keywords=["[MOCK]", "verify"],
        must_not_contain_keywords=[],
        min_knowledge_sources=1,
    )
    h = score_heuristic(rec, exp)
    assert h.keywords_ok is True
    assert h.score == 1.0


def test_score_heuristic_perfect_match() -> None:
    rec = Recommendation(
        recommended_action="[MOCK] Inspect PV_SITE_001 (PV): 'Soiling' (warning, conf=0.90) within 1 week.",
        urgency=Urgency.SCHEDULED,
        reasoning_trace=[
            ReasoningStep(step=0, phase="observe", thought="seen alert"),
        ],
        knowledge_sources=["placeholder_partial_shading"],
        confidence=AgentConfidence.MEDIUM,
    )
    exp = ExpectedOutcome(
        expected_urgency=Urgency.SCHEDULED,
        must_contain_keywords=["[MOCK]", "Inspect"],
        must_not_contain_keywords=["wait and see"],
        min_knowledge_sources=1,
    )
    h = score_heuristic(rec, exp)
    assert h.score == 1.0


def test_score_heuristic_fails_on_forbidden_phrase() -> None:
    rec = Recommendation(
        recommended_action="[MOCK] wait and see for 48h",
        urgency=Urgency.IMMEDIATE,
        reasoning_trace=[ReasoningStep(step=0, phase="observe", thought="x")],
        knowledge_sources=["a"],
        confidence=AgentConfidence.HIGH,
    )
    exp = ExpectedOutcome(
        expected_urgency=Urgency.IMMEDIATE,
        must_contain_keywords=["[MOCK]"],
        must_not_contain_keywords=["wait and see"],
        min_knowledge_sources=1,
    )
    h = score_heuristic(rec, exp)
    assert h.forbidden_ok is False
    assert h.score < 1.0


def test_build_benchmark_agent_rejects_unknown_ablation_token() -> None:
    kb = get_settings().knowledge_base_dir
    with pytest.raises(ValueError, match="unknown disabled_tools"):
        build_benchmark_agent(
            knowledge_base_dir=kb,
            disabled_tools=frozenset({"not_a_real_tool"}),
        )


def test_all_ablation_tokens_are_subset_of_registry() -> None:
    for _mode, disabled in ABLATION_DISABLED_MAP.items():
        assert disabled <= ALL_TOOL_NAMES


@pytest.mark.asyncio
async def test_run_benchmark_no_reasoning_trace_ablation() -> None:
    """Trace-stripping ablation — kept on the mock backend so this test
    never reaches the network (CI requirement, §6)."""

    kb = get_settings().knowledge_base_dir
    summary = await run_benchmark(
        benchmark_path=None,
        knowledge_base_dir=kb,
        ablations=("no_reasoning_trace",),
        use_llm_judge=False,
        llm_backend="mock",
    )
    stripped = [r for r in summary.records if r.ablation == "no_reasoning_trace"]
    assert stripped
    assert all(r.n_reasoning_steps == 1 for r in stripped)


@pytest.mark.asyncio
async def test_run_benchmark_smoke_one_ablation() -> None:
    """Mock-backend smoke — must produce the deterministic mean=1.0."""

    kb = get_settings().knowledge_base_dir
    summary = await run_benchmark(
        benchmark_path=None,
        knowledge_base_dir=kb,
        ablations=("full",),
        use_llm_judge=False,
        llm_backend="mock",
    )
    assert summary.n_scenarios >= 30
    assert summary.mean_heuristic == pytest.approx(1.0, abs=1e-6)
    assert summary.llm_judge_n_scored == 0
    assert summary.llm_backend == "mock"


def test_local_ollama_judge_base_detection() -> None:
    """LLM-as-judge may run without API key when base is local Ollama OpenAI shim."""

    assert _local_ollama_openai_base("http://127.0.0.1:11434/v1")
    assert _local_ollama_openai_base("http://localhost:11434/v1")
    assert _local_ollama_openai_base("HTTP://LOCALHOST:11434/v1")
    assert not _local_ollama_openai_base("https://api.openai.com/v1")
    assert not _local_ollama_openai_base("http://127.0.0.1:9999/v1")
    assert not _local_ollama_openai_base("http://example.com:11434/v1")
