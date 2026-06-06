"""Unit tests for the dashboard's pure data layer.

These tests intentionally do **not** import Streamlit so the data layer
can be exercised in isolation (rule §27 — minimal viable system; UI in a
separate testable layer).
"""
from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from api.schemas import (
    AgentConfidence,
    Alert,
    OrchestratorEvent,
    ReasoningStep,
    Recommendation,
    Severity,
    SystemType,
    Urgency,
)
from dashboard.data import (
    LoadResult,
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

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _alert(*, severity: str, fault_class: str, system_id: str = "sys") -> Alert:
    return Alert(
        timestamp=datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC),
        system_id=system_id,
        system_type=SystemType.PV,
        fault_class=fault_class,
        severity=Severity(severity),
        confidence=0.9,
        sensor_snapshot={"P_dc": 200.0},
    )


def _recommendation(*, n_sources: int = 1) -> Recommendation:
    return Recommendation(
        recommended_action="[MOCK] inspect within 4h",
        urgency=Urgency.IMMEDIATE,
        reasoning_trace=[
            ReasoningStep(step=0, phase="observe", thought="ok"),
            ReasoningStep(step=1, phase="report", thought="done"),
        ],
        knowledge_sources=[f"doc-{i}" for i in range(n_sources)],
        confidence=AgentConfidence.HIGH if n_sources else AgentConfidence.LOW,
    )


def _event(
    *,
    event_id: str,
    node_id: str = "n1",
    step: int = 0,
    when: datetime | None = None,
    label: str = "PV_Normal",
    severity: str = "monitor",
    fault_class: str = "PV_Normal",
    with_recommendation: bool = False,
    error: str | None = None,
    edge_ms: float | None = 1.5,
    agent_ms: float | None = None,
    skip_alert: bool = False,
) -> OrchestratorEvent:
    when = when or datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC)
    alert = None if skip_alert else _alert(
        severity=severity, fault_class=fault_class, system_id=f"{node_id}-S"
    )
    return OrchestratorEvent(
        event_id=event_id,
        timestamp=when,
        node_id=node_id,
        system_id=f"{node_id}-S",
        system_type=SystemType.PV,
        step_number=step,
        ground_truth_label=label,
        alert=alert,
        recommendation=_recommendation() if with_recommendation else None,
        error=error,
        edge_elapsed_ms=edge_ms,
        agent_elapsed_ms=agent_ms,
    )


# ---------------------------------------------------------------------------
# load_events
# ---------------------------------------------------------------------------


def test_load_events_missing_file_returns_empty(tmp_path) -> None:
    out = load_events(tmp_path / "nope.jsonl")
    assert isinstance(out, LoadResult)
    assert out.events == []
    assert out.n_skipped == 0


def test_load_events_round_trips_valid_jsonl(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    e1 = _event(event_id="a")
    e2 = _event(event_id="b", step=1)
    path.write_text(
        e1.model_dump_json() + "\n" + e2.model_dump_json() + "\n",
        encoding="utf-8",
    )
    out = load_events(path)
    assert out.n_skipped == 0
    assert [e.event_id for e in out.events] == ["a", "b"]


def test_load_events_skips_invalid_lines_without_aborting(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    valid = _event(event_id="ok").model_dump_json()
    path.write_text(
        f"{valid}\n"
        "not json at all\n"
        "{}\n"  # JSON but missing required fields
        f"{valid}\n",
        encoding="utf-8",
    )
    out = load_events(path)
    assert len(out.events) == 2
    assert out.n_skipped == 2
    assert any("validation" in r.lower() or "json" in r.lower() for r in out.skipped_reasons)


def test_load_events_ignores_blank_lines(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n" + _event(event_id="x").model_dump_json() + "\n\n",
        encoding="utf-8",
    )
    out = load_events(path)
    assert len(out.events) == 1
    assert out.n_skipped == 0


# ---------------------------------------------------------------------------
# events_to_dataframe
# ---------------------------------------------------------------------------


def test_events_to_dataframe_empty_returns_empty_with_columns() -> None:
    df = events_to_dataframe([])
    assert len(df) == 0
    assert "event_id" in df.columns
    assert "n_knowledge_sources" in df.columns


def test_events_to_dataframe_flattens_nested_alert_and_recommendation() -> None:
    events = [
        _event(
            event_id="a",
            severity="critical",
            fault_class="Inverter_fault",
            with_recommendation=True,
            agent_ms=42.0,
        ),
        _event(event_id="b"),
    ]
    df = events_to_dataframe(events)
    assert len(df) == 2
    row_a = df.iloc[0]
    assert row_a["predicted_class"] == "Inverter_fault"
    assert row_a["severity"] == "critical"
    assert row_a["urgency"] == "immediate"
    assert row_a["agent_confidence"] == "high"
    assert row_a["n_knowledge_sources"] == 1
    assert row_a["has_recommendation"]
    assert row_a["agent_elapsed_ms"] == 42.0
    row_b = df.iloc[1]
    assert row_b["urgency"] is None
    assert not row_b["has_recommendation"]


def test_events_to_dataframe_handles_error_row() -> None:
    events = [_event(event_id="e", skip_alert=True, error="edge_predict_failed: 503")]
    df = events_to_dataframe(events)
    assert pd.isna(df.iloc[0]["severity"]) or df.iloc[0]["severity"] is None
    assert df.iloc[0]["error"].startswith("edge_predict_failed")


# ---------------------------------------------------------------------------
# per_node_summary
# ---------------------------------------------------------------------------


def test_per_node_summary_counts_steps_alerts_recommendations() -> None:
    base = datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC)
    events = [
        _event(event_id="1", node_id="n1", step=0, when=base, severity="monitor"),
        _event(
            event_id="2",
            node_id="n1",
            step=1,
            when=base + timedelta(seconds=1),
            severity="warning",
            with_recommendation=True,
        ),
        _event(
            event_id="3", node_id="n2", step=0, when=base, severity="critical"
        ),
        _event(
            event_id="4",
            node_id="n2",
            step=1,
            when=base + timedelta(seconds=2),
            skip_alert=True,
            error="boom",
        ),
    ]
    df = per_node_summary(events)
    assert list(df["node_id"]) == ["n1", "n2"]
    n1 = df[df["node_id"] == "n1"].iloc[0]
    n2 = df[df["node_id"] == "n2"].iloc[0]
    assert n1["n_steps"] == 2
    assert n1["n_alerts"] == 2
    assert n1["n_recommendations"] == 1
    assert n1["last_severity"] == "warning"
    assert n2["n_errors"] == 1
    assert n2["n_alerts"] == 1


def test_per_node_summary_empty_returns_empty_with_columns() -> None:
    df = per_node_summary([])
    assert len(df) == 0
    assert "n_alerts" in df.columns


# ---------------------------------------------------------------------------
# severity_counts / fault_class_counts
# ---------------------------------------------------------------------------


def test_severity_counts_includes_zero_rows() -> None:
    events = [_event(event_id="1", severity="warning")]
    df = severity_counts(events)
    assert set(df["severity"]) == {"monitor", "warning", "critical"}
    assert df[df["severity"] == "warning"]["count"].iloc[0] == 1
    assert df[df["severity"] == "critical"]["count"].iloc[0] == 0


def test_fault_class_counts_sorted_descending() -> None:
    events = [
        _event(event_id=str(i), fault_class="A", severity="warning") for i in range(3)
    ] + [_event(event_id="x", fault_class="B", severity="warning")]
    df = fault_class_counts(events)
    assert df.iloc[0]["fault_class"] == "A"
    assert df.iloc[0]["count"] == 3
    assert df.iloc[1]["fault_class"] == "B"


# ---------------------------------------------------------------------------
# latency_stats
# ---------------------------------------------------------------------------


def test_latency_stats_only_edge() -> None:
    events = [
        _event(event_id=str(i), edge_ms=float(i + 1), agent_ms=None)
        for i in range(5)
    ]
    s = latency_stats(events)
    assert s["n_edge"] == 5
    assert s["n_agent"] == 0
    assert s["edge_mean_ms"] == 3.0  # mean of [1,2,3,4,5]
    assert math.isnan(s["agent_mean_ms"])


def test_latency_stats_no_events_all_nan() -> None:
    s = latency_stats([])
    assert s["n_edge"] == 0
    assert math.isnan(s["edge_mean_ms"])
    assert math.isnan(s["agent_mean_ms"])


# ---------------------------------------------------------------------------
# get_event_by_id / filter_events
# ---------------------------------------------------------------------------


def test_get_event_by_id_found_and_missing() -> None:
    events = [_event(event_id="a"), _event(event_id="b")]
    assert get_event_by_id(events, "b") is events[1]
    assert get_event_by_id(events, "missing") is None


def test_filter_events_by_severity_only() -> None:
    events = [
        _event(event_id="1", severity="monitor"),
        _event(event_id="2", severity="warning"),
        _event(event_id="3", severity="critical"),
    ]
    out = filter_events(events, severities=[Severity.WARNING, Severity.CRITICAL])
    assert {e.event_id for e in out} == {"2", "3"}


def test_filter_events_by_node() -> None:
    events = [
        _event(event_id="a", node_id="n1"),
        _event(event_id="b", node_id="n2"),
    ]
    out = filter_events(events, node_ids=["n2"])
    assert [e.event_id for e in out] == ["b"]


def test_filter_events_only_with_recommendation() -> None:
    events = [
        _event(event_id="a", with_recommendation=True),
        _event(event_id="b"),
    ]
    out = filter_events(events, only_with_recommendation=True)
    assert [e.event_id for e in out] == ["a"]


def test_filter_events_combined() -> None:
    events = [
        _event(event_id="a", node_id="n1", severity="warning"),
        _event(event_id="b", node_id="n2", severity="warning"),
        _event(event_id="c", node_id="n1", severity="monitor"),
    ]
    out = filter_events(
        events, node_ids=["n1"], severities=[Severity.WARNING]
    )
    assert [e.event_id for e in out] == ["a"]


# ---------------------------------------------------------------------------
# severity_over_time
# ---------------------------------------------------------------------------


def test_severity_over_time_buckets_by_seconds() -> None:
    base = datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC)
    events = [
        _event(event_id="1", when=base, severity="warning"),
        _event(event_id="2", when=base + timedelta(seconds=2), severity="warning"),
        _event(
            event_id="3", when=base + timedelta(seconds=10), severity="critical"
        ),
    ]
    df = severity_over_time(events, bucket_seconds=5.0)
    assert len(df) >= 2  # 至少两个桶
    # 第一个 5 秒桶里有两条 warning
    bucket0 = df[df["severity"] == "warning"]["count"].sum()
    assert bucket0 == 2


def test_severity_over_time_empty_returns_empty_with_columns() -> None:
    df = severity_over_time([], bucket_seconds=5.0)
    assert list(df.columns) == ["bucket", "severity", "count"]
    assert len(df) == 0


def test_severity_over_time_skips_error_only_events() -> None:
    base = datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC)
    events = [_event(event_id="x", when=base, skip_alert=True, error="boom")]
    df = severity_over_time(events, bucket_seconds=5.0)
    assert len(df) == 0


# ---------------------------------------------------------------------------
# Round-trip via JSONL on disk (sanity)
# ---------------------------------------------------------------------------


def test_round_trip_load_then_aggregate(tmp_path) -> None:
    """Load real-shaped JSONL → aggregations don't error."""

    path = tmp_path / "events.jsonl"
    events = [
        _event(event_id="1", severity="critical", with_recommendation=True),
        _event(event_id="2", severity="monitor"),
    ]
    with path.open("w", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(json.loads(e.model_dump_json())) + "\n")
    loaded = load_events(path)
    assert loaded.n_skipped == 0
    df = events_to_dataframe(loaded.events)
    assert len(df) == 2
    sev = severity_counts(loaded.events)
    assert sev[sev["severity"] == "critical"]["count"].iloc[0] == 1


# ---------------------------------------------------------------------------
# Streamlit app importability (catches import-time syntax / wiring errors
# without spinning up the UI process)
# ---------------------------------------------------------------------------


def test_streamlit_app_module_imports_clean() -> None:
    """Importing dashboard.app must not raise; main() must be callable."""

    pytest.importorskip("streamlit")
    import importlib

    mod = importlib.import_module("dashboard.app")
    assert callable(getattr(mod, "main", None))
