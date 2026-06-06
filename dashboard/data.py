"""Pure data layer for the dashboard (no Streamlit imports).

This module is intentionally **UI-framework-agnostic** so it can be
unit-tested without spinning up Streamlit. All Streamlit code lives in
:mod:`dashboard.app`.

The data layer reads ``OrchestratorEvent`` JSONL written by the
orchestrator (Component 7) and turns it into typed structures + flat
dataframes ready for rendering. The contract on the JSONL is owned by
:class:`api.schemas.OrchestratorEvent` (rule §3 — single source of truth).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from api.schemas import OrchestratorEvent, Severity, SystemType
from utils.logging_config import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadResult:
    """Result of loading a JSONL event stream.

    Bad lines are recorded but never abort the load — operators must still
    see whatever the dashboard *can* render even if the orchestrator wrote
    a single corrupt entry.
    """

    events: list[OrchestratorEvent]
    n_skipped: int
    skipped_reasons: list[str]


def load_events(path: Path) -> LoadResult:
    """Load events from a JSONL path. Missing file → empty result (not error)."""

    events: list[OrchestratorEvent] = []
    skipped_reasons: list[str] = []
    if not path.exists():
        log.info("dashboard_event_log_missing", extra={"path": str(path)})
        return LoadResult(events=events, n_skipped=0, skipped_reasons=skipped_reasons)
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                events.append(OrchestratorEvent.model_validate_json(stripped))
            except ValidationError as exc:
                skipped_reasons.append(f"line {lineno}: validation error — {exc.errors()[0].get('msg', 'unknown')}")
            except ValueError as exc:
                skipped_reasons.append(f"line {lineno}: {exc}")
    if skipped_reasons:
        log.warning(
            "dashboard_event_log_partial_load",
            extra={"path": str(path), "n_skipped": len(skipped_reasons)},
        )
    return LoadResult(
        events=events, n_skipped=len(skipped_reasons), skipped_reasons=skipped_reasons
    )


# ---------------------------------------------------------------------------
# Aggregations (returned as plain dicts / dataframes for the UI layer)
# ---------------------------------------------------------------------------


def events_to_dataframe(events: list[OrchestratorEvent]) -> pd.DataFrame:
    """Flatten events into a tabular DataFrame keyed on event_id."""

    if not events:
        return pd.DataFrame(
            columns=[
                "event_id",
                "timestamp",
                "node_id",
                "system_id",
                "system_type",
                "step_number",
                "ground_truth_label",
                "predicted_class",
                "severity",
                "confidence",
                "has_recommendation",
                "urgency",
                "agent_confidence",
                "n_knowledge_sources",
                "edge_elapsed_ms",
                "agent_elapsed_ms",
                "error",
            ]
        )
    rows = []
    for e in events:
        row: dict[str, Any] = {
            "event_id": e.event_id,
            "timestamp": e.timestamp,
            "node_id": e.node_id,
            "system_id": e.system_id,
            "system_type": e.system_type.value,
            "step_number": e.step_number,
            "ground_truth_label": e.ground_truth_label,
            "predicted_class": e.alert.fault_class if e.alert else None,
            "severity": e.alert.severity.value if e.alert else None,
            "confidence": e.alert.confidence if e.alert else None,
            "has_recommendation": e.recommendation is not None,
            "urgency": e.recommendation.urgency.value if e.recommendation else None,
            "agent_confidence": (
                e.recommendation.confidence.value if e.recommendation else None
            ),
            "n_knowledge_sources": (
                len(e.recommendation.knowledge_sources) if e.recommendation else 0
            ),
            "edge_elapsed_ms": e.edge_elapsed_ms,
            "agent_elapsed_ms": e.agent_elapsed_ms,
            "error": e.error,
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Pandas represents missing values as NaN/NA. For optional string fields
    # that the UI layer treats as "unset", keep Python None so unit tests can
    # assert `is None` without depending on pandas' NaN semantics.
    for col in ("urgency", "agent_confidence"):
        if col in df.columns:
            df[col] = df[col].where(pd.notna(df[col]), None)
    return df


def per_node_summary(events: list[OrchestratorEvent]) -> pd.DataFrame:
    """One row per node summarising counts + last seen alert."""

    if not events:
        return pd.DataFrame(
            columns=[
                "node_id",
                "system_id",
                "system_type",
                "n_steps",
                "n_alerts",
                "n_recommendations",
                "n_errors",
                "last_severity",
                "last_fault_class",
                "last_seen",
            ]
        )
    by_node: dict[str, dict[str, Any]] = {}
    for e in events:
        b = by_node.setdefault(
            e.node_id,
            {
                "node_id": e.node_id,
                "system_id": e.system_id,
                "system_type": e.system_type.value,
                "n_steps": 0,
                "n_alerts": 0,
                "n_recommendations": 0,
                "n_errors": 0,
                "last_severity": None,
                "last_fault_class": None,
                "last_seen": None,
            },
        )
        b["n_steps"] += 1
        if e.alert is not None:
            b["n_alerts"] += 1
            # 取时间最大的那条
            if b["last_seen"] is None or e.timestamp > b["last_seen"]:
                b["last_severity"] = e.alert.severity.value
                b["last_fault_class"] = e.alert.fault_class
                b["last_seen"] = e.timestamp
        else:
            if b["last_seen"] is None or e.timestamp > b["last_seen"]:
                b["last_seen"] = e.timestamp
        if e.recommendation is not None:
            b["n_recommendations"] += 1
        if e.error is not None:
            b["n_errors"] += 1
    df = pd.DataFrame(list(by_node.values()))
    df = df.sort_values("node_id").reset_index(drop=True)
    return df


def severity_counts(events: list[OrchestratorEvent]) -> pd.DataFrame:
    """Counts per severity, including a 0-row for missing severities.

    Returning all three rows even when zero keeps the bar chart's x-axis
    stable across refreshes.
    """

    counts = {sev.value: 0 for sev in Severity}
    for e in events:
        if e.alert is not None:
            counts[e.alert.severity.value] += 1
    return pd.DataFrame(
        [{"severity": k, "count": v} for k, v in counts.items()]
    )


def fault_class_counts(events: list[OrchestratorEvent]) -> pd.DataFrame:
    """Counts per predicted fault class (one row per class seen)."""

    counts: dict[str, int] = {}
    for e in events:
        if e.alert is None:
            continue
        counts[e.alert.fault_class] = counts.get(e.alert.fault_class, 0) + 1
    rows = sorted(
        ({"fault_class": k, "count": v} for k, v in counts.items()),
        key=lambda r: -r["count"],
    )
    return pd.DataFrame(rows, columns=["fault_class", "count"])


def latency_stats(events: list[OrchestratorEvent]) -> dict[str, float]:
    """Return mean / p50 / p95 latency for edge and agent calls.

    Missing samples (e.g. agent never called) yield ``float("nan")``.
    """

    edge_ms = [e.edge_elapsed_ms for e in events if e.edge_elapsed_ms is not None]
    agent_ms = [e.agent_elapsed_ms for e in events if e.agent_elapsed_ms is not None]
    return {
        "n_edge": len(edge_ms),
        "n_agent": len(agent_ms),
        "edge_mean_ms": float(pd.Series(edge_ms).mean()) if edge_ms else float("nan"),
        "edge_p50_ms": float(pd.Series(edge_ms).quantile(0.50)) if edge_ms else float("nan"),
        "edge_p95_ms": float(pd.Series(edge_ms).quantile(0.95)) if edge_ms else float("nan"),
        "agent_mean_ms": float(pd.Series(agent_ms).mean()) if agent_ms else float("nan"),
        "agent_p50_ms": float(pd.Series(agent_ms).quantile(0.50)) if agent_ms else float("nan"),
        "agent_p95_ms": float(pd.Series(agent_ms).quantile(0.95)) if agent_ms else float("nan"),
    }


def get_event_by_id(
    events: list[OrchestratorEvent], event_id: str
) -> OrchestratorEvent | None:
    """Lookup helper used by the event-detail view."""

    for e in events:
        if e.event_id == event_id:
            return e
    return None


def filter_events(
    events: list[OrchestratorEvent],
    *,
    node_ids: list[str] | None = None,
    system_types: list[SystemType] | None = None,
    severities: list[Severity] | None = None,
    only_with_recommendation: bool = False,
) -> list[OrchestratorEvent]:
    """Apply UI-side filters in pure-Python (deterministic / testable)."""

    out: list[OrchestratorEvent] = []
    for e in events:
        if node_ids and e.node_id not in node_ids:
            continue
        if system_types and e.system_type not in system_types:
            continue
        if severities:
            if e.alert is None or e.alert.severity not in severities:
                continue
        if only_with_recommendation and e.recommendation is None:
            continue
        out.append(e)
    return out


# ---------------------------------------------------------------------------
# Time-bucketed history (for the global activity chart)
# ---------------------------------------------------------------------------


def severity_over_time(
    events: list[OrchestratorEvent],
    *,
    bucket_seconds: float = 5.0,
) -> pd.DataFrame:
    """Bucket alerts by ``floor(timestamp, bucket_seconds)`` × severity.

    Returns a long-form DataFrame ready for ``alt.Chart`` or
    ``st.bar_chart`` in stacked mode.
    """

    if not events:
        return pd.DataFrame(columns=["bucket", "severity", "count"])
    rows = []
    for e in events:
        if e.alert is None:
            continue
        ts = pd.Timestamp(e.timestamp)
        bucket = ts.floor(f"{int(max(bucket_seconds, 1))}s")
        rows.append({"bucket": bucket, "severity": e.alert.severity.value})
    if not rows:
        return pd.DataFrame(columns=["bucket", "severity", "count"])
    df = pd.DataFrame(rows)
    grouped = df.groupby(["bucket", "severity"]).size().reset_index(name="count")
    return grouped
