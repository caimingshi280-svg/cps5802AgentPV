"""Component 7 deliverable — interactive fault-injection demo (script form).

Exercises :func:`dashboard.inject.inject_fault_demo` against the running
edge + agent services. Outputs a presentation-ready Markdown summary at
``reports/integration/fault_injection_demo.md`` and a JSON sidecar with
the raw events for traceability.

The Streamlit UI in ``dashboard/app.py`` drives the same code path —
this script exists so the deliverable can be reproduced + included in
the final report without spinning up a browser.

Usage::

    # In two other terminals first:
    python -m uvicorn api.edge_service:app  --host 127.0.0.1 --port 8000
    python -m uvicorn api.agent_service:app --host 127.0.0.1 --port 8001

    # Then:
    python scripts/demo_fault_injection.py
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from api.schemas import OperatingCondition, SystemType  # noqa: E402
from dashboard.inject import InjectionResult, inject_fault_demo  # noqa: E402

DEFAULT_SCENARIOS: tuple[dict, ...] = (
    {
        "label": "PV inverter fault (critical) — full pipeline",
        "system_type": SystemType.PV,
        "fault_class": "Inverter_fault",
        "operating_condition": OperatingCondition.HIGH_IRRADIANCE,
        "system_id": "DEMO-PV-INV-001",
        "seed": 4242,
        "skip_agent": False,
    },
    {
        "label": "PV partial shading (warning) — full pipeline",
        "system_type": SystemType.PV,
        "fault_class": "Partial_shading",
        "operating_condition": OperatingCondition.HIGH_IRRADIANCE,
        "system_id": "DEMO-PV-SHADE-001",
        "seed": 13,
        "skip_agent": False,
    },
    {
        "label": "BESS thermal anomaly (critical) — full pipeline",
        "system_type": SystemType.BESS,
        "fault_class": "Thermal_anomaly",
        "operating_condition": OperatingCondition.HIGH_TEMPERATURE,
        "system_id": "DEMO-BESS-THERMAL-001",
        "seed": 99,
        "skip_agent": False,
    },
    {
        "label": "PV normal — agent skipped (edge-only response)",
        "system_type": SystemType.PV,
        "fault_class": "PV_Normal",
        "operating_condition": OperatingCondition.LOW_IRRADIANCE,
        "system_id": "DEMO-PV-NORMAL-001",
        "seed": 7,
        "skip_agent": False,
    },
    {
        "label": "PV critical fault with skip_agent=True (graceful degradation)",
        "system_type": SystemType.PV,
        "fault_class": "String_disconnection",
        "operating_condition": OperatingCondition.HIGH_IRRADIANCE,
        "system_id": "DEMO-PV-DEGRADE-001",
        "seed": 21,
        "skip_agent": True,
    },
)


def _summarise(result: InjectionResult) -> dict:
    event = result.event
    return {
        "event_id": event.event_id,
        "system_id": event.system_id,
        "ground_truth": event.ground_truth_label,
        "alert_severity": event.alert.severity.value if event.alert else None,
        "alert_fault_class": event.alert.fault_class if event.alert else None,
        "alert_confidence": (
            round(event.alert.confidence, 4) if event.alert else None
        ),
        "recommendation_urgency": (
            event.recommendation.urgency.value if event.recommendation else None
        ),
        "recommendation_confidence": (
            event.recommendation.confidence.value
            if event.recommendation
            else None
        ),
        "recommendation_action": (
            event.recommendation.recommended_action
            if event.recommendation
            else None
        ),
        "knowledge_source_count": (
            len(event.recommendation.knowledge_sources)
            if event.recommendation
            else 0
        ),
        "edge_ms": (
            round(result.edge_ms, 2) if result.edge_ms is not None else None
        ),
        "agent_ms": (
            round(result.agent_ms, 2) if result.agent_ms is not None else None
        ),
        "agent_called": result.agent_called,
        "edge_error": result.edge_error,
        "agent_error": result.agent_error,
        "ok": result.ok,
    }


def _format_markdown(
    scenarios: list[tuple[dict, dict]],
    *,
    events_path: Path,
) -> str:
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%MZ")
    lines: list[str] = []
    lines.append("# Component 7 — Interactive fault-injection demo\n")
    lines.append(
        "> Assignment §4.7 / Deliverable #9. Single-click fault triggers, full "
        "edge → agent pipeline response, persisted to the dashboard's JSONL "
        "feed for live inspection.\n"
    )
    lines.append(f"_Generated_: {now}\n")
    lines.append(
        f"_Events appended to_: `{events_path.as_posix()}` (the dashboard "
        f"picks them up on the next 🔄 Refresh)\n"
    )

    lines.append("## How to drive it interactively\n")
    lines.append(
        "1. `python -m uvicorn api.edge_service:app  --host 127.0.0.1 --port 8000`\n"
        "2. `python -m uvicorn api.agent_service:app --host 127.0.0.1 --port 8001`\n"
        "3. `streamlit run dashboard/app.py`\n"
        "4. Open the sidebar's **🔥 Fault injection (demo)** expander → pick "
        "system / fault / operating condition / seed → click **Inject fault**.\n"
        "5. The success banner appears at the top of the main pane; the new "
        "event also lands in the 'Event timeline' and 'Event detail' tabs.\n"
    )

    lines.append("## Scripted reproduction (this report)\n")
    lines.append("```powershell\npython scripts/demo_fault_injection.py\n```\n")

    lines.append("## Headline outcomes\n")
    lines.append(
        "| # | Scenario | Severity | Urgency | Edge ms | Agent ms | "
        "Knowledge sources | OK |"
    )
    lines.append("| ---: | --- | --- | --- | ---: | ---: | ---: | :---: |")
    for i, (scenario, summary) in enumerate(scenarios, start=1):
        edge_ms = (
            f"{summary['edge_ms']:.2f}" if summary["edge_ms"] is not None else "—"
        )
        agent_ms = (
            f"{summary['agent_ms']:.0f}" if summary["agent_ms"] is not None else "—"
        )
        urgency = summary["recommendation_urgency"] or "—"
        sev = summary["alert_severity"] or "—"
        kn = summary["knowledge_source_count"]
        ok = "✅" if summary["ok"] else "❌"
        lines.append(
            f"| {i} | {scenario['label']} | `{sev}` | `{urgency}` | "
            f"{edge_ms} | {agent_ms} | {kn} | {ok} |"
        )
    lines.append("")

    lines.append("## Per-scenario detail\n")
    for i, (scenario, summary) in enumerate(scenarios, start=1):
        lines.append(f"### {i}. {scenario['label']}\n")
        lines.append(
            f"- **Input**: `system={scenario['system_type'].value}`, "
            f"`fault={scenario['fault_class']}`, "
            f"`op={scenario['operating_condition'].value}`, "
            f"`seed={scenario['seed']}`, "
            f"`skip_agent={scenario['skip_agent']}`"
        )
        lines.append(
            f"- **Edge classifier output**: "
            f"fault_class=`{summary['alert_fault_class']}`, "
            f"severity=`{summary['alert_severity']}`, "
            f"confidence=`{summary['alert_confidence']}` "
            f"(edge latency = **{summary['edge_ms']} ms**)."
        )
        if summary["agent_called"]:
            if summary["recommendation_action"]:
                snippet = summary["recommendation_action"]
                if len(snippet) > 280:
                    snippet = snippet[:277] + "…"
                lines.append(
                    f"- **Agent recommendation** "
                    f"(urgency=`{summary['recommendation_urgency']}`, "
                    f"confidence=`{summary['recommendation_confidence']}`, "
                    f"agent latency = **{summary['agent_ms']} ms**, "
                    f"**{summary['knowledge_source_count']}** knowledge sources):  \n"
                    f"  > {snippet}"
                )
            else:
                lines.append(
                    f"- **Agent call failed** "
                    f"(`{summary['agent_error']}`); orchestrator-equivalent "
                    "event still persisted for traceability."
                )
        else:
            if scenario["skip_agent"]:
                lines.append(
                    "- **Agent intentionally skipped** "
                    "(`skip_agent=True`); the dashboard would show alert-only "
                    "output — demonstrates graceful-degradation UX when LLM "
                    "is unavailable."
                )
            else:
                lines.append(
                    f"- Severity `{summary['alert_severity']}` does not "
                    "trigger the agent (matches the orchestrator's "
                    "`AGENT_TRIGGER_SEVERITIES = {warning, critical}`)."
                )
        lines.append(f"- `event_id = {summary['event_id']}`\n")

    lines.append("## Provenance\n")
    lines.append(
        "All raw `OrchestratorEvent` JSON for the runs above is available in "
        "`reports/integration/fault_injection_demo.json` and replayed into "
        f"`{events_path.as_posix()}`.\n"
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-url", default="http://127.0.0.1:8000")
    parser.add_argument("--agent-url", default="http://127.0.0.1:8001")
    parser.add_argument(
        "--events-path",
        type=Path,
        default=Path("data/orchestrator/events.jsonl"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("reports/integration/fault_injection_demo.md"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("reports/integration/fault_injection_demo.json"),
    )
    args = parser.parse_args()

    args.events_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)

    results: list[tuple[dict, dict]] = []
    full_events: list[dict] = []

    for scenario in DEFAULT_SCENARIOS:
        print(f"→ {scenario['label']}")
        result = inject_fault_demo(
            system_type=scenario["system_type"],
            fault_class=scenario["fault_class"],
            operating_condition=scenario["operating_condition"],
            system_id=scenario["system_id"],
            edge_url=args.edge_url,
            agent_url=args.agent_url,
            events_path=args.events_path,
            seed=int(scenario["seed"]),
            skip_agent=bool(scenario["skip_agent"]),
        )
        summary = _summarise(result)
        results.append((scenario, summary))
        full_events.append(result.event.model_dump(mode="json"))
        print(
            f"   ok={summary['ok']}  "
            f"severity={summary['alert_severity']}  "
            f"urgency={summary['recommendation_urgency']}  "
            f"edge={summary['edge_ms']}ms  agent={summary['agent_ms']}ms"
        )

    md = _format_markdown(results, events_path=args.events_path)
    args.out_md.write_text(md, encoding="utf-8")
    args.out_json.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "edge_url": args.edge_url,
                "agent_url": args.agent_url,
                "events_path": str(args.events_path.as_posix()),
                "scenarios": [
                    {"input": {**{k: v for k, v in s.items() if k != "system_type"
                                  and k != "operating_condition"},
                               "system_type": s["system_type"].value,
                               "operating_condition":
                                   s["operating_condition"].value},
                     "summary": summary}
                    for s, summary in results
                ],
                "events": full_events,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {args.out_md}\nWrote {args.out_json}")


if __name__ == "__main__":
    main()
