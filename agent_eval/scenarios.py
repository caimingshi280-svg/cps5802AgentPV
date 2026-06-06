"""Benchmark scenario contracts + default ≥30-scenario suite (Component 5).

Scenarios are loaded from ``agent_eval/benchmark.json`` when present; otherwise
the programmatic default from :func:`default_benchmark_scenarios` is used so
unit tests never depend on an on-disk JSON file.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field

from agent.orchestration.llm_client import urgency_for_severity
from api.schemas import (
    ALL_FAULT_CLASSES,
    BESS_FAULT_CLASSES,
    PV_FAULT_CLASSES,
    Alert,
    Severity,
    StrictBaseModel,
    SystemType,
    Urgency,
)

_PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_BENCHMARK_PATH = _PACKAGE_DIR / "benchmark.json"


class ExpectedOutcome(StrictBaseModel):
    """Oracle-style expectations for heuristic checks + judge context."""

    expected_urgency: Urgency
    must_contain_keywords: list[str] = Field(
        default_factory=list,
        description="Each keyword must appear as substring in recommended_action (case-insensitive).",
    )
    must_not_contain_keywords: list[str] = Field(
        default_factory=list,
        description="None of these phrases may appear in recommended_action (case-insensitive).",
    )
    min_knowledge_sources: int = Field(default=0, ge=0, le=50)


class BenchmarkScenario(StrictBaseModel):
    """One graded agent scenario."""

    id: str = Field(min_length=1, max_length=128)
    stakes: str = Field(
        default="medium",
        description="low | medium | high — informational for the LLM judge prompt.",
    )
    tags: list[str] = Field(default_factory=list)
    alert: Alert
    expected: ExpectedOutcome


def default_benchmark_scenarios() -> list[BenchmarkScenario]:
    """Return **36** scenarios covering PV/BESS taxonomies + ambiguous / edge cases.

    Coverage target (assignment Component 5):
    * ≥ 30 scenarios total
    * ≥ 5 ambiguous / cross-edge rows (``tags`` contains ``ambiguous``)
    """

    def _ts() -> datetime:
        return datetime(2026, 3, 15, 14, 23, tzinfo=UTC)

    def _snap(**kwargs: float | str) -> dict[str, float | str]:
        return {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in kwargs.items()}

    scenarios: list[BenchmarkScenario] = []

    def add(
        sid: str,
        *,
        stakes: str,
        alert: Alert,
        expected: ExpectedOutcome,
        tags: list[str] | None = None,
    ) -> None:
        scenarios.append(
            BenchmarkScenario(
                id=sid,
                stakes=stakes,
                tags=tags or [],
                alert=alert,
                expected=expected,
            )
        )

    # --- PV normal (2) -------------------------------------------------
    for i, conf in enumerate((0.58, 0.82), start=1):
        alert = Alert(
            timestamp=_ts(),
            system_id=f"PV_SITE_{i:03d}",
            system_type=SystemType.PV,
            fault_class="PV_Normal",
            severity=Severity.MONITOR,
            confidence=conf,
            sensor_snapshot=_snap(irradiance_Wm2=980.0, pac_kW=4.2),
        )
        add(
            f"scn_pv_normal_monitor_{i:02d}",
            stakes="low",
            alert=alert,
            expected=ExpectedOutcome(
                expected_urgency=urgency_for_severity(alert.severity),
                must_contain_keywords=["[MOCK]", "Inspect"],
                must_not_contain_keywords=["wait and see", "ignore"],
                min_knowledge_sources=0,
            ),
        )

    # --- BESS normal (2) ----------------------------------------------
    for i, conf in enumerate((0.61, 0.88), start=1):
        alert = Alert(
            timestamp=_ts(),
            system_id=f"BESS_RACK_{i:03d}",
            system_type=SystemType.BESS,
            fault_class="BESS_Normal",
            severity=Severity.MONITOR,
            confidence=conf,
            sensor_snapshot=_snap(T_cell_C=32.0, soc_pct=55.0),
        )
        add(
            f"scn_bess_normal_monitor_{i:02d}",
            stakes="low",
            alert=alert,
            expected=ExpectedOutcome(
                expected_urgency=urgency_for_severity(alert.severity),
                must_contain_keywords=["[MOCK]", "Inspect"],
                must_not_contain_keywords=["wait and see"],
                min_knowledge_sources=0,
            ),
        )

    # --- PV faults: one scenario per non-normal class (6) --------------
    for fault in PV_FAULT_CLASSES:
        if fault == "PV_Normal":
            continue
        sev = (
            Severity.CRITICAL
            if fault in {"Inverter_fault", "String_disconnection"}
            else Severity.WARNING
        )
        alert = Alert(
            timestamp=_ts(),
            system_id="PV_SITE_100",
            system_type=SystemType.PV,
            fault_class=fault,
            severity=sev,
            confidence=0.91,
            sensor_snapshot=_snap(irradiance_Wm2=650.0, pac_kW=2.1, t_module_C=48.0),
        )
        add(
            f"scn_pv_fault_{fault.lower()}",
            stakes="high" if sev is Severity.CRITICAL else "medium",
            alert=alert,
            expected=ExpectedOutcome(
                expected_urgency=urgency_for_severity(sev),
                must_contain_keywords=["[MOCK]", "Inspect"],
                must_not_contain_keywords=["wait and see", "do nothing"],
                min_knowledge_sources=0,
            ),
        )

    # --- BESS faults: one per non-normal (5) --------------------------
    for fault in BESS_FAULT_CLASSES:
        if fault == "BESS_Normal":
            continue
        sev = Severity.CRITICAL if fault == "Thermal_anomaly" else Severity.WARNING
        alert = Alert(
            timestamp=_ts(),
            system_id="BESS_RACK_200",
            system_type=SystemType.BESS,
            fault_class=fault,
            severity=sev,
            confidence=0.89,
            sensor_snapshot=_snap(T_cell_C=44.0, v_pack_V=720.0, i_a=12.0),
        )
        add(
            f"scn_bess_fault_{fault.lower()}",
            stakes="high" if sev is Severity.CRITICAL else "medium",
            alert=alert,
            expected=ExpectedOutcome(
                expected_urgency=urgency_for_severity(sev),
                must_contain_keywords=["[MOCK]", "Inspect"],
                must_not_contain_keywords=["wait and see"],
                min_knowledge_sources=0,
            ),
        )

    # --- Ambiguous / edge (≥8, tag ``ambiguous``) --------------------
    ambiguous_specs: list[tuple[str, str, Alert, ExpectedOutcome]] = [
        (
            "scn_ambiguous_low_conf_critical_label",
            "Operator suspects critical but model confidence is low.",
            Alert(
                timestamp=_ts(),
                system_id="BESS_RACK_777",
                system_type=SystemType.BESS,
                fault_class="Thermal_anomaly",
                severity=Severity.CRITICAL,
                confidence=0.41,
                sensor_snapshot=_snap(T_cell_C=52.0, soc_pct=90.0),
            ),
            ExpectedOutcome(
                expected_urgency=Urgency.IMMEDIATE,
                must_contain_keywords=["[MOCK]"],
                must_not_contain_keywords=["wait and see"],
                min_knowledge_sources=0,
            ),
        ),
        (
            "scn_ambiguous_high_conf_warning",
            "High confidence warning — should still be scheduled urgency.",
            Alert(
                timestamp=_ts(),
                system_id="PV_SITE_888",
                system_type=SystemType.PV,
                fault_class="Soiling",
                severity=Severity.WARNING,
                confidence=0.97,
                sensor_snapshot=_snap(irradiance_Wm2=820.0, pac_kW=3.4),
            ),
            ExpectedOutcome(
                expected_urgency=Urgency.SCHEDULED,
                must_contain_keywords=["[MOCK]", "Inspect"],
                must_not_contain_keywords=[],
                min_knowledge_sources=0,
            ),
        ),
        (
            "scn_ambiguous_mixed_string_id",
            "Non-standard ID format (still valid string).",
            Alert(
                timestamp=_ts(),
                system_id="pv-site-west-09",
                system_type=SystemType.PV,
                fault_class="Partial_shading",
                severity=Severity.WARNING,
                confidence=0.76,
                sensor_snapshot=_snap(irradiance_Wm2=540.0),
            ),
            ExpectedOutcome(
                expected_urgency=Urgency.SCHEDULED,
                must_contain_keywords=["pv-site-west-09"],
                must_not_contain_keywords=[],
                min_knowledge_sources=0,
            ),
        ),
        (
            "scn_ambiguous_bess_cell_imbalance_warning",
            "Cell imbalance as warning (not yet critical).",
            Alert(
                timestamp=_ts(),
                system_id="BESS_RACK_303",
                system_type=SystemType.BESS,
                fault_class="Cell_imbalance",
                severity=Severity.WARNING,
                confidence=0.84,
                sensor_snapshot=_snap(delta_v_mv=120.0, soc_pct=40.0),
            ),
            ExpectedOutcome(
                expected_urgency=Urgency.SCHEDULED,
                must_contain_keywords=["Cell_imbalance"],
                must_not_contain_keywords=["wait and see"],
                min_knowledge_sources=0,
            ),
        ),
        (
            "scn_ambiguous_pv_degradation_monitor",
            "Slow degradation flagged as monitor.",
            Alert(
                timestamp=_ts(),
                system_id="PV_SITE_404",
                system_type=SystemType.PV,
                fault_class="Degradation",
                severity=Severity.MONITOR,
                confidence=0.63,
                sensor_snapshot=_snap(pr_mpp_pct=96.0),
            ),
            ExpectedOutcome(
                expected_urgency=Urgency.MONITOR,
                must_contain_keywords=["Degradation"],
                must_not_contain_keywords=[],
                min_knowledge_sources=0,
            ),
        ),
        (
            "scn_ambiguous_cross_history_spike",
            "Thermal anomaly critical — mock history tool still returns zeros.",
            Alert(
                timestamp=_ts(),
                system_id="BESS_RACK_909",
                system_type=SystemType.BESS,
                fault_class="Thermal_anomaly",
                severity=Severity.CRITICAL,
                confidence=0.93,
                sensor_snapshot=_snap(T_cell_C=68.0, t_rate_C_per_h=3.5),
            ),
            ExpectedOutcome(
                expected_urgency=Urgency.IMMEDIATE,
                must_contain_keywords=["[MOCK]", "Thermal_anomaly"],
                must_not_contain_keywords=["wait"],
                min_knowledge_sources=0,
            ),
        ),
        (
            "scn_ambiguous_pv_bypass_diode_warning",
            "Bypass diode warning with borderline confidence.",
            Alert(
                timestamp=_ts(),
                system_id="PV_SITE_515",
                system_type=SystemType.PV,
                fault_class="Bypass_diode_fault",
                severity=Severity.WARNING,
                confidence=0.55,
                sensor_snapshot=_snap(v_string_V=720.0, i_string_A=8.1),
            ),
            ExpectedOutcome(
                expected_urgency=Urgency.SCHEDULED,
                must_contain_keywords=["Bypass_diode_fault"],
                must_not_contain_keywords=[],
                min_knowledge_sources=0,
            ),
        ),
        (
            "scn_ambiguous_bess_internal_resistance_warning",
            "Internal resistance increase — scheduled follow-up.",
            Alert(
                timestamp=_ts(),
                system_id="BESS_RACK_616",
                system_type=SystemType.BESS,
                fault_class="Internal_resistance_increase",
                severity=Severity.WARNING,
                confidence=0.87,
                sensor_snapshot=_snap(r_internal_mohm=14.5),
            ),
            ExpectedOutcome(
                expected_urgency=Urgency.SCHEDULED,
                must_contain_keywords=["Internal_resistance_increase"],
                must_not_contain_keywords=[],
                min_knowledge_sources=0,
            ),
        ),
    ]
    for sid, desc, alert, expected in ambiguous_specs:
        add(sid, stakes="medium", alert=alert, expected=expected, tags=["ambiguous", desc])

    # --- Stress / taxonomy completeness (7) ----------------------------
    for idx, fault in enumerate(sorted(set(ALL_FAULT_CLASSES))):
        if idx >= 7:
            break
        st = SystemType.PV if fault in PV_FAULT_CLASSES else SystemType.BESS
        alert = Alert(
            timestamp=_ts(),
            system_id=f"LOOP_{idx:02d}",
            system_type=st,
            fault_class=fault,
            severity=Severity.WARNING,
            confidence=0.8,
            sensor_snapshot=_snap(idx=float(idx)),
        )
        add(
            f"scn_taxonomy_sweep_{idx:02d}_{fault.lower()}",
            stakes="low",
            alert=alert,
            expected=ExpectedOutcome(
                expected_urgency=Urgency.SCHEDULED,
                must_contain_keywords=["[MOCK]"],
                must_not_contain_keywords=[],
                min_knowledge_sources=0,
            ),
            tags=["taxonomy_sweep"],
        )

    # --- Extra edge rows (push total comfortably above 30) ------------
    add(
        "scn_edge_pv_soiling_monitor",
        stakes="low",
        alert=Alert(
            timestamp=_ts(),
            system_id="PV_SITE_EDGE_01",
            system_type=SystemType.PV,
            fault_class="Soiling",
            severity=Severity.MONITOR,
            confidence=0.66,
            sensor_snapshot=_snap(irradiance_Wm2=910.0, pr_mpp_pct=94.0),
        ),
        expected=ExpectedOutcome(
            expected_urgency=Urgency.MONITOR,
            must_contain_keywords=["Soiling", "[MOCK]"],
            must_not_contain_keywords=[],
            min_knowledge_sources=0,
        ),
        tags=["edge_case"],
    )
    add(
        "scn_edge_bess_capacity_warning_repeat",
        stakes="medium",
        alert=Alert(
            timestamp=_ts(),
            system_id="BESS_RACK_EDGE_02",
            system_type=SystemType.BESS,
            fault_class="Capacity_fade",
            severity=Severity.WARNING,
            confidence=0.92,
            sensor_snapshot=_snap(soc_pct=72.0, q_throughput_ah=4100.0),
        ),
        expected=ExpectedOutcome(
            expected_urgency=Urgency.SCHEDULED,
            must_contain_keywords=["Capacity_fade"],
            must_not_contain_keywords=["wait and see"],
            min_knowledge_sources=0,
        ),
        tags=["edge_case"],
    )
    add(
        "scn_edge_pv_string_disc_warning_not_critical_severity",
        stakes="high",
        alert=Alert(
            timestamp=_ts(),
            system_id="PV_SITE_EDGE_03",
            system_type=SystemType.PV,
            fault_class="String_disconnection",
            severity=Severity.WARNING,
            confidence=0.79,
            sensor_snapshot=_snap(v_string_V=0.0, i_string_A=0.0),
        ),
        expected=ExpectedOutcome(
            expected_urgency=Urgency.SCHEDULED,
            must_contain_keywords=["String_disconnection"],
            must_not_contain_keywords=["wait and see"],
            min_knowledge_sources=0,
        ),
        tags=["edge_case", "ambiguous"],
    )
    add(
        "scn_edge_bess_normal_as_warning_operator_override",
        stakes="low",
        alert=Alert(
            timestamp=_ts(),
            system_id="BESS_RACK_EDGE_04",
            system_type=SystemType.BESS,
            fault_class="BESS_Normal",
            severity=Severity.WARNING,
            confidence=0.7,
            sensor_snapshot=_snap(T_cell_C=35.0),
        ),
        expected=ExpectedOutcome(
            expected_urgency=Urgency.SCHEDULED,
            must_contain_keywords=["BESS_Normal"],
            must_not_contain_keywords=[],
            min_knowledge_sources=0,
        ),
        tags=["edge_case", "ambiguous"],
    )

    assert len(scenarios) >= 30, f"expected ≥30 scenarios, got {len(scenarios)}"
    ambiguous_count = sum(1 for s in scenarios if "ambiguous" in s.tags)
    assert ambiguous_count >= 5, f"expected ≥5 ambiguous, got {ambiguous_count}"
    return scenarios


def load_benchmark_json(path: Path | None = None) -> list[BenchmarkScenario]:
    """Load scenarios from JSON; fall back to :func:`default_benchmark_scenarios`."""

    p = path or DEFAULT_BENCHMARK_PATH
    if p.exists():
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"benchmark JSON must be a list, got {type(raw)}")
        return [BenchmarkScenario.model_validate(item) for item in raw]
    return default_benchmark_scenarios()


def write_default_benchmark_json(path: Path | None = None) -> Path:
    """Write :func:`default_benchmark_scenarios` to ``agent_eval/benchmark.json``."""

    out = path or DEFAULT_BENCHMARK_PATH
    data = [s.model_dump(mode="json") for s in default_benchmark_scenarios()]
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
