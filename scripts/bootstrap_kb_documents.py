"""Generate curriculum-style Markdown articles under ``rag/knowledge_base/documents``.

Run once from repo root::

    python scripts/bootstrap_kb_documents.py

Skips filenames that already exist.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DOCS = _ROOT / "rag" / "knowledge_base" / "documents"

_ARTICLES: list[tuple[str, str, str]] = [
    ("kb_pv_normal_ops.md", "PV Normal Operations — Field Checklist", "Steady-state I–V, inverter efficiency, and when to log benign variance."),
    ("kb_partial_shading_iv.md", "Partial Shading — IV Curve Indicators", "Stepwise current plateaus, string mismatch, and safe string isolation order."),
    ("kb_soiling_loss.md", "Soiling Loss Estimation and Cleaning Windows", "Soiling ratio from performance ratio; scheduling washes vs ROI."),
    ("kb_bypass_diode_failure.md", "Bypass Diode Fault Signatures", "Hot-spot risk, substring voltage steps, and thermal imaging cues."),
    ("kb_string_disconnection.md", "String Disconnection — Isolation Procedure", "Zero-current string under irradiance; lockout/tagout and combiner checks."),
    ("kb_inverter_ground_fault.md", "Inverter Ground Fault Detection Response", "GFCI trips, insulation resistance trending, and vendor escalation."),
    ("kb_degradation_rate.md", "PV Degradation Rate — Tracking and Thresholds", "Linear vs seasonal PR drift; warranty claim evidence packages."),
    ("kb_bess_normal_ops.md", "BESS Normal Operations — Balancing and Limits", "SOC windows, C-rate adherence, and BMS alarm hygiene."),
    ("kb_capacity_fade_tracking.md", "Capacity Fade — Trending and Forecasting", "End-of-charge capacity tests, temperature compensation, and RUL hints."),
    ("kb_internal_resistance_rise.md", "Internal Resistance Increase — Diagnostics", "DCIR growth, pulse tests, and cell-level imbalance correlation."),
    ("kb_thermal_anomaly_bess.md", "BESS Thermal Anomaly — Containment First Steps", "Cooling loop verification, module isolation, and NFPA 855 awareness."),
    ("kb_cell_imbalance.md", "Cell Imbalance — Voltage Spread Interpretation", "Weak-cell detection, balancing cycles, and when to derate."),
    ("kb_arc_flash_pv.md", "Arc Flash and DC Safety on Large PV", "DC arc characteristics, PPE baselines, and rapid shutdown verification."),
    ("kb_fire_response_bess.md", "BESS Site Fire Response Playbook", "Venting hazards, water application cautions, and mutual aid coordination."),
    ("kb_comms_loss_scada.md", "SCADA / Comms Loss at the Plant Level", "Last-known-good states, local HMI checks, and alarm storm triage."),
    ("kb_weather_hail_pv.md", "Hail and Extreme Weather — Post-Event Inspection", "Module glass crack patterns, string I–V resampling, and insurance docs."),
    ("kb_grid_code_ride_through.md", "Grid Code Ride-Through — Inverter Settings", "LVRT/HVRT summaries, reactive power priority, and logging for compliance."),
    ("kb_curtailment_strategy.md", "Curtailment and Zero-Export Scenarios", "Plant-level power limits, ramp rates, and BESS absorption strategies."),
    ("kb_omc_shift_handover.md", "O&M Shift Handover for Hybrid PV+BESS", "Open alerts, permit status, and test-in-progress flags."),
    ("kb_spare_parts_inventory.md", "Critical Spares — Inverter and BESS Modules", "Lead times, interchangeable SKUs, and cold-storage for cells."),
    ("kb_warranty_evidence.md", "Warranty Evidence Collection for OEM Claims", "Environmental data completeness, fault counts, and firmware revision logs."),
    ("kb_cyber_nerc_cip_primer.md", "Cyber Hygiene Primer for Plant Networks", "Segmentation, VPN jump hosts, and vendor remote access controls."),
    ("kb_calibration_iv_tracer.md", "IV Tracer Calibration and Uncertainty", "Reference cell placement, temperature correction, and repeatability."),
    ("kb_module_mismatch_install.md", "Installation Mismatch — Batch and Rating Mix", "Nameplate audits, string homogeneity rules, and repower options."),
    ("kb_bess_soc_calibration.md", "SOC Estimation Drift and Recalibration", "OCV rest points, coulomb counting bias, and BMS recal events."),
]


def main() -> None:
    _DOCS.mkdir(parents=True, exist_ok=True)
    written = 0
    for filename, title, summary in _ARTICLES:
        path = _DOCS / filename
        if path.exists():
            continue
        body = textwrap.dedent(
            f"""\
            # {title}

            ## Summary

            {summary}

            ## Field actions

            1. Verify sensor plausibility against weather and neighbor strings or racks.
            2. Capture timestamps, firmware versions, and one-line electrical state.
            3. Escalate per local safety rules if personnel hazard or equipment damage risk.

            ## References (internal curriculum)

            This article is part of the AgentPV operator curriculum corpus for retrieval-augmented guidance.
            """
        ).strip()
        path.write_text(body + "\n", encoding="utf-8")
        written += 1
    existing = len(list(_DOCS.glob("*.md")))
    print(f"Wrote {written} new files; directory now has {existing} markdown document(s).")


if __name__ == "__main__":
    main()
