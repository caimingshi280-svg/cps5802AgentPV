# Inverter Fault — Operator Playbook (PLACEHOLDER)

> **Document status (rule §12)**: this is a **placeholder MVP document**.
> Replace with the manufacturer's commissioning manual in the polish phase.

## Summary

Inverter faults span a broad failure mode space: DC-link capacitor aging,
IGBT module failure, control-board firmware faults, AC-side over/undervoltage
trips. AgentPV's edge classifier flags the family; the operator's job is to
narrow down to the root cause **before** approving a truck roll.

## Symptoms in sensor signals

| Channel | Pattern |
|---|---|
| `P_ac` | Collapses to ~0 while `P_dc` remains positive |
| `eta` | Drops to < 0.5 (efficiency = P_ac / P_dc) |
| `V_dc` | May rise above MPPT setpoint (no current draw) |
| `T_module` | Slowly elevated (passive heating only) |

## Severity

`critical` — equipment is safe to leave for hours, but production loss is
immediate and full. Schedule a same-day intervention.

## Recommended actions

1. **Within 30 minutes**: verify the alert with the inverter vendor's
   monitoring portal (cross-check error code).
2. **Within 4 hours**: dispatch a qualified technician. Bring spare DC-link
   capacitors and IGBT modules per the spare-parts catalog.
3. **Within 24 hours**: if vendor portal shows arc-fault detection (AFCI),
   isolate strings and treat as electrical safety event (see
   `placeholder_general_safety.md`).

## Escalation criteria

Escalate to vendor RMA if:

- Same fault recurs within 30 days after part replacement, OR
- Firmware version is older than the latest critical-path patch.

## See also

- `placeholder_thermal_anomaly.md`
- `placeholder_general_safety.md`
