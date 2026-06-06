# BESS Capacity Fade — Operator Playbook (PLACEHOLDER)

> **Document status (rule §12)**: this is a **placeholder MVP document**.

## Summary

Capacity fade is the gradual reduction of usable energy capacity across cycles.
Unlike acute faults, it accumulates slowly and is **expected** within
contractual limits. AgentPV flags pack-level deviations from the warranty
curve so the operator can plan replacement before service-level agreements
are breached.

## Symptoms in sensor signals

| Channel | Pattern |
|---|---|
| `SoC` (estimated) | Reaches 100% on smaller `Wh` totals each cycle |
| `Wh` | Discharge energy below the contracted curve |
| `R_est` | Slowly rising (sustained, not transient) |
| `T_cell` | Normal (key differentiator from thermal anomaly) |

## Severity

`monitor` — production is degraded but safety is not at risk. Convert to
`warning` when fade rate exceeds 1.5× the warranty curve.

## Recommended actions

**Within 1 month** (planning horizon):

1. Compare actual capacity against the warranty curve at the pack level.
2. If at or below warranty curve: continue normal operation; recompute the
   site's power-purchase-agreement headroom.
3. If degrading 20% faster than warranty: open a vendor-warranty claim.

## Escalation criteria

- Capacity below 80% of nameplate at < 3000 cycles → vendor RMA.
- Capacity loss accelerating month-over-month → suspect cell-level imbalance
  (see `placeholder_general_safety.md` Cell Balancing section).

## See also

- `placeholder_thermal_anomaly.md`
