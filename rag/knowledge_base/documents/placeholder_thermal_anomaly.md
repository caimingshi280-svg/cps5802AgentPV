# BESS Thermal Anomaly — Operator Playbook (PLACEHOLDER)

> **Document status (rule §12)**: this is a **placeholder MVP document**.

## Summary

BESS thermal anomalies are the **highest-risk** fault family because runaway
exotherm in lithium chemistries is irreversible once triggered. AgentPV's
edge classifier looks for elevated cell temperature combined with abnormal
internal-resistance behavior to flag this **before** thermal runaway begins.

## Symptoms in sensor signals

| Channel | Pattern |
|---|---|
| `T_cell` | Rises 5–15 °C above thermal-management setpoint |
| `R_est` | Rises 20–50% in correlation with temperature |
| `sigma_V` | Cell-to-cell voltage spread widens |
| `I_pack` | Often elevated relative to historical baseline |

## Severity

`critical` — every minute matters. The action plan must be initiated
immediately upon receiving the alert.

## Recommended actions

**Immediate (within 5 minutes)**:

1. Initiate emergency cooling (chiller setpoint -10 °C, fans 100%).
2. De-rate the affected pack to 30% C-rate via the PCS.
3. Alert the on-call BMS engineer and the local fire response team.

**Short-term (within 1 hour)**:

4. Isolate the affected rack from the parallel string at the DC contactor.
5. Run a depth-of-discharge audit; cells that exceeded DoD limits must be
   tagged for individual capacity test.

**Within 24 hours**:

6. Manufacturer notification + RMA process initiation.
7. Document in the fault registry; cross-reference any prior alerts on this
   pack ID.

## Escalation criteria

This event is **already at maximum severity**. The escalation here is to
**evacuation procedures** if `T_cell` exceeds the abuse threshold defined by
the cell manufacturer (typically 80 °C for NMC, 100 °C for LFP).

## See also

- `placeholder_general_safety.md`
