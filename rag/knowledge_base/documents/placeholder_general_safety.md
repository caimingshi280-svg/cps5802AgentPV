# General Safety & Escalation Procedures (PLACEHOLDER)

> **Document status (rule §12)**: this is a **placeholder MVP document**.

## Purpose

Cross-cutting safety guidance that applies to multiple fault classes. Cited
when the AgentPV agent decides escalation, vendor contact, or evacuation
guidance is warranted.

## Always-applicable principles

1. **Lock-out / tag-out (LOTO)** before any DC-side intervention. Verify
   absence of voltage with two independent meters.
2. **PPE**: arc-rated coverall (cat-2 minimum), insulating gloves, safety
   glasses. Add SCBA for any thermal anomaly response.
3. **Two-person rule** for any energized work above 50 V DC.

## When to escalate to a human supervisor

The agent must escalate (i.e., set `urgency=immediate` and notify the
supervisor channel) whenever:

- Severity is `critical` AND fault class is in
  `{Inverter_fault, String_disconnection, Thermal_anomaly}`.
- Confidence is `high` but the suggested action affects more than one
  asset (string, rack, inverter).
- The recommended remediation requires de-energization for > 1 hour
  during peak production hours.

## Cell balancing (BESS)

A widening cell-to-cell voltage spread (`sigma_V`) is a leading indicator of
imminent imbalance faults. If `sigma_V > 50 mV` is sustained for > 1 hour:

1. Force a balancing cycle at low C-rate (0.2 C).
2. If `sigma_V` does not converge below 30 mV after the cycle, schedule
   in-rack capacity test.

## Documentation hygiene

Every action taken in response to an AgentPV alert MUST be logged with:

- Alert trace id
- Operator name
- Action taken
- Outcome / next step
- Time

The cloud agent's `escalate_alert` tool emits a structured record so the
audit trail is uniform regardless of operator.

## See also

- `placeholder_partial_shading.md`
- `placeholder_inverter_fault.md`
- `placeholder_thermal_anomaly.md`
- `placeholder_capacity_fade.md`
