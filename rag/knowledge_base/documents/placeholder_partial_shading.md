# Partial Shading — Operator Playbook (PLACEHOLDER)

> **Document status (rule §12)**: this is a **placeholder MVP document**.
> The polish-phase knowledge base will replace it with vendor-sourced material.

## Summary

Partial shading occurs when only part of a PV string receives sunlight, causing
the shaded cells to become reverse-biased. Bypass diodes activate to prevent
hotspots, but produce a characteristic **multi-step I-V curve** with reduced
peak power.

## Symptoms in sensor signals

| Channel | Pattern |
|---|---|
| `P_dc` | Drops 20–60% below modeled MPPT envelope |
| `V_dc` | Step pattern, multiple local optima |
| `eta` | Drops 0.1–0.3 below clean baseline |
| `G` | Normal (key differentiator from soiling) |

If irradiance is near design point but power is low, partial shading is far
more likely than soiling or degradation.

## Severity

`warning` — does not damage equipment immediately. However, persistent
shading causes long-term thermal stress on bypass diodes.

## Recommended actions

1. **Within 48 h**: visual inspection / drone thermography to identify the
   shading source (vegetation, bird droppings, structure shadow).
2. **Within 1 week**: clear vegetation, schedule cleaning.
3. **If shading is structural (cannot be removed)**: file a re-routing
   ticket; consider module-level power electronics retrofit.

## Escalation criteria

Escalate to `critical` if:

- Bypass diode open-circuit voltage exceeds 0.7 V at low current (indicates
  failed diode), OR
- Power loss > 60% for more than 4 consecutive hours.

## See also

- `placeholder_inverter_fault.md`
- `placeholder_general_safety.md`
