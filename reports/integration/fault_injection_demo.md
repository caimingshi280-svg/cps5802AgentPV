# Component 7 — Interactive fault-injection demo

> Assignment §4.7 / Deliverable #9. Single-click fault triggers, full edge → agent pipeline response, persisted to the dashboard's JSONL feed for live inspection.

_Generated_: 2026-06-03T16:19Z

_Events appended to_: `data/orchestrator/events_c7_demo.jsonl` (the dashboard picks them up on the next 🔄 Refresh)

## How to drive it interactively

1. `python -m uvicorn api.edge_service:app  --host 127.0.0.1 --port 8000`
2. `python -m uvicorn api.agent_service:app --host 127.0.0.1 --port 8001`
3. `streamlit run dashboard/app.py`
4. Open the sidebar's **🔥 Fault injection (demo)** expander → pick system / fault / operating condition / seed → click **Inject fault**.
5. The success banner appears at the top of the main pane; the new event also lands in the 'Event timeline' and 'Event detail' tabs.

## Scripted reproduction (this report)

```powershell
python scripts/demo_fault_injection.py
```

## Headline outcomes

| # | Scenario | Severity | Urgency | Edge ms | Agent ms | Knowledge sources | OK |
| ---: | --- | --- | --- | ---: | ---: | ---: | :---: |
| 1 | PV inverter fault (critical) — full pipeline | `critical` | `immediate` | 26.15 | 8653 | 3 | ✅ |
| 2 | PV partial shading (warning) — full pipeline | `warning` | `scheduled` | 6.79 | 8476 | 3 | ✅ |
| 3 | BESS thermal anomaly (critical) — full pipeline | `critical` | `immediate` | 8.15 | 8705 | 3 | ✅ |
| 4 | PV normal — agent skipped (edge-only response) | `monitor` | `—` | 6.53 | — | 0 | ✅ |
| 5 | PV critical fault with skip_agent=True (graceful degradation) | `critical` | `—` | 11.31 | — | 0 | ✅ |

## Per-scenario detail

### 1. PV inverter fault (critical) — full pipeline

- **Input**: `system=PV`, `fault=Inverter_fault`, `op=high_irradiance`, `seed=4242`, `skip_agent=False`
- **Edge classifier output**: fault_class=`Inverter_fault`, severity=`critical`, confidence=`1.0` (edge latency = **26.15 ms**).
- **Agent recommendation** (urgency=`immediate`, confidence=`medium`, agent latency = **8653.2 ms**, **3** knowledge sources):  
  > Run a partial shading test to identify the fault source. If the issue persists, perform an inverter ground fault detection response. If still present, inspect for bypass diode faults.
- `event_id = 11e3126f943647fcbf945942f84dd5bd`

### 2. PV partial shading (warning) — full pipeline

- **Input**: `system=PV`, `fault=Partial_shading`, `op=high_irradiance`, `seed=13`, `skip_agent=False`
- **Edge classifier output**: fault_class=`Partial_shading`, severity=`warning`, confidence=`0.9998` (edge latency = **6.79 ms**).
- **Agent recommendation** (urgency=`scheduled`, confidence=`medium`, agent latency = **8476.5 ms**, **3** knowledge sources):  
  > Check the inverter's MPPT envelope to ensure it is functioning correctly. Verify that the irradiance levels are within design point values. If issues persist, consider running a diagnostic test on the system.
- `event_id = 6102c43fde234b77b0df33259d261915`

### 3. BESS thermal anomaly (critical) — full pipeline

- **Input**: `system=BESS`, `fault=Thermal_anomaly`, `op=high_temperature`, `seed=99`, `skip_agent=False`
- **Edge classifier output**: fault_class=`Thermal_anomaly`, severity=`critical`, confidence=`0.9961` (edge latency = **8.15 ms**).
- **Agent recommendation** (urgency=`immediate`, confidence=`medium`, agent latency = **8705.1 ms**, **3** knowledge sources):  
  > Trigger immediate intervention for Thermal_anomaly fault in BESS system DEMO-BESS-THERMAL-001. Refer to 'BESS Thermal Anomaly — Containment First Steps' procedure for containment first steps. Escalate to operations team for further action.
- `event_id = 116d603eab194fc984fa4e10044eae8c`

### 4. PV normal — agent skipped (edge-only response)

- **Input**: `system=PV`, `fault=PV_Normal`, `op=low_irradiance`, `seed=7`, `skip_agent=False`
- **Edge classifier output**: fault_class=`PV_Normal`, severity=`monitor`, confidence=`0.9986` (edge latency = **6.53 ms**).
- Severity `monitor` does not trigger the agent (matches the orchestrator's `AGENT_TRIGGER_SEVERITIES = {warning, critical}`).
- `event_id = 3e3a5bd688f64f2dbb3a8438bb18ed88`

### 5. PV critical fault with skip_agent=True (graceful degradation)

- **Input**: `system=PV`, `fault=String_disconnection`, `op=high_irradiance`, `seed=21`, `skip_agent=True`
- **Edge classifier output**: fault_class=`String_disconnection`, severity=`critical`, confidence=`0.9999` (edge latency = **11.31 ms**).
- **Agent intentionally skipped** (`skip_agent=True`); the dashboard would show alert-only output — demonstrates graceful-degradation UX when LLM is unavailable.
- `event_id = d9bea5b2330f46b2af1f1c07644cd975`

## Provenance

All raw `OrchestratorEvent` JSON for the runs above is available in `reports/integration/fault_injection_demo.json` and replayed into `data/orchestrator/events_c7_demo.jsonl`.

