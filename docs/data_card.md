# AgentPV — Data Card v0.1.0

**Deliverable #1** of the AgentPV course project (CPS 5802 SP26).
This card documents the simulated PV + BESS time-series dataset on which
all downstream Component 2 / 3 / 6 work is trained and benchmarked.
Format follows the 14-section project documentation template (rule §24).

---

## 1. Purpose

A reproducible, fully-labelled time-series dataset that lets us train and
evaluate a multi-class fault classifier for photovoltaic (PV) and battery
energy storage systems (BESS) **without** access to proprietary
industrial data, in line with assignment §4.1 and §8.5.

---

## 2. Why it exists (why not Kaggle)

Per assignment §8.5, using a pre-built PV / battery fault dataset from
Kaggle / HuggingFace is forbidden. Real industrial data is also
proprietary and non-shareable. We therefore generate physics-inspired
synthetic data that is

- fully labelled per fault class,
- reproducible from a fixed seed,
- balanced enough to train a multi-class classifier without resampling
  hacks,
- distributed across three operating conditions to expose the model to
  the heterogeneity of real deployments.

---

## 3. Provenance & generation

| Field | Value |
|---|---|
| Generator | `simulation/generate_dataset.py` (this repository) |
| Seed | `42` (fixed; rule §6 reproducibility) |
| Sample rate | `1.0 Hz` (one row per second) |
| Window size | `60` samples (= 60 s of context per training example) |
| Generated at (UTC) | `2026-06-03T15:27:30Z` (last regeneration; `data/version.txt` is the live record) |
| Schema version | `0.1.0` |
| Source code revision | governed by git; `data/version.txt` records the seed and parameters used to produce the on-disk artefacts |

### Generation pipeline (high level)

```text
PVSimulator      ──╮
                   ├──► sample window  ──► fault_injector ──► (X[i], y[i])
BatterySimulator ──╯                       (label, class-specific perturbation)
```

- `PVSimulator` (`simulation/pv_simulator.py`) — physics-inspired model:
  - Irradiance `G` sampled per operating condition.
  - `T_module = T_amb + 0.030 * G` (NOCT).
  - `I_dc = (G / 1000) * (9.0 - 0.005 * (T_module - 25))`.
  - `V_dc = 38.0 + (-0.30) * (T_module - 25)`.
  - `P = V_dc * I_dc`; `P_ac = η_inv * P` with `η_inv ≈ 0.96`.
- `BatterySimulator` (`simulation/battery_simulator.py`) — RC ECM with
  SOC-driven OCV, Coulomb counter, thermal exponential, ageing terms.
- `fault_injector` (`simulation/fault_injector.py`) — deterministic,
  rng-driven perturbations per fault class.

---

## 4. Dataset layout (on disk)

```text
data/
├── processed/
│   ├── X_pv.npz        # array key "X" — shape (28000, 60, 8) float32
│   ├── y_pv.npz        # array key "y" — shape (28000,)  <U32 string labels
│   ├── meta_pv.csv     # columns: local_idx, sample_idx, system_id, system_type, label, operating_condition
│   ├── X_bess.npz      # shape (22500, 60, 8) float32
│   ├── y_bess.npz      # shape (22500,)
│   └── meta_bess.csv
├── splits/
│   ├── train.csv       # 35 126 rows
│   ├── val.csv         #  7 768 rows
│   └── test.csv        #  7 606 rows
└── version.txt         # frozen JSON metadata snapshot for the run above (n_samples = 50 500)
```

### Feature order (contract — do not reorder)

| System | Feature order |
|---|---|
| PV (`X_pv`)   | `V_dc, I_dc, P, T_module, T_amb, G, P_ac, eta` |
| BESS (`X_bess`)| `V_term, I, SOC, T, R_est, sigma_V, N_cycle, SoH` |

Both systems use 8 channels so the same 1D-CNN architecture can serve
either after re-training (rule §7).

---

## 5. Class taxonomy

### PV (7 classes; assignment §4.1.0)

| Class id | Label | Description |
|---|---|---|
| 0 | `PV_Normal` | Healthy PV string operating within nominal envelope |
| 1 | `Partial_shading` | Localised shading on a sub-string |
| 2 | `Soiling` | Dust / dirt accumulation reducing transmittance |
| 3 | `Bypass_diode_fault` | One bypass diode failing short / open |
| 4 | `String_disconnection` | A string breaking electrical continuity |
| 5 | `Inverter_fault` | DC→AC conversion fault |
| 6 | `Degradation` | Long-term efficiency loss (LID/PID) |

### BESS (5 classes; assignment §4.1.0)

| Class id | Label | Description |
|---|---|---|
| 0 | `BESS_Normal` | Healthy battery rack |
| 1 | `Capacity_fade` | Gradual SoH decline |
| 2 | `Internal_resistance_increase` | Rising R_est over cycles |
| 3 | `Thermal_anomaly` | Cell or pack abnormal temperature |
| 4 | `Cell_imbalance` | Voltage spread across cells (`sigma_V`) |

The label taxonomy is sourced from the single-truth tuple
`PV_FAULT_CLASSES` / `BESS_FAULT_CLASSES` in `api/schemas.py` (rule §3).

---

## 6. Class distribution

Total samples: **50 500** (≥ 50 000 — assignment §4.1.0 requirement).

| Class | Count | % of total |
|---|---:|---:|
| `PV_Normal` | 8 002 | 15.85 % |
| `Partial_shading` | 3 333 | 6.60 % |
| `Soiling` | 3 333 | 6.60 % |
| `Bypass_diode_fault` | 3 333 | 6.60 % |
| `String_disconnection` | 3 333 | 6.60 % |
| `Inverter_fault` | 3 333 | 6.60 % |
| `Degradation` | 3 333 | 6.60 % |
| `BESS_Normal` | 5 000 | 9.90 % |
| `Capacity_fade` | 4 375 | 8.66 % |
| `Internal_resistance_increase` | 4 375 | 8.66 % |
| `Thermal_anomaly` | 4 375 | 8.66 % |
| `Cell_imbalance` | 4 375 | 8.66 % |

Imbalance is moderate (max/min ratio = 8 002 / 3 333 ≈ **2.40**) — the
`*_Normal` classes are intentionally over-represented because operators
need many healthy windows to suppress false alarms. A weighted
cross-entropy loss (`training/data.py::class_weights`) reweights the
minority fault classes during training so per-class F1 remains balanced
(see `reports/{pv,bess}/onnx_fp32/classification_report.md`).

---

## 7. Operating-condition distribution

Sampled with weights 5 : 3 : 2 — high irradiance is the dominant case
and high temperature the rarest.

| Operating condition | Count | % |
|---|---:|---:|
| `high_irradiance` | 25 110 | 49.7 % |
| `low_irradiance`  | 15 359 | 30.4 % |
| `high_temperature`| 10 031 | 19.9 % |

Each condition appears in **every** class (no condition × class pair is
empty). Distribution matches assignment §4.1.0 requirement of "≥ 3
operating conditions".

---

## 8. Splits (70 / 15 / 15)

Splits are **stratified by `system_id`**, not by sample row, to prevent
leakage of the same simulated asset across train / val / test
(`simulation/generate_dataset.py::_stratified_split_by_system_id`).
Same seed `42` is used so any future re-generation produces identical
splits.

| Split | Samples | Share |
|---|---:|---:|
| `train` | 35 126 | 69.6 % |
| `val`   |  7 768 | 15.4 % |
| `test`  |  7 606 | 15.1 % |

The slight deviation from 70 / 15 / 15 is the cost of ID-level
stratification: a `system_id` lands fully in one split.

---

## 9. Schema validation

A 1-of-N (`validate_every=500`) sample is round-tripped through
`api.schemas.SensorWindow` + `RawSample` Pydantic validation during
generation. This catches:

- non-finite values (NaN / Inf),
- mismatched window-size / feature-length,
- system_type vs label class mismatches (`PV_Normal` cannot appear with a
  BESS window).

Hence sample-level validity is checked even at scale without paying the
100 % per-row Pydantic overhead.

---

## 10. Known limitations

| # | Limitation | Impact | Mitigation / planned fix |
|---|---|---|---|
| 1 | Physics models are "inspired" not "calibrated" — no field data fit | Real-world distribution shift expected | Document as MVP; polish phase considers `pvlib` for PV |
| 2 | Sample rate is 1 Hz (per-second); real PV inverter telemetry is often 1–5 min | Frequency-domain features absent | Acceptable per assignment ("statistically distinguishable signatures") |
| 3 | All classes balanced 4 000–4 500; real fault rates are ≪ 1 % | Class weighting may over-state recall on rare faults | Documented; future work: imbalanced ablation |
| 4 | No degraded-sensor noise (stuck readings, outliers) | Robustness undertested | Polish phase: add sensor-fault overlay |
| 5 | Independent windows (no temporal correlation across windows for the same `system_id`) | Cannot evaluate sequence-level cumulative drift | By design; sequence-level work is Component 6 / 7 territory |
| 6 | BESS_Normal vs Thermal_anomaly partly overlap on idle cells | Slight confusion (~5 %) on test set | Documented in `reports/bess/summary.md`; polish phase: feature engineering |

---

## 11. Reproducibility

```powershell
python -m simulation.generate_dataset --seed 42 \
       --n-pv 28000 --n-bess 22500 \
       --n-pv-normal 4000 --n-bess-normal 4500 \
       --window-size 60
```

Produces byte-identical `X_*.npz` / `y_*.npz` / split CSVs across runs
(rule §6). The `data/version.txt` JSON snapshot records every parameter
above; comparing two `version.txt` files is sufficient to know whether
two researchers have the same data.

---

## 12. Intended use

| Use | Status |
|---|---|
| Training the AgentPV multi-class fault classifier (Component 2) | ✅ Primary use |
| Evaluating the classifier (Component 3) | ✅ Test split is held out |
| Calibrating the simulation-to-edge alert pipeline (Component 6/7) | ✅ Used by orchestrator + dashboard |
| Publication-quality industrial PV/BESS analysis | ❌ Not validated against field data |
| Decision-making for any real PV/BESS asset | ❌ Synthetic only — explicitly out of scope |

---

## 13. Ethical / legal

- No personally identifiable information.
- No proprietary data (synthetic only).
- No health / safety claims about real assets — this dataset is a
  pedagogical artefact for the CPS 5802 course project.

---

## 14. Versioning & changelog

| Version | Date | Notes |
|---|---|---|
| `0.1.0` | 2026-05-09 | First full-scale dataset (50 500 samples, seed 42, window 60). |
| (planned) | — | `0.2.0` will add a sensor-noise / dropout overlay for robustness. |

The version string is shared with `api/schemas.py::DatasetMetadata.schema_version`;
bumping it is the contractual signal that downstream code may need to
re-train or re-validate.
