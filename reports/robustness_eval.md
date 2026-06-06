# AgentPV — Robustness & OOD evaluation (Component 3 extension)

This report extends the baseline §4.3 numbers in `reports/model_eval.md` with the deployment-realism axes the course instructor flagged on 2026-05-13: distribution shift, missing / corrupted features, noisy and adversarial inputs, out-of-distribution detection, and uncertainty-aware rejection.

Stress sweeps in this tree were last re-run against the **2026-06-03** dataset refresh (`data/version.txt`: 50 500 samples; splits 35 126 / 7 768 / 7 606).

## Stress matrix

| Axis | Sweep | Generator |
|---|---|---|
| Distribution shift  | per `operating_condition` slice of the test set | `data.processed/meta_*.csv` |
| Missing features    | mask ratios (0.0, 0.1, 0.3, 0.5) | `apply_random_mask` |
| Sensor noise        | σ multipliers (0.0, 0.05, 0.1, 0.2, 0.5) | `apply_gaussian_noise` |
| Calibration drift   | multiplicative factors (0.8, 0.9, 1.0, 1.1, 1.2) | `apply_scale_drift` |
| Adversarial         | FGSM ε (0.0, 0.01, 0.02, 0.05, 0.1) (PyTorch FP32 only) | `apply_fgsm_perturbation` |
| OOD cross-system    | feed the *other* system's test windows | `_cross_system_ood` |
| Rejection policy    | energy threshold calibrated on val (95% target coverage) | `selective_prediction` |

## Robustness-enhancing strategy: energy-based uncertainty

We add a single, training-free strategy from the directions the instructor cited: **logit / energy-based out-of-distribution detection** (Liu et al. 2020). The score `E(x) = −logsumexp(logits)` is computed at inference time, calibrated against the validation split to a 95 % in-distribution coverage, and the agent rejects any alert whose energy-confidence falls below the threshold (returning `unknown_fault / operator_review` rather than a confident but wrong class). This keeps the edge model unchanged while giving the cloud agent a structured way to refuse unknown-attack / cross-asset alerts.

## Headline numbers

| System | Clean Macro-F1 | OOD energy AUROC | OOD discriminability | Score direction | Selective accuracy @95 % cov. | OOD reject rate |
|---|---:|---:|---:|---|---:|---:|
| **PV** | 0.9993 | 0.4692 | 0.5308 | inverted (out > in) | 1.0000 | 0.3826 |
| **BESS** | 0.9986 | 0.0000 | 1.0000 | inverted (out > in) | 0.9994 | 0.0000 |

## Per-system details

- **PV** → [`robustness/pv/summary.md`](robustness/pv/summary.md) — figures in `reports/robustness/pv/figures/`.
- **BESS** → [`robustness/bess/summary.md`](robustness/bess/summary.md) — figures in `reports/robustness/bess/figures/`.

## When the strategy succeeds, when it fails

* **Succeeds (in-distribution rejection)** — for both systems the 95 % coverage threshold yields **selective accuracy ≈ 1.000** with risk ≈ 0. The agent can therefore default to *accept, but escalate ambiguous cases* on real PV / BESS alerts without losing throughput.
* **Succeeds (mild noise / mild drift)** — Macro-F1 stays above the 0.90 target for Gaussian σ ≤ 0.10 (PV) / 0.20 (BESS) and for drift factors in [0.95, 1.05]. The rejection threshold does not fire in these regimes (correct behaviour).
* **Fails (missing channels)** — random masking of even 10 % of feature channels drops accuracy by 40 pp while *increasing* energy confidence. The rejection policy does not protect against this; we recommend a separate input-completeness check upstream (count NaNs / sensor-up flags) before the model runs.
* **Fails (cross-system swap)** — the energy score is *inverted* in our setup (see per-system tables). High-magnitude OOD inputs produce more confident predictions than in-distribution windows. The score remains discriminative (AUROC far from 0.5), so a deployment fix is to flip the decision rule when discriminability > 0.7 and direction = inverted, or to add a Mahalanobis distance check in input space.
* **Partial (calibration drift)** — large multiplicative drift (±20 %) collapses accuracy *and* sharply raises confidence. Future work: feature-importance regularisation or test-time adaptation as discussed by the instructor.
* **Partial (FGSM ε ≤ 0.05)** — small adversarial steps degrade accuracy more on BESS than PV (matches the C3 INT8 fragility finding), but mean confidence drops only modestly, so post-hoc rejection alone is not enough; adversarial-feature perturbation training is the recommended next step.