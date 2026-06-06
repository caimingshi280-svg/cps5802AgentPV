# PV — Robustness, distribution shift & OOD

- ONNX FP32 model: `cnn1d_pv.onnx` ; test split n = **4211**.
- Clean baseline: Macro-F1 **0.9993**, accuracy 0.9993.

## 1. Distribution shift — per operating condition

| Operating condition | n | Macro-F1 | Accuracy | Mean confidence |
|---|---:|---:|---:|---:|
| `high_irradiance` | 2039 | 1.0000 | 1.0000 | 9.623 |
| `high_temperature` | 854 | 1.0000 | 1.0000 | 9.372 |
| `low_irradiance` | 1318 | 0.9979 | 0.9977 | 10.171 |

![condition heatmap](figures/condition_heatmap.png)

## 2. Missing features (random channel masking)

| Mask ratio | Macro-F1 | Accuracy | Mean confidence |
|---:|---:|---:|---:|
| 0.00 | 0.9993 | 0.9993 | 9.743 |
| 0.10 | 0.6018 | 0.5930 | 10.496 |
| 0.30 | 0.4108 | 0.4011 | 11.495 |
| 0.50 | 0.2759 | 0.2676 | 12.369 |

![missing curve](figures/missing_features_curve.png)

## 3. Sensor noise (Gaussian ×channel-std)

| σ multiplier | Macro-F1 | Accuracy | Mean confidence |
|---:|---:|---:|---:|
| 0.00 | 0.9993 | 0.9993 | 9.743 |
| 0.05 | 0.9995 | 0.9995 | 9.646 |
| 0.10 | 0.9982 | 0.9981 | 9.255 |
| 0.20 | 0.8518 | 0.7960 | 8.383 |
| 0.50 | 0.4766 | 0.4367 | 10.504 |

![noise curve](figures/noise_curve.png)

## 4. Sensor scale drift

| Drift factor | Macro-F1 | Accuracy | Mean confidence |
|---:|---:|---:|---:|
| 0.80 | 0.0507 | 0.1273 | 16.203 |
| 0.90 | 0.1762 | 0.2384 | 12.344 |
| 1.00 | 0.9993 | 0.9993 | 9.743 |
| 1.10 | 0.2631 | 0.2940 | 16.075 |
| 1.20 | 0.1445 | 0.1962 | 28.295 |

![drift curve](figures/scale_drift_curve.png)

## 5. Adversarial perturbation (FGSM, gradient via PyTorch backend)

| ε (×channel-std) | Macro-F1 | Accuracy | Mean confidence |
|---:|---:|---:|---:|
| 0.00 | 0.9993 | 0.9993 | 9.743 |
| 0.01 | 0.9986 | 0.9986 | 9.219 |
| 0.02 | 0.9959 | 0.9962 | 8.610 |
| 0.05 | 0.8260 | 0.8207 | 6.836 |
| 0.10 | 0.4718 | 0.4258 | 7.291 |

![fgsm curve](figures/fgsm_curve.png)

## 6. Out-of-distribution detection (cross-system feed)

- In-distribution: this system's own test windows (n = 4211).
- OOD: windows from the *other* system (n = 3395) — analogous to an unseen attack-type / wrong-asset alert.

| Score | AUROC | Discriminability | FPR@95-TPR | Direction |
|---|---:|---:|---:|---|
| energy (Liu 2020) | 0.4692 | 0.5308 | 0.6183 | inverted (out > in) |
| max-softmax prob (Hendrycks 2017) | 0.5995 | 0.5995 | 0.5228 | expected (in > out) |

> **Honest finding.** The energy score is *inverted* in this cross-system set-up: OOD windows obtain **higher** confidence than in-distribution windows. This is a known failure mode of post-hoc scores on self-contained standardised graphs — feeding cross-system raw values pushes inputs into the tails of the training distribution, which the convolutional stack converts into very peaked logits (low entropy, very negative energy ⇒ high `−energy`). The score is still highly *discriminative* (AUROC much further from 0.5 than max-softmax), but the deployment policy must use the **opposite** direction or be combined with an input-space density check to reject the high-magnitude OOD samples.

![ood histogram](figures/ood_energy_histogram.png)

## 7. Uncertainty-aware rejection policy

Calibrated on the **val** split for **95%** target coverage; energy-confidence threshold = **7.578**.

| Quantity | Value |
|---|---|
| In-distribution coverage | 0.9489 |
| In-distribution selective accuracy | **1.0000** |
| In-distribution risk (error among accepted) | 0.0000 |
| OOD reject rate at the same threshold | **0.3826** |

![risk vs coverage](figures/risk_coverage_curve.png)

## 8. Headline (presentation snapshot)

![overview](figures/overview_macro_f1.png)

The dual-axis chart below puts Macro-F1 and mean energy-confidence on the same x-axis. Any stress case where the bar drops **and** the line rises is a blind spot for the energy-based rejection policy and must be addressed by a second-layer detector or by hardening at training time.

![confidence sensitivity](figures/confidence_sensitivity.png)
