# BESS — Robustness, distribution shift & OOD

- ONNX FP32 model: `cnn1d_bess.onnx` ; test split n = **3395**.
- Clean baseline: Macro-F1 **0.9986**, accuracy 0.9985.

## 1. Distribution shift — per operating condition

| Operating condition | n | Macro-F1 | Accuracy | Mean confidence |
|---|---:|---:|---:|---:|
| `high_irradiance` | 1738 | 0.9989 | 0.9988 | 9.302 |
| `high_temperature` | 686 | 0.9986 | 0.9985 | 9.704 |
| `low_irradiance` | 971 | 0.9981 | 0.9979 | 9.915 |

![condition heatmap](figures/condition_heatmap.png)

## 2. Missing features (random channel masking)

| Mask ratio | Macro-F1 | Accuracy | Mean confidence |
|---:|---:|---:|---:|
| 0.00 | 0.9986 | 0.9985 | 9.559 |
| 0.10 | 0.6613 | 0.6622 | 13.379 |
| 0.30 | 0.5046 | 0.4999 | 16.491 |
| 0.50 | 0.3634 | 0.3508 | 17.744 |

![missing curve](figures/missing_features_curve.png)

## 3. Sensor noise (Gaussian ×channel-std)

| σ multiplier | Macro-F1 | Accuracy | Mean confidence |
|---:|---:|---:|---:|
| 0.00 | 0.9986 | 0.9985 | 9.559 |
| 0.05 | 0.9932 | 0.9929 | 9.532 |
| 0.10 | 0.9718 | 0.9708 | 9.441 |
| 0.20 | 0.9062 | 0.9049 | 9.226 |
| 0.50 | 0.7516 | 0.7599 | 8.813 |

![noise curve](figures/noise_curve.png)

## 4. Sensor scale drift

| Drift factor | Macro-F1 | Accuracy | Mean confidence |
|---:|---:|---:|---:|
| 0.80 | 0.2231 | 0.3175 | 20.615 |
| 0.90 | 0.4460 | 0.5125 | 17.198 |
| 1.00 | 0.9986 | 0.9985 | 9.559 |
| 1.10 | 0.1512 | 0.2571 | 33.351 |
| 1.20 | 0.0737 | 0.2256 | 65.934 |

![drift curve](figures/scale_drift_curve.png)

## 5. Adversarial perturbation (FGSM, gradient via PyTorch backend)

| ε (×channel-std) | Macro-F1 | Accuracy | Mean confidence |
|---:|---:|---:|---:|
| 0.00 | 0.9986 | 0.9985 | 9.559 |
| 0.01 | 0.8538 | 0.8471 | 8.681 |
| 0.02 | 0.6396 | 0.6274 | 8.396 |
| 0.05 | 0.5134 | 0.4925 | 9.104 |
| 0.10 | 0.4388 | 0.3991 | 11.027 |

![fgsm curve](figures/fgsm_curve.png)

## 6. Out-of-distribution detection (cross-system feed)

- In-distribution: this system's own test windows (n = 3395).
- OOD: windows from the *other* system (n = 4211) — analogous to an unseen attack-type / wrong-asset alert.

| Score | AUROC | Discriminability | FPR@95-TPR | Direction |
|---|---:|---:|---:|---|
| energy (Liu 2020) | 0.0000 | 1.0000 | 1.0000 | inverted (out > in) |
| max-softmax prob (Hendrycks 2017) | 0.0529 | 0.9471 | 1.0000 | inverted (out > in) |

> **Honest finding.** The energy score is *inverted* in this cross-system set-up: OOD windows obtain **higher** confidence than in-distribution windows. This is a known failure mode of post-hoc scores on self-contained standardised graphs — feeding cross-system raw values pushes inputs into the tails of the training distribution, which the convolutional stack converts into very peaked logits (low entropy, very negative energy ⇒ high `−energy`). The score is still highly *discriminative* (AUROC much further from 0.5 than max-softmax), but the deployment policy must use the **opposite** direction or be combined with an input-space density check to reject the high-magnitude OOD samples.

![ood histogram](figures/ood_energy_histogram.png)

## 7. Uncertainty-aware rejection policy

Calibrated on the **val** split for **95%** target coverage; energy-confidence threshold = **5.354**.

| Quantity | Value |
|---|---|
| In-distribution coverage | 0.9464 |
| In-distribution selective accuracy | **0.9994** |
| In-distribution risk (error among accepted) | 0.0006 |
| OOD reject rate at the same threshold | **0.0000** |

![risk vs coverage](figures/risk_coverage_curve.png)

## 8. Headline (presentation snapshot)

![overview](figures/overview_macro_f1.png)

The dual-axis chart below puts Macro-F1 and mean energy-confidence on the same x-axis. Any stress case where the bar drops **and** the line rises is a blind spot for the energy-based rejection policy and must be addressed by a second-layer detector or by hardening at training time.

![confidence sensitivity](figures/confidence_sensitivity.png)
