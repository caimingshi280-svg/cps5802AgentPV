"""Robustness, distribution-shift and OOD diagnostics for the edge classifier.

Component 3 deepens the assignment §4.3 evaluation with the directions
the instructor flagged in S15: distribution shift, missing / corrupted
features, unseen-fault detection, and an uncertainty-aware rejection
policy.  Everything in this module is **pure-function-ish** numpy code
— it never touches disk and never creates figures; the runner in
:mod:`scripts.run_robustness_eval` is the side-effect layer.

Public surface
--------------
Stress generators (deterministic given ``seed``):

* :func:`apply_random_mask`            — drop random feature channels
* :func:`apply_gaussian_noise`         — per-channel Gaussian noise
* :func:`apply_scale_drift`            — multiplicative sensor drift
* :func:`apply_fgsm_perturbation`      — gradient-based input attack (PyTorch backend only)

Uncertainty / OOD scores:

* :func:`softmax`                      — numerically stable
* :func:`max_softmax_probability`      — classic baseline (Hendrycks 2017)
* :func:`logit_margin`                 — top-1 minus top-2
* :func:`energy_score`                 — free-energy OOD score (Liu 2020)

Aggregation helpers:

* :func:`auroc_fpr95`                  — binary in/out separation
* :func:`risk_coverage_curve`          — coverage / risk under a score threshold
* :func:`threshold_at_target_coverage` — pick a threshold from a calibration set
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Score functions
# ---------------------------------------------------------------------------


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along ``axis``."""

    a = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=axis, keepdims=True)


def max_softmax_probability(logits: np.ndarray) -> np.ndarray:
    """Maximum softmax probability — Hendrycks & Gimpel 2017 baseline OOD score."""

    return softmax(logits, axis=-1).max(axis=-1)


def logit_margin(logits: np.ndarray) -> np.ndarray:
    """Top-1 logit minus top-2 logit. Larger ⇒ more confident."""

    sorted_logits = np.sort(logits, axis=-1)
    return sorted_logits[..., -1] - sorted_logits[..., -2]


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Energy-based OOD score: ``E(x) = -T * logsumexp(logits / T)`` (Liu et al. 2020).

    Smaller / more negative energy ⇒ higher confidence (in-distribution).
    Larger / less negative energy ⇒ likely OOD.

    We negate this convention before plotting so that *higher score =
    more confident* matches the softmax-style score and produces a
    clean risk-coverage curve.
    """

    if temperature <= 0:
        raise ValueError("energy_score temperature must be > 0")
    a = logits / temperature
    m = a.max(axis=-1, keepdims=True)
    lse = m.squeeze(-1) + np.log(np.exp(a - m).sum(axis=-1))
    return -temperature * lse  # raw "energy" sign; runners may negate


def confidence_from_energy(energy: np.ndarray) -> np.ndarray:
    """Flip energy so that higher = more confident (matches softmax direction)."""

    return -energy


# ---------------------------------------------------------------------------
# Stress generators — operate on raw window arrays of shape (N, T, F)
# ---------------------------------------------------------------------------


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def apply_random_mask(
    x: np.ndarray,
    channel_means: np.ndarray,
    *,
    mask_ratio: float,
    seed: int = 42,
) -> np.ndarray:
    """Randomly mask whole feature channels per sample.

    A "missing" channel is replaced by its **training-set mean** so that
    after the ONNX graph's built-in ``(x - mean) / std`` standardisation
    the masked channel becomes ~0 (the "neutral" input the network is
    most calibrated for). Without this trick a zero-filled raw channel
    would be a huge OOD signal on its own and the model would crash
    immediately, conflating "missing" with "broken sensor".
    """

    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError(f"mask_ratio must be in [0,1], got {mask_ratio}")
    if x.ndim != 3:
        raise ValueError(f"expected (N, T, F), got {x.shape}")
    n, _t, f = x.shape
    if channel_means.shape != (f,):
        raise ValueError(
            f"channel_means shape {channel_means.shape} != (F={f},)"
        )

    rng = _rng(seed)
    n_drop = int(round(mask_ratio * f))
    if n_drop == 0:
        return x.copy()

    out = x.copy()
    for i in range(n):
        drop_idx = rng.choice(f, size=n_drop, replace=False)
        # ``out[i, :, drop_idx]`` triggers numpy's advanced-indexing
        # shape inversion; using ``out[i][:, drop_idx]`` keeps the
        # (T, n_drop) layout we want and lets (n_drop,) broadcast.
        out[i][:, drop_idx] = channel_means[drop_idx]
    return out


def apply_gaussian_noise(
    x: np.ndarray,
    channel_std: np.ndarray,
    *,
    sigma_mult: float,
    seed: int = 42,
) -> np.ndarray:
    """Add per-channel Gaussian noise with std = ``sigma_mult × channel_std``."""

    if sigma_mult < 0:
        raise ValueError(f"sigma_mult must be ≥ 0, got {sigma_mult}")
    if x.ndim != 3:
        raise ValueError(f"expected (N, T, F), got {x.shape}")
    if channel_std.shape != (x.shape[-1],):
        raise ValueError(
            f"channel_std shape {channel_std.shape} != (F={x.shape[-1]},)"
        )
    if sigma_mult == 0:
        return x.copy()

    rng = _rng(seed)
    noise = rng.standard_normal(size=x.shape).astype(np.float32)
    # Broadcasting (1, 1, F) over (N, T, F)
    return (x + noise * (sigma_mult * channel_std)[None, None, :]).astype(np.float32)


def apply_scale_drift(
    x: np.ndarray,
    *,
    factor: float,
    channels: Iterable[int] | None = None,
) -> np.ndarray:
    """Multiplicative calibration drift on selected channels (all if ``None``)."""

    if x.ndim != 3:
        raise ValueError(f"expected (N, T, F), got {x.shape}")
    out = x.copy()
    if channels is None:
        return (out * np.float32(factor)).astype(np.float32)
    chans = list(channels)
    for c in chans:
        if c < 0 or c >= x.shape[-1]:
            raise IndexError(f"channel {c} out of range for F={x.shape[-1]}")
    out[..., chans] = (out[..., chans] * np.float32(factor)).astype(np.float32)
    return out


def apply_fgsm_perturbation(
    pytorch_predictor,
    x: np.ndarray,
    y_true: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    """FGSM-style gradient attack in **raw input space** for a PyTorch backend.

    The sign of the gradient is taken w.r.t. the raw input (the
    predictor handles standardisation internally), then a small
    epsilon-scaled step is added. We use a per-channel scaling of
    ``epsilon × std`` so the attack budget is comparable across the
    8 sensors which span very different physical units.

    Parameters
    ----------
    pytorch_predictor
        An object exposing ``_model``, ``_mean``, ``_std`` (i.e. the
        :class:`evaluation.pytorch_runner.PyTorchClassifier` class).
        ONNX backends do not expose gradients, so adversarial testing
        is reported only against the PyTorch FP32 baseline.
    x
        ``(N, T, F)`` float32 raw inputs.
    y_true
        ``(N,)`` integer class labels matching ``predictor.label_classes``.
    epsilon
        Step size in units of per-channel std. ``0.02`` is a tiny
        physically-plausible drift; ``0.10`` is aggressive.
    """

    import torch
    from torch import nn

    model = pytorch_predictor._model
    mean = pytorch_predictor._mean
    std = pytorch_predictor._std

    model.eval()
    x_t = torch.from_numpy(x.astype(np.float32)).clone().requires_grad_(True)
    y_t = torch.from_numpy(y_true.astype(np.int64))

    x_std = (x_t - mean) / std
    logits = model(x_std)
    loss = nn.functional.cross_entropy(logits, y_t)
    loss.backward()

    grad_sign = x_t.grad.detach().sign()
    # Scale per-channel by std so all sensors share a comparable budget.
    step = (epsilon * std)[None, None, :] * grad_sign
    x_adv = (x_t.detach() + step).cpu().numpy().astype(np.float32)
    return x_adv


# ---------------------------------------------------------------------------
# OOD aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OODSeparation:
    """In-distribution vs OOD score separation metrics."""

    auroc: float
    fpr_at_95_tpr: float
    n_in: int
    n_out: int

    def to_json(self) -> dict[str, float | int]:
        return {
            "auroc": round(float(self.auroc), 4),
            "fpr_at_95_tpr": round(float(self.fpr_at_95_tpr), 4),
            "n_in": int(self.n_in),
            "n_out": int(self.n_out),
        }


def auroc_fpr95(scores_in: np.ndarray, scores_out: np.ndarray) -> OODSeparation:
    """AUROC + FPR@95-TPR for a confidence score (higher = more in-dist).

    Returns
    -------
    OODSeparation
        Containing the AUROC of the classifier "in vs out" and the
        false-positive-rate (OOD wrongly accepted) when the threshold
        is set to keep 95% of in-distribution samples.
    """

    if scores_in.size == 0 or scores_out.size == 0:
        return OODSeparation(auroc=float("nan"), fpr_at_95_tpr=float("nan"),
                             n_in=int(scores_in.size), n_out=int(scores_out.size))

    y = np.concatenate([np.ones_like(scores_in), np.zeros_like(scores_out)])
    s = np.concatenate([scores_in, scores_out])
    auroc = float(roc_auc_score(y, s))

    threshold = float(np.quantile(scores_in, 0.05))  # 5th percentile → 95% TPR
    fpr = float(np.mean(scores_out >= threshold))
    return OODSeparation(auroc=auroc, fpr_at_95_tpr=fpr,
                         n_in=int(scores_in.size), n_out=int(scores_out.size))


# ---------------------------------------------------------------------------
# Selective prediction (rejection policy)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskCoveragePoint:
    threshold: float
    coverage: float
    risk: float
    n_accepted: int
    n_total: int


def threshold_at_target_coverage(
    calibration_scores: np.ndarray,
    *,
    target_coverage: float = 0.95,
) -> float:
    """Pick a confidence threshold that accepts ``target_coverage`` of calibration."""

    if not 0.0 < target_coverage <= 1.0:
        raise ValueError(f"target_coverage must be in (0,1], got {target_coverage}")
    if calibration_scores.size == 0:
        return float("-inf")
    q = 1.0 - target_coverage
    return float(np.quantile(calibration_scores, q))


def risk_coverage_curve(
    scores: np.ndarray,
    correct: np.ndarray,
    *,
    n_points: int = 21,
) -> list[RiskCoveragePoint]:
    """Sweep thresholds and return (coverage, risk) pairs.

    Risk is the error rate among **accepted** samples (those with score
    ≥ threshold). Coverage is the fraction accepted. As threshold rises,
    coverage falls and risk usually falls too — a steep, monotonically
    decreasing curve indicates a well-calibrated confidence signal.
    """

    if scores.shape != correct.shape:
        raise ValueError(
            f"scores shape {scores.shape} != correct shape {correct.shape}"
        )
    if scores.size == 0:
        return []

    quantiles = np.linspace(0.0, 1.0, n_points)
    thresholds = np.quantile(scores, quantiles)

    out: list[RiskCoveragePoint] = []
    n_total = int(scores.size)
    for t in thresholds:
        accepted = scores >= t
        n_acc = int(accepted.sum())
        if n_acc == 0:
            out.append(
                RiskCoveragePoint(threshold=float(t), coverage=0.0, risk=0.0,
                                  n_accepted=0, n_total=n_total)
            )
            continue
        risk = 1.0 - float(correct[accepted].mean())
        coverage = n_acc / n_total
        out.append(
            RiskCoveragePoint(threshold=float(t), coverage=float(coverage),
                              risk=float(risk), n_accepted=n_acc,
                              n_total=n_total)
        )
    return out


def selective_metrics(
    correct: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float | int]:
    """Coverage + risk + selective accuracy at one threshold."""

    accepted = scores >= threshold
    n_acc = int(accepted.sum())
    n_total = int(scores.size)
    if n_acc == 0:
        return {
            "coverage": 0.0, "risk": 0.0, "selective_accuracy": 0.0,
            "n_accepted": 0, "n_total": n_total,
        }
    sel_acc = float(correct[accepted].mean())
    return {
        "coverage": round(n_acc / n_total, 4),
        "risk": round(1.0 - sel_acc, 4),
        "selective_accuracy": round(sel_acc, 4),
        "n_accepted": n_acc,
        "n_total": n_total,
    }
