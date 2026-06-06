"""Pure-function metrics used by Component 3 (Model Evaluation).

This module contains **only** numeric kernels. No I/O, no plotting, no
model loading — everything that takes a prediction array + label array
and produces a scalar / table.

The functions delegate to scikit-learn for the heavy lifting (well-tested,
already pinned in ``requirements/ml.txt``) but wrap the outputs in typed
``pydantic`` models so the downstream JSON / Markdown writers do not have
to care about ``np.float64`` vs ``float``.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)


@dataclass(frozen=True)
class EvaluationPredictions:
    """Container for a single split's predictions vs ground truth.

    Used by every other evaluation routine so the public API takes a
    well-typed value rather than a pair of stringly-shaped arrays.
    """

    y_true: np.ndarray  # shape (N,) int label ids
    y_pred: np.ndarray  # shape (N,) int label ids
    label_classes: tuple[str, ...]  # length n_classes; index = id

    def __post_init__(self) -> None:
        if self.y_true.ndim != 1 or self.y_pred.ndim != 1:
            raise ValueError(
                f"y_true and y_pred must be 1-D, got shapes "
                f"{self.y_true.shape} / {self.y_pred.shape}"
            )
        if self.y_true.shape != self.y_pred.shape:
            raise ValueError(
                f"y_true / y_pred length mismatch: "
                f"{self.y_true.shape[0]} vs {self.y_pred.shape[0]}"
            )
        n_classes = len(self.label_classes)
        if n_classes == 0:
            raise ValueError("label_classes must be non-empty")
        for arr_name, arr in (("y_true", self.y_true), ("y_pred", self.y_pred)):
            if arr.size and (arr.min() < 0 or arr.max() >= n_classes):
                raise ValueError(
                    f"{arr_name} contains label id outside [0, {n_classes}); "
                    f"min={int(arr.min())}, max={int(arr.max())}"
                )

    @property
    def n_classes(self) -> int:
        return len(self.label_classes)

    @property
    def n_samples(self) -> int:
        return int(self.y_true.shape[0])


def accuracy(predictions: EvaluationPredictions) -> float:
    """Plain top-1 accuracy. Returns 0.0 on an empty split (no division by zero)."""

    if predictions.n_samples == 0:
        return 0.0
    return float(accuracy_score(predictions.y_true, predictions.y_pred))


def macro_f1(predictions: EvaluationPredictions) -> float:
    """Macro-average F1 — the assignment §4.3 primary target (≥ 0.90).

    Uses ``zero_division=0`` so a class with no predictions contributes
    0.0 rather than emitting a warning. The macro average therefore
    *penalises* classes the model never predicts, which is the safety
    behaviour we want.
    """

    if predictions.n_samples == 0:
        return 0.0
    return float(
        f1_score(
            predictions.y_true,
            predictions.y_pred,
            labels=list(range(predictions.n_classes)),
            average="macro",
            zero_division=0,
        )
    )


@dataclass(frozen=True)
class PerClassRow:
    """One class's precision / recall / F1 / support."""

    label: str
    label_id: int
    precision: float
    recall: float
    f1: float
    support: int


def per_class_metrics(predictions: EvaluationPredictions) -> list[PerClassRow]:
    """Return one row per class with precision / recall / F1 / support.

    Classes with zero support **and** zero predictions are still returned
    (with metrics = 0) so the report has a stable, deterministic shape
    regardless of which classes the model happened to predict.
    """

    n_classes = predictions.n_classes
    if predictions.n_samples == 0:
        # sklearn refuses empty arrays; we keep a uniform deterministic shape
        # so callers (Markdown / JSON writers) never have to special-case it.
        return [
            PerClassRow(
                label=predictions.label_classes[idx],
                label_id=idx,
                precision=0.0,
                recall=0.0,
                f1=0.0,
                support=0,
            )
            for idx in range(n_classes)
        ]

    p, r, f, s = precision_recall_fscore_support(
        predictions.y_true,
        predictions.y_pred,
        labels=list(range(n_classes)),
        zero_division=0,
    )
    rows: list[PerClassRow] = []
    for idx in range(n_classes):
        rows.append(
            PerClassRow(
                label=predictions.label_classes[idx],
                label_id=idx,
                precision=float(p[idx]),
                recall=float(r[idx]),
                f1=float(f[idx]),
                support=int(s[idx]),
            )
        )
    return rows


def predictions_from_arrays(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    label_classes: Sequence[str],
) -> EvaluationPredictions:
    """Convenience constructor that coerces python sequences to ndarrays."""

    return EvaluationPredictions(
        y_true=np.asarray(y_true, dtype=np.int64),
        y_pred=np.asarray(y_pred, dtype=np.int64),
        label_classes=tuple(label_classes),
    )


def to_dict(rows: list[PerClassRow]) -> list[dict[str, Any]]:
    """Helper for JSON serialisation."""

    return [
        {
            "label": r.label,
            "label_id": r.label_id,
            "precision": round(r.precision, 6),
            "recall": round(r.recall, 6),
            "f1": round(r.f1, 6),
            "support": r.support,
        }
        for r in rows
    ]
