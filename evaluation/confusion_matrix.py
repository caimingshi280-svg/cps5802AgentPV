"""Confusion matrix + heatmap rendering (assignment §4.3).

We split the work into:

* :func:`compute_confusion_matrix` — pure NumPy, fully testable.
* :func:`render_confusion_matrix_png` — uses matplotlib to dump a PNG.

Splitting matters because matplotlib pulls in a large rendering backend,
and we want unit tests of the math to stay headless and fast.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix as _sk_confusion_matrix

from evaluation.metrics import EvaluationPredictions
from utils.logging_config import get_logger

log = get_logger(__name__)


def compute_confusion_matrix(predictions: EvaluationPredictions) -> np.ndarray:
    """Return an integer confusion matrix of shape ``(n_classes, n_classes)``.

    Rows index the true class, columns index the predicted class — the
    standard convention also used by ``sklearn``.
    """

    n_classes = predictions.n_classes
    if predictions.n_samples == 0:
        return np.zeros((n_classes, n_classes), dtype=np.int64)
    cm = _sk_confusion_matrix(
        predictions.y_true,
        predictions.y_pred,
        labels=list(range(n_classes)),
    )
    return cm.astype(np.int64)


def confusion_matrix_to_dict(cm: np.ndarray, label_classes: tuple[str, ...]) -> dict[str, Any]:
    """JSON-serialisable dict of the matrix + axis labels."""

    if cm.shape != (len(label_classes), len(label_classes)):
        raise ValueError(
            f"confusion matrix shape {cm.shape} doesn't match "
            f"len(label_classes)={len(label_classes)}"
        )
    return {
        "labels": list(label_classes),
        "matrix": cm.astype(int).tolist(),
    }


def render_confusion_matrix_png(
    cm: np.ndarray,
    label_classes: tuple[str, ...],
    *,
    output_path: Path,
    title: str = "Confusion matrix",
    normalise: bool = True,
) -> Path:
    """Save a heatmap PNG of the confusion matrix.

    Parameters
    ----------
    cm
        Integer matrix from :func:`compute_confusion_matrix`.
    label_classes
        Class names used for both axes (length must equal ``cm.shape[0]``).
    output_path
        Destination ``.png`` path; parent directory is created if missing.
    title
        Figure title.
    normalise
        If True, cell colour intensity reflects per-row recall (each row
        sums to 1). The cell text still shows raw integer counts.
    """

    # Lazy import keeps tests of the math fast and lets headless servers
    # without matplotlib still import the module successfully (rule §13
    # graceful degradation — not strictly required here, but cheap).
    import matplotlib

    matplotlib.use("Agg")  # never pop a window during CI
    import matplotlib.pyplot as plt

    if cm.shape != (len(label_classes), len(label_classes)):
        raise ValueError(
            f"confusion matrix shape {cm.shape} doesn't match "
            f"len(label_classes)={len(label_classes)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True)
        # 1.0 for empty rows — avoids division by zero and yields a flat row.
        safe = np.where(row_sums == 0, 1, row_sums)
        cm_display = cm.astype(np.float64) / safe
    else:
        cm_display = cm.astype(np.float64)

    fig, ax = plt.subplots(
        figsize=(max(6.0, 0.6 * len(label_classes) + 3), max(5.0, 0.5 * len(label_classes) + 3))
    )
    im = ax.imshow(cm_display, cmap="Blues", aspect="auto")

    ax.set_xticks(np.arange(len(label_classes)))
    ax.set_yticks(np.arange(len(label_classes)))
    ax.set_xticklabels(label_classes, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(label_classes, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Annotate each cell with the integer count.
    threshold = cm_display.max() / 2.0 if cm_display.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm[i, j])
            colour = "white" if cm_display[i, j] > threshold else "black"
            ax.text(j, i, str(count), ha="center", va="center", color=colour, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalised" if normalise else "count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    log.info(
        "confusion_matrix_png_written",
        extra={"path": str(output_path), "n_classes": len(label_classes)},
    )
    return output_path
