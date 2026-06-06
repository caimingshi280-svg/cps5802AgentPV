"""AgentPV Component 3 — Model Evaluation.

This package implements the evaluation pipeline mandated by the project
specification §4.3:

* Per-class precision / recall / F1 score
* Macro-average F1-score (assignment target ≥ 90 %)
* Confusion matrix (PNG heatmap)
* Inference latency: mean, p50, p95 over 1000 runs on CPU
* Model size on disk (rule §17 < 50 MB budget)

The package is intentionally side-effect free: each submodule exposes
pure functions / dataclasses, and the orchestration lives in
:mod:`evaluation.runner`. Outputs are written to ``reports/<system>/``.
"""

from evaluation.classification_report import ClassificationReport, build_classification_report
from evaluation.compare_variants import (
    VariantComparison,
    VariantRow,
    build_comparison_rows,
    compare_variants,
    comparison_to_markdown,
    load_variant_summary,
)
from evaluation.confusion_matrix import compute_confusion_matrix, render_confusion_matrix_png
from evaluation.latency_benchmark import LatencyResult, benchmark_latency
from evaluation.metrics import EvaluationPredictions, accuracy, macro_f1, per_class_metrics
from evaluation.model_size import ModelSizeReport, measure_model_size
from evaluation.predictor import Predictor

__all__ = [
    "EvaluationPredictions",
    "ClassificationReport",
    "LatencyResult",
    "ModelSizeReport",
    "Predictor",
    "VariantRow",
    "VariantComparison",
    "accuracy",
    "macro_f1",
    "per_class_metrics",
    "build_classification_report",
    "compute_confusion_matrix",
    "render_confusion_matrix_png",
    "benchmark_latency",
    "measure_model_size",
    "load_variant_summary",
    "build_comparison_rows",
    "comparison_to_markdown",
    "compare_variants",
]
