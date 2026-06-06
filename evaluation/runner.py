"""Top-level evaluation runner — Component 3 orchestration.

For one ``(system_type, onnx_path, test_split)`` combination this module:

1. Loads the test split (raw arrays — standardisation is embedded in
   the ONNX graph by :func:`quantization.onnx_export.export_checkpoint`).
2. Runs predictions through the ONNX classifier.
3. Builds a :class:`evaluation.classification_report.ClassificationReport`.
4. Computes the confusion matrix + writes a PNG heatmap.
5. Benchmarks ``n_runs`` single-sample CPU latency calls.
6. Measures the on-disk model size.
7. Writes a JSON summary, the heatmap PNG, and a Markdown summary into
   ``out_dir`` (default: ``reports/<system_lower>/``).

The runner is intentionally side-effect heavy — that is what an
"evaluation pipeline" is supposed to do. All numeric kernels live in
sibling modules and are independently unit-tested.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from api.schemas import SplitName, SystemType
from evaluation.classification_report import (
    ClassificationReport,
    build_classification_report,
)
from evaluation.confusion_matrix import (
    compute_confusion_matrix,
    confusion_matrix_to_dict,
    render_confusion_matrix_png,
)
from evaluation.latency_benchmark import LatencyResult, benchmark_latency
from evaluation.metrics import EvaluationPredictions
from evaluation.model_size import ModelSizeReport, measure_model_size
from evaluation.predictor import Predictor
from inference.onnx_runner import OnnxClassifier
from training.data import _load_split_arrays
from utils.logging_config import get_logger
from utils.paths import PROCESSED_DIR, PROJECT_ROOT, SPLITS_DIR, ensure_dir

log = get_logger(__name__)

REPORTS_DIR: Path = PROJECT_ROOT / "reports"


# ---------------------------------------------------------------------------
# High-level result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationArtifacts:
    """All artefacts written for one evaluation run."""

    summary_json_path: Path
    summary_markdown_path: Path
    confusion_matrix_png_path: Path
    classification_report: ClassificationReport
    latency: LatencyResult
    model_size: ModelSizeReport

    def to_json(self) -> dict[str, Any]:
        return {
            "summary_json_path": str(self.summary_json_path),
            "summary_markdown_path": str(self.summary_markdown_path),
            "confusion_matrix_png_path": str(self.confusion_matrix_png_path),
            "classification_report": self.classification_report.to_json(),
            "latency": self.latency.to_json(),
            "model_size": self.model_size.to_json(),
        }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _run_predictions(
    classifier: Predictor,
    x_split: np.ndarray,
    y_strings: np.ndarray,
    *,
    batch_size: int = 256,
) -> EvaluationPredictions:
    """Convert raw test arrays + a Predictor into typed predictions.

    The predictor is responsible for any pre-processing (standardisation,
    type conversion). We batch through the split so memory stays bounded
    even when N is in the tens of thousands.
    """

    label_classes = classifier.label_classes
    label_to_id = {lbl: idx for idx, lbl in enumerate(label_classes)}

    n = int(x_split.shape[0])
    if n == 0:
        return EvaluationPredictions(
            y_true=np.empty((0,), dtype=np.int64),
            y_pred=np.empty((0,), dtype=np.int64),
            label_classes=label_classes,
        )

    pred_ids = np.empty(n, dtype=np.int64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        logits = classifier.run_logits(x_split[start:end])
        pred_ids[start:end] = logits.argmax(axis=1).astype(np.int64)

    y_true_ids = np.array(
        [label_to_id[str(lbl)] for lbl in y_strings.tolist()], dtype=np.int64
    )
    return EvaluationPredictions(
        y_true=y_true_ids,
        y_pred=pred_ids,
        label_classes=label_classes,
    )


def _write_summary_markdown(
    *,
    out_path: Path,
    classification: ClassificationReport,
    latency: LatencyResult,
    size: ModelSizeReport,
    system_type: SystemType,
    model_path: Path,
    variant_name: str,
    confusion_png_path: Path,
) -> None:
    """Write a human-readable Markdown summary."""

    rel_png = confusion_png_path.name  # link relative to out_path's directory
    lines: list[str] = [
        f"# AgentPV Component 3 — {system_type.value} `{variant_name}` evaluation summary",
        "",
        f"- **Variant**: `{variant_name}`",
        f"- **Model artefact**: `{model_path}`",
        f"- **Split**: `{classification.split}`",
        f"- **Samples**: {classification.n_samples}",
        f"- **Classes**: {classification.n_classes}",
        "",
        "## Aggregate metrics",
        "",
        f"- Accuracy: **{classification.accuracy:.4f}**",
        f"- Macro-F1: **{classification.macro_f1:.4f}** "
        + ("✅ ≥ 0.90 target met" if classification.macro_f1 >= 0.90 else "⚠️ below 0.90 target"),
        f"- Weighted-F1: {classification.weighted_f1:.4f}",
        "",
        classification.to_markdown(),
        "",
        "## Confusion matrix",
        "",
        f"![confusion matrix]({rel_png})",
        "",
        "## CPU latency benchmark",
        "",
        f"- Runs: {latency.n_runs} (warm-up {latency.n_warmup}, batch={latency.batch_size})",
        f"- Mean: {latency.mean_ms:.3f} ms",
        f"- p50: {latency.p50_ms:.3f} ms",
        f"- p95: **{latency.p95_ms:.3f} ms** "
        + ("✅ ≤ 100 ms" if latency.p95_ms <= 100.0 else "❌ over budget"),
        f"- p99: {latency.p99_ms:.3f} ms",
        f"- min / max: {latency.min_ms:.3f} ms / {latency.max_ms:.3f} ms",
        "",
        "## Model size",
        "",
        f"- File: `{size.path}`",
        f"- Size: {size.kib:.2f} KiB ({size.mib:.4f} MiB)",
        f"- Budget: {size.budget_mib:.0f} MiB — "
        + ("✅ within budget" if size.within_budget else "❌ over budget"),
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def evaluate_predictor(
    *,
    predictor: Predictor,
    model_path: Path,
    system_type: SystemType,
    variant_name: str,
    out_dir: Path,
    split: SplitName = SplitName.TEST,
    processed_dir: Path = PROCESSED_DIR,
    splits_dir: Path = SPLITS_DIR,
    n_latency_runs: int = 1000,
    n_latency_warmup: int = 50,
    latency_seed: int = 42,
    size_budget_mib: float = 50.0,
    extra_summary: dict[str, Any] | None = None,
) -> EvaluationArtifacts:
    """Run Component 3 evaluation on any object satisfying :class:`Predictor`.

    This is the shared kernel behind :func:`evaluate_onnx` and
    :func:`evaluate_pytorch` — keeps both backends honest about
    metric definitions, file layout, and output schema.

    Parameters
    ----------
    predictor
        Anything implementing the Predictor protocol (ONNX classifier,
        PyTorch wrapper, or future variant).
    model_path
        Path to the deployed-binary file. Used for the on-disk size
        report; the predictor itself doesn't need to live there.
    system_type
        Sanity-checked against the predictor's own ``system_type``.
    variant_name
        Short label written into the summary JSON / Markdown so the
        comparison step can identify this variant.
    out_dir
        Output directory; created if missing.
    """

    if not model_path.exists():
        raise FileNotFoundError(f"Model artefact not found: {model_path}")
    if predictor.system_type is not system_type:
        raise ValueError(
            f"predictor system_type={predictor.system_type} disagrees with "
            f"requested system_type={system_type}"
        )
    ensure_dir(out_dir)

    x_split, y_split = _load_split_arrays(processed_dir, splits_dir, system_type, split)
    log.info(
        "evaluation_split_loaded",
        extra={
            "system_type": system_type.value,
            "split": split.value,
            "variant": variant_name,
            "n_samples": int(x_split.shape[0]),
            "window_size": int(x_split.shape[1]),
            "in_channels": int(x_split.shape[2]),
        },
    )

    predictions = _run_predictions(predictor, x_split, y_split)
    classification = build_classification_report(
        predictions, system_type=system_type.value, split=split.value
    )

    cm = compute_confusion_matrix(predictions)
    confusion_png_path = out_dir / "confusion_matrix.png"
    render_confusion_matrix_png(
        cm,
        predictor.label_classes,
        output_path=confusion_png_path,
        title=(
            f"{system_type.value} ({variant_name}) confusion matrix "
            f"(split={split.value}, n={predictions.n_samples})"
        ),
        normalise=True,
    )

    latency = benchmark_latency(
        lambda batch: predictor.run_logits(batch),
        window_size=int(x_split.shape[1]),
        in_channels=int(x_split.shape[2]),
        n_runs=n_latency_runs,
        n_warmup=n_latency_warmup,
        batch_size=1,
        seed=latency_seed,
        extra={
            "system_type": system_type.value,
            "variant": variant_name,
            "model_path": str(model_path),
        },
    )

    size = measure_model_size(model_path, budget_mib=size_budget_mib)

    summary_payload: dict[str, Any] = {
        "system_type": system_type.value,
        "variant": variant_name,
        "split": split.value,
        "model_path": str(model_path),
        "classification_report": classification.to_json(),
        "confusion_matrix": confusion_matrix_to_dict(cm, predictor.label_classes),
        "latency": latency.to_json(),
        "model_size": size.to_json(),
    }
    if extra_summary:
        summary_payload.update(extra_summary)

    summary_json_path = out_dir / "summary.json"
    summary_json_path.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary_markdown_path = out_dir / "summary.md"
    _write_summary_markdown(
        out_path=summary_markdown_path,
        classification=classification,
        latency=latency,
        size=size,
        system_type=system_type,
        model_path=model_path,
        variant_name=variant_name,
        confusion_png_path=confusion_png_path,
    )

    log.info(
        "evaluation_done",
        extra={
            "system_type": system_type.value,
            "variant": variant_name,
            "macro_f1": round(classification.macro_f1, 4),
            "p95_ms": round(latency.p95_ms, 4),
            "size_mib": round(size.mib, 4),
        },
    )
    return EvaluationArtifacts(
        summary_json_path=summary_json_path,
        summary_markdown_path=summary_markdown_path,
        confusion_matrix_png_path=confusion_png_path,
        classification_report=classification,
        latency=latency,
        model_size=size,
    )


def evaluate_onnx(
    *,
    onnx_path: Path,
    system_type: SystemType,
    out_dir: Path | None = None,
    variant_name: str = "onnx_fp32",
    split: SplitName = SplitName.TEST,
    processed_dir: Path = PROCESSED_DIR,
    splits_dir: Path = SPLITS_DIR,
    n_latency_runs: int = 1000,
    n_latency_warmup: int = 50,
    latency_seed: int = 42,
    size_budget_mib: float = 50.0,
) -> EvaluationArtifacts:
    """Run Component 3 evaluation on an ONNX classifier (FP32 or INT8).

    Thin wrapper around :func:`evaluate_predictor` that constructs an
    :class:`OnnxClassifier`. Variant labelling is the caller's
    responsibility (default ``"onnx_fp32"``); pass
    ``variant_name="onnx_int8"`` when evaluating an INT8 artefact so
    the comparison step can tell the variants apart.
    """

    classifier = OnnxClassifier(onnx_path)
    if out_dir is None:
        out_dir = REPORTS_DIR / system_type.value.lower()
    return evaluate_predictor(
        predictor=classifier,
        model_path=onnx_path,
        system_type=system_type,
        variant_name=variant_name,
        out_dir=out_dir,
        split=split,
        processed_dir=processed_dir,
        splits_dir=splits_dir,
        n_latency_runs=n_latency_runs,
        n_latency_warmup=n_latency_warmup,
        latency_seed=latency_seed,
        size_budget_mib=size_budget_mib,
    )


def evaluate_pytorch(
    *,
    checkpoint_path: Path,
    system_type: SystemType,
    out_dir: Path | None = None,
    variant_name: str = "pytorch_fp32",
    split: SplitName = SplitName.TEST,
    processed_dir: Path = PROCESSED_DIR,
    splits_dir: Path = SPLITS_DIR,
    n_latency_runs: int = 1000,
    n_latency_warmup: int = 50,
    latency_seed: int = 42,
    size_budget_mib: float = 50.0,
    device: str = "cpu",
) -> EvaluationArtifacts:
    """Run Component 3 evaluation on a PyTorch ``.pt`` checkpoint.

    Used as the FP32 *baseline* in the §4.3 multi-variant comparison.
    The reported model size is the **state-dict-only** byte size, not
    the full training checkpoint (which carries optimizer + history),
    so it is comparable to the ONNX FP32 / INT8 deployed-binary sizes.
    """

    # Lazy import to avoid pulling torch into evaluation.predictor.
    from evaluation.pytorch_runner import (
        PyTorchClassifier,
        measure_pytorch_state_dict_bytes,
    )

    if out_dir is None:
        out_dir = REPORTS_DIR / system_type.value.lower()

    predictor = PyTorchClassifier(checkpoint_path, device=device)

    # Materialise state-dict-only bytes into a sibling .weights.pt file
    # so :func:`measure_model_size` has a real path to stat. This also
    # gives operators a deployable binary for comparison purposes.
    weights_path = checkpoint_path.with_suffix(".weights.pt")
    state_bytes = measure_pytorch_state_dict_bytes(checkpoint_path)
    import torch
    state_dict_only = torch.load(
        str(checkpoint_path), map_location=device, weights_only=False
    )["model_state_dict"]
    torch.save(state_dict_only, str(weights_path))
    actual_bytes = weights_path.stat().st_size
    if actual_bytes != state_bytes:
        # Should be exactly equal (we used the same serialiser); log
        # the discrepancy if any deserialisation quirk shows up.
        log.warning(
            "pytorch_weight_size_mismatch",
            extra={"in_memory": state_bytes, "on_disk": actual_bytes},
        )

    return evaluate_predictor(
        predictor=predictor,
        model_path=weights_path,
        system_type=system_type,
        variant_name=variant_name,
        out_dir=out_dir,
        split=split,
        processed_dir=processed_dir,
        splits_dir=splits_dir,
        n_latency_runs=n_latency_runs,
        n_latency_warmup=n_latency_warmup,
        latency_seed=latency_seed,
        size_budget_mib=size_budget_mib,
        extra_summary={
            "pytorch_full_checkpoint_path": str(checkpoint_path),
            "pytorch_full_checkpoint_bytes": int(checkpoint_path.stat().st_size),
        },
    )
