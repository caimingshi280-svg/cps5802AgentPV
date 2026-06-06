"""Unit tests for the evaluation/ package (Component 3).

Each numeric kernel is tested independently with synthetic predictions —
no model loading, no I/O — so the suite stays under a second. The
runner integration test is exercised separately in
``tests/integration/test_evaluation_e2e.py``.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from evaluation.classification_report import (
    ClassificationReport,
    build_classification_report,
)
from evaluation.confusion_matrix import (
    compute_confusion_matrix,
    confusion_matrix_to_dict,
    render_confusion_matrix_png,
)
from evaluation.latency_benchmark import benchmark_latency
from evaluation.metrics import (
    EvaluationPredictions,
    accuracy,
    macro_f1,
    per_class_metrics,
    predictions_from_arrays,
    to_dict,
)
from evaluation.model_size import measure_model_size

# ---------------------------------------------------------------------------
# EvaluationPredictions validation
# ---------------------------------------------------------------------------


def test_evaluation_predictions_validates_shapes() -> None:
    with pytest.raises(ValueError, match="must be 1-D"):
        EvaluationPredictions(
            y_true=np.zeros((2, 2), dtype=np.int64),
            y_pred=np.zeros((4,), dtype=np.int64),
            label_classes=("a", "b"),
        )


def test_evaluation_predictions_validates_lengths() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        EvaluationPredictions(
            y_true=np.array([0, 1], dtype=np.int64),
            y_pred=np.array([0], dtype=np.int64),
            label_classes=("a", "b"),
        )


def test_evaluation_predictions_validates_label_range() -> None:
    with pytest.raises(ValueError, match="outside"):
        predictions_from_arrays([0, 5], [0, 0], label_classes=("a", "b", "c"))


def test_evaluation_predictions_validates_non_empty_labels() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        EvaluationPredictions(
            y_true=np.array([], dtype=np.int64),
            y_pred=np.array([], dtype=np.int64),
            label_classes=(),
        )


# ---------------------------------------------------------------------------
# accuracy / macro_f1 / per_class_metrics
# ---------------------------------------------------------------------------


def test_accuracy_perfect_predictions() -> None:
    p = predictions_from_arrays([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2], ("a", "b", "c"))
    assert accuracy(p) == 1.0
    assert macro_f1(p) == 1.0


def test_accuracy_empty_split_returns_zero() -> None:
    p = predictions_from_arrays([], [], ("a", "b"))
    assert accuracy(p) == 0.0
    assert macro_f1(p) == 0.0


def test_macro_f1_penalises_missing_class() -> None:
    """Class 2 has support but the model never predicts it → recall=0 → macro F1 < 1."""

    p = predictions_from_arrays([0, 1, 2], [0, 1, 0], ("a", "b", "c"))
    rows = per_class_metrics(p)
    by_label = {r.label: r for r in rows}
    assert by_label["c"].recall == 0.0
    assert by_label["c"].f1 == 0.0
    assert macro_f1(p) < 1.0  # 0.0 contribution from class c


def test_per_class_metrics_returns_one_row_per_class_in_order() -> None:
    """Even classes the model never sees still appear (with metrics 0)."""

    p = predictions_from_arrays([0, 0, 0], [0, 0, 0], ("a", "b", "c"))
    rows = per_class_metrics(p)
    assert [r.label for r in rows] == ["a", "b", "c"]
    assert rows[0].precision == 1.0
    assert rows[1].precision == 0.0
    assert rows[1].support == 0


def test_to_dict_rounds_floats() -> None:
    p = predictions_from_arrays([0, 1, 1], [0, 0, 1], ("a", "b"))
    rows = per_class_metrics(p)
    d = to_dict(rows)
    for r in d:
        for f in ("precision", "recall", "f1"):
            assert isinstance(r[f], float)


# ---------------------------------------------------------------------------
# ClassificationReport
# ---------------------------------------------------------------------------


def test_build_classification_report_aggregates_correctly() -> None:
    p = predictions_from_arrays([0, 0, 1, 1, 2, 2], [0, 0, 1, 0, 2, 2], ("a", "b", "c"))
    report = build_classification_report(p, system_type="PV", split="test")
    assert isinstance(report, ClassificationReport)
    assert report.n_samples == 6
    assert report.n_classes == 3
    assert 0.7 < report.accuracy < 1.0  # 5/6
    assert report.macro_f1 < 1.0
    assert report.weighted_f1 > 0.0


def test_classification_report_to_json_round_trip_serialisable() -> None:
    p = predictions_from_arrays([0, 1], [0, 1], ("x", "y"))
    report = build_classification_report(p, system_type="BESS", split="val")
    payload = report.to_json()
    # 必须能够无错通过 json.dumps（catch np.float64 / np.int64 等问题）
    json.dumps(payload)
    assert payload["system_type"] == "BESS"
    assert payload["accuracy"] == 1.0
    assert len(payload["per_class"]) == 2


def test_classification_report_to_markdown_includes_aggregate_block() -> None:
    p = predictions_from_arrays([0, 1, 1], [0, 1, 1], ("a", "b"))
    report = build_classification_report(p, system_type="PV", split="test")
    md = report.to_markdown()
    assert "| Accuracy |" in md
    assert "| Macro-F1 |" in md
    assert "| Weighted-F1 |" in md
    assert "Precision" in md


def test_weighted_f1_is_zero_on_empty_support() -> None:
    p = predictions_from_arrays([], [], ("a", "b"))
    report = build_classification_report(p, system_type="PV", split="test")
    assert report.weighted_f1 == 0.0


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def test_compute_confusion_matrix_shape_and_counts() -> None:
    p = predictions_from_arrays([0, 1, 2, 1, 0], [0, 1, 1, 1, 0], ("a", "b", "c"))
    cm = compute_confusion_matrix(p)
    assert cm.shape == (3, 3)
    # row 0 (true=a): 2/2 predicted as a; row 1 (true=b): 2/2 as b; row 2: 1 as b
    assert cm[0, 0] == 2
    assert cm[1, 1] == 2
    assert cm[2, 1] == 1


def test_compute_confusion_matrix_empty_returns_zero_matrix() -> None:
    p = predictions_from_arrays([], [], ("a", "b", "c"))
    cm = compute_confusion_matrix(p)
    assert cm.shape == (3, 3)
    assert cm.sum() == 0


def test_confusion_matrix_to_dict_rejects_shape_mismatch() -> None:
    cm = np.zeros((3, 3), dtype=np.int64)
    with pytest.raises(ValueError, match="doesn't match"):
        confusion_matrix_to_dict(cm, ("a", "b"))


def test_render_confusion_matrix_png_writes_file(tmp_path) -> None:
    p = predictions_from_arrays([0, 1, 1], [0, 1, 0], ("a", "b"))
    cm = compute_confusion_matrix(p)
    out = tmp_path / "sub" / "cm.png"
    rendered = render_confusion_matrix_png(
        cm, ("a", "b"), output_path=out, title="test"
    )
    assert rendered == out
    assert out.exists()
    assert out.stat().st_size > 1000  # 真 PNG 至少 1 KB


def test_render_confusion_matrix_png_with_normalise_off(tmp_path) -> None:
    cm = np.array([[3, 0], [1, 1]], dtype=np.int64)
    out = tmp_path / "cm.png"
    render_confusion_matrix_png(cm, ("a", "b"), output_path=out, normalise=False)
    assert out.exists()


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------


def test_benchmark_latency_returns_stats_and_calls_predict_n_times() -> None:
    n_calls = {"count": 0}

    def fake_predict(x: np.ndarray) -> np.ndarray:
        n_calls["count"] += 1
        return x.sum(axis=1, keepdims=True)

    res = benchmark_latency(
        fake_predict,
        window_size=10,
        in_channels=4,
        n_runs=20,
        n_warmup=3,
        batch_size=1,
    )
    # warmup + timed
    assert n_calls["count"] == 23
    assert res.n_runs == 20
    assert res.n_warmup == 3
    assert res.batch_size == 1
    assert res.window_size == 10
    assert res.in_channels == 4
    assert res.mean_ms >= 0.0
    assert res.p95_ms >= res.p50_ms >= res.min_ms
    assert res.max_ms >= res.p99_ms


def test_benchmark_latency_extra_field_propagates() -> None:
    res = benchmark_latency(
        lambda x: x,
        window_size=4,
        in_channels=2,
        n_runs=5,
        n_warmup=1,
        extra={"system_type": "PV"},
    )
    assert res.extra == {"system_type": "PV"}
    payload = res.to_json()
    assert payload["extra"]["system_type"] == "PV"


def test_benchmark_latency_validates_inputs() -> None:
    with pytest.raises(ValueError, match="n_runs"):
        benchmark_latency(lambda x: x, window_size=1, in_channels=1, n_runs=0)
    with pytest.raises(ValueError, match="n_warmup"):
        benchmark_latency(lambda x: x, window_size=1, in_channels=1, n_warmup=-1)
    with pytest.raises(ValueError, match="positive"):
        benchmark_latency(lambda x: x, window_size=0, in_channels=1)


# ---------------------------------------------------------------------------
# Model size
# ---------------------------------------------------------------------------


def test_measure_model_size_within_budget(tmp_path) -> None:
    p = tmp_path / "model.bin"
    p.write_bytes(b"x" * 1024)  # 1 KiB
    rep = measure_model_size(p, budget_mib=50.0)
    assert rep.bytes == 1024
    assert pytest.approx(rep.kib, rel=1e-6) == 1.0
    assert rep.within_budget


def test_measure_model_size_over_budget(tmp_path) -> None:
    p = tmp_path / "model.bin"
    p.write_bytes(b"x" * 1024)
    rep = measure_model_size(p, budget_mib=0.0005)  # ~0.5 KiB
    assert not rep.within_budget


def test_measure_model_size_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        measure_model_size(tmp_path / "no.bin")


def test_measure_model_size_rejects_bad_budget(tmp_path) -> None:
    p = tmp_path / "model.bin"
    p.write_bytes(b"abc")
    with pytest.raises(ValueError, match="positive"):
        measure_model_size(p, budget_mib=0.0)
