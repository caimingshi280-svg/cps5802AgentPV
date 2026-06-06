"""Unit tests for :mod:`evaluation.compare_variants`.

All tests work on synthetic ``summary.json`` payloads written into
``tmp_path``, so they don't touch the real ``reports/`` dir or any
trained model. Pure-function transforms get the most thorough coverage
since they're the API surface external code depends on.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from evaluation.compare_variants import (
    VariantComparison,
    VariantRow,
    build_comparison_rows,
    compare_variants,
    comparison_to_markdown,
    load_variant_summary,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_summary(
    *,
    system_type: str,
    variant: str,
    macro_f1: float,
    p95_ms: float,
    size_kib: float,
    split: str = "test",
    accuracy: float | None = None,
    weighted_f1: float | None = None,
    n_samples: int = 1000,
) -> dict:
    """Build a minimal summary.json payload mirroring runner output."""

    return {
        "system_type": system_type,
        "variant": variant,
        "split": split,
        "model_path": f"/fake/{variant}.bin",
        "classification_report": {
            "system_type": system_type,
            "split": split,
            "n_samples": n_samples,
            "accuracy": accuracy if accuracy is not None else macro_f1,
            "macro_f1": macro_f1,
            "weighted_f1": (
                weighted_f1 if weighted_f1 is not None else macro_f1
            ),
            "per_class": [],
        },
        "confusion_matrix": {"labels": ["a", "b"], "matrix": [[1, 0], [0, 1]]},
        "latency": {
            "p50_ms": p95_ms * 0.5,
            "p95_ms": p95_ms,
            "p99_ms": p95_ms * 1.2,
            "mean_ms": p95_ms * 0.6,
            "min_ms": p95_ms * 0.3,
            "max_ms": p95_ms * 1.5,
            "std_ms": p95_ms * 0.1,
            "n_runs": 1000,
            "n_warmup": 50,
            "extra": {},
        },
        "model_size": {
            "path": f"/fake/{variant}.bin",
            "bytes": int(size_kib * 1024),
            "kib": size_kib,
            "mib": size_kib / 1024.0,
            "budget_mib": 50.0,
            "within_budget": True,
        },
    }


def _write(tmp_path: Path, name: str, payload: dict) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# load_variant_summary
# ---------------------------------------------------------------------------


def test_load_variant_summary_round_trips(tmp_path: Path) -> None:
    payload = _make_summary(
        system_type="PV", variant="onnx_fp32", macro_f1=0.99, p95_ms=0.2, size_kib=180
    )
    path = _write(tmp_path, "fp32.json", payload)
    loaded = load_variant_summary(path)
    assert loaded["variant"] == "onnx_fp32"
    assert loaded["system_type"] == "PV"


def test_load_variant_summary_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_variant_summary(tmp_path / "nope.json")


def test_load_variant_summary_rejects_missing_keys(tmp_path: Path) -> None:
    payload = _make_summary(
        system_type="PV", variant="onnx_fp32", macro_f1=0.9, p95_ms=0.1, size_kib=10
    )
    del payload["latency"]
    path = _write(tmp_path, "broken.json", payload)
    with pytest.raises(KeyError, match="missing keys"):
        load_variant_summary(path)


# ---------------------------------------------------------------------------
# build_comparison_rows
# ---------------------------------------------------------------------------


def test_build_comparison_rows_picks_pytorch_baseline_by_default(
    tmp_path: Path,
) -> None:
    a = _write(
        tmp_path,
        "pt.json",
        _make_summary(
            system_type="PV", variant="pytorch_fp32", macro_f1=0.99, p95_ms=0.2, size_kib=180
        ),
    )
    b = _write(
        tmp_path,
        "onnx.json",
        _make_summary(
            system_type="PV", variant="onnx_fp32", macro_f1=0.99, p95_ms=0.15, size_kib=180
        ),
    )
    c = _write(
        tmp_path,
        "int8.json",
        _make_summary(
            system_type="PV", variant="onnx_int8", macro_f1=0.97, p95_ms=0.13, size_kib=60
        ),
    )

    cmp_ = build_comparison_rows([a, b, c])
    assert isinstance(cmp_, VariantComparison)
    assert cmp_.system_type == "PV"
    assert cmp_.split == "test"
    assert cmp_.baseline_variant == "pytorch_fp32"
    assert tuple(r.variant for r in cmp_.rows) == ("pytorch_fp32", "onnx_fp32", "onnx_int8")


def test_build_comparison_rows_falls_back_to_first_when_no_pytorch(
    tmp_path: Path,
) -> None:
    a = _write(
        tmp_path,
        "fp32.json",
        _make_summary(
            system_type="BESS", variant="onnx_fp32", macro_f1=0.95, p95_ms=0.1, size_kib=180
        ),
    )
    b = _write(
        tmp_path,
        "int8.json",
        _make_summary(
            system_type="BESS", variant="onnx_int8", macro_f1=0.92, p95_ms=0.07, size_kib=60
        ),
    )
    cmp_ = build_comparison_rows([a, b])
    assert cmp_.baseline_variant == "onnx_fp32"


def test_build_comparison_rows_rejects_mixed_systems(tmp_path: Path) -> None:
    a = _write(
        tmp_path,
        "pv.json",
        _make_summary(
            system_type="PV", variant="x", macro_f1=0.9, p95_ms=0.1, size_kib=10
        ),
    )
    b = _write(
        tmp_path,
        "bess.json",
        _make_summary(
            system_type="BESS", variant="y", macro_f1=0.9, p95_ms=0.1, size_kib=10
        ),
    )
    with pytest.raises(ValueError, match="same system_type"):
        build_comparison_rows([a, b])


def test_build_comparison_rows_rejects_mixed_splits(tmp_path: Path) -> None:
    a = _write(
        tmp_path,
        "test.json",
        _make_summary(
            system_type="PV", variant="x", macro_f1=0.9, p95_ms=0.1, size_kib=10, split="test"
        ),
    )
    b = _write(
        tmp_path,
        "val.json",
        _make_summary(
            system_type="PV", variant="y", macro_f1=0.9, p95_ms=0.1, size_kib=10, split="val"
        ),
    )
    with pytest.raises(ValueError, match="same split"):
        build_comparison_rows([a, b])


def test_build_comparison_rows_rejects_duplicate_variant_names(
    tmp_path: Path,
) -> None:
    a = _write(
        tmp_path,
        "a.json",
        _make_summary(
            system_type="PV", variant="onnx_fp32", macro_f1=0.9, p95_ms=0.1, size_kib=10
        ),
    )
    b = _write(
        tmp_path,
        "b.json",
        _make_summary(
            system_type="PV", variant="onnx_fp32", macro_f1=0.91, p95_ms=0.1, size_kib=10
        ),
    )
    with pytest.raises(ValueError, match="duplicate"):
        build_comparison_rows([a, b])


def test_build_comparison_rows_rejects_unknown_baseline(tmp_path: Path) -> None:
    a = _write(
        tmp_path,
        "a.json",
        _make_summary(
            system_type="PV", variant="onnx_fp32", macro_f1=0.9, p95_ms=0.1, size_kib=10
        ),
    )
    with pytest.raises(KeyError, match="not among"):
        build_comparison_rows([a], baseline_variant="missing_variant")


def test_build_comparison_rows_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="≥ 1"):
        build_comparison_rows([])


# ---------------------------------------------------------------------------
# Markdown / VariantRow
# ---------------------------------------------------------------------------


def test_comparison_to_markdown_contains_expected_columns(tmp_path: Path) -> None:
    a = _write(
        tmp_path,
        "fp32.json",
        _make_summary(
            system_type="PV", variant="pytorch_fp32", macro_f1=0.99, p95_ms=0.2, size_kib=180
        ),
    )
    b = _write(
        tmp_path,
        "int8.json",
        _make_summary(
            system_type="PV", variant="onnx_int8", macro_f1=0.97, p95_ms=0.13, size_kib=60
        ),
    )
    cmp_ = build_comparison_rows([a, b])
    md = comparison_to_markdown(cmp_)
    assert "Macro-F1" in md
    assert "Δ vs base" in md
    assert "p95 (ms)" in md
    assert "`pytorch_fp32`" in md
    assert "`onnx_int8`" in md
    # Δ size for the baseline must be the em-dash, not a number.
    assert "—" in md
    # Compression ratio for INT8 should be ≈ ×3.0 (180/60).
    assert "×3.00" in md or "×3.01" in md or "×2.99" in md


def test_variant_row_freeze_and_to_json() -> None:
    fake_path = Path("fake") / "summary.json"
    row = VariantRow(
        variant="onnx_fp32",
        macro_f1=0.99,
        accuracy=0.99,
        weighted_f1=0.99,
        p50_ms=0.1,
        p95_ms=0.2,
        p99_ms=0.3,
        size_kib=180.5,
        size_mib=0.176,
        n_samples=1000,
        summary_path=fake_path,
    )
    assert row.to_json()["summary_path"] == str(fake_path)
    # frozen=True dataclass disallows attribute mutation.
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.macro_f1 = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# End-to-end orchestrator
# ---------------------------------------------------------------------------


def test_compare_variants_writes_md_json_and_png(tmp_path: Path) -> None:
    summaries = [
        _write(
            tmp_path,
            f"{name}.json",
            _make_summary(
                system_type="PV", variant=name, macro_f1=f1, p95_ms=lat, size_kib=sz
            ),
        )
        for name, f1, lat, sz in [
            ("pytorch_fp32", 0.99, 0.2, 180),
            ("onnx_fp32", 0.99, 0.15, 180),
            ("onnx_int8", 0.97, 0.13, 60),
        ]
    ]

    out_dir = tmp_path / "out"
    md_path, json_path, png_path = compare_variants(
        summary_paths=summaries, out_dir=out_dir
    )

    assert md_path.exists() and md_path.read_text(encoding="utf-8")
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["system_type"] == "PV"
    assert payload["baseline_variant"] == "pytorch_fp32"
    assert len(payload["rows"]) == 3
    assert png_path.exists()
    assert png_path.stat().st_size > 0
