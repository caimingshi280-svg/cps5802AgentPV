"""Component 3 multi-variant comparison.

Aggregates per-variant ``summary.json`` files (produced by
:mod:`evaluation.runner`) into one Markdown table + tradeoff plot per
system, satisfying the assignment §4.3 *compare ≥ 2 variants*
requirement.

Public surface
--------------
- :class:`VariantRow`           — one variant's headline numbers
- :class:`VariantComparison`    — per-system aggregate
- :func:`load_variant_summary`  — read / validate one ``summary.json``
- :func:`build_comparison_rows` — pure transform
- :func:`compare_variants`      — orchestrator (writes md + png + json)

Design choices
--------------
1. **Pure-function core, side-effects at the edge** — same pattern as
   :mod:`evaluation.metrics`. ``build_comparison_rows`` is fully
   testable with no I/O.
2. **No averaging across systems** — PV (7 classes) and BESS (5
   classes) classes don't align; we report per-system tables.
3. **Tradeoff plot uses matplotlib** (lazy imported), the same dep
   added in S11 — no new packages.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from utils.logging_config import get_logger
from utils.paths import PROJECT_ROOT, ensure_dir

log = get_logger(__name__)

REPORTS_DIR = PROJECT_ROOT / "reports"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantRow:
    """Headline metrics for one model variant.

    All numeric fields are rounded only at render time, never at the
    data-class level, so downstream consumers can re-format them.
    """

    variant: str
    macro_f1: float
    accuracy: float
    weighted_f1: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    size_kib: float
    size_mib: float
    n_samples: int
    summary_path: Path

    def to_json(self) -> dict[str, object]:
        d = asdict(self)
        d["summary_path"] = str(self.summary_path)
        return d


@dataclass(frozen=True)
class VariantComparison:
    """One system's worth of compared variants.

    The first variant in ``rows`` is the *baseline* against which deltas
    are calculated. Convention: list ``pytorch_fp32`` first when it's
    present (the closest thing to "ground truth FP32 reference"),
    otherwise the first variant the caller passed.
    """

    system_type: str
    split: str
    rows: tuple[VariantRow, ...]
    baseline_variant: str

    def baseline(self) -> VariantRow:
        for row in self.rows:
            if row.variant == self.baseline_variant:
                return row
        raise KeyError(
            f"baseline_variant={self.baseline_variant!r} missing from rows"
        )

    def to_json(self) -> dict[str, object]:
        return {
            "system_type": self.system_type,
            "split": self.split,
            "baseline_variant": self.baseline_variant,
            "rows": [r.to_json() for r in self.rows],
        }


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def load_variant_summary(summary_path: Path) -> dict[str, object]:
    """Load + minimally validate a per-variant ``summary.json`` file."""

    if not summary_path.exists():
        raise FileNotFoundError(f"variant summary missing: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    required = (
        "system_type",
        "variant",
        "split",
        "classification_report",
        "latency",
        "model_size",
    )
    missing = [k for k in required if k not in payload]
    if missing:
        raise KeyError(
            f"summary.json at {summary_path} missing keys {missing}; "
            "re-run `python -m evaluation` to regenerate."
        )
    return payload


def _row_from_payload(payload: dict[str, object], summary_path: Path) -> VariantRow:
    cls = payload["classification_report"]
    lat = payload["latency"]
    sz = payload["model_size"]
    return VariantRow(
        variant=str(payload["variant"]),
        macro_f1=float(cls["macro_f1"]),
        accuracy=float(cls["accuracy"]),
        weighted_f1=float(cls["weighted_f1"]),
        p50_ms=float(lat["p50_ms"]),
        p95_ms=float(lat["p95_ms"]),
        p99_ms=float(lat["p99_ms"]),
        size_kib=float(sz["kib"]),
        size_mib=float(sz["mib"]),
        n_samples=int(cls["n_samples"]),
        summary_path=summary_path,
    )


def build_comparison_rows(
    summary_paths: list[Path],
    *,
    baseline_variant: str | None = None,
) -> VariantComparison:
    """Pure-function transform: list of summary.json paths → comparison.

    Validates that every summary.json belongs to the same system /
    split. If ``baseline_variant`` is None we pick ``pytorch_fp32``
    when present, otherwise the first variant.
    """

    if not summary_paths:
        raise ValueError("summary_paths must contain ≥ 1 path")

    rows: list[VariantRow] = []
    system_types: set[str] = set()
    splits: set[str] = set()
    for path in summary_paths:
        payload = load_variant_summary(path)
        system_types.add(str(payload["system_type"]))
        splits.add(str(payload["split"]))
        rows.append(_row_from_payload(payload, path))

    if len(system_types) != 1:
        raise ValueError(
            f"all variants must share the same system_type, got {system_types}"
        )
    if len(splits) != 1:
        raise ValueError(f"all variants must share the same split, got {splits}")

    variant_names = [r.variant for r in rows]
    if len(set(variant_names)) != len(variant_names):
        raise ValueError(f"duplicate variant names: {variant_names}")

    if baseline_variant is None:
        baseline_variant = (
            "pytorch_fp32" if "pytorch_fp32" in variant_names else variant_names[0]
        )
    if baseline_variant not in variant_names:
        raise KeyError(
            f"baseline_variant={baseline_variant!r} not among {variant_names}"
        )

    return VariantComparison(
        system_type=next(iter(system_types)),
        split=next(iter(splits)),
        rows=tuple(rows),
        baseline_variant=baseline_variant,
    )


def comparison_to_markdown(comparison: VariantComparison) -> str:
    """Render the headline comparison table as Markdown."""

    base = comparison.baseline()
    header = (
        "| Variant | Macro-F1 | Δ vs base | Acc. | p50 (ms) | p95 (ms) | p99 (ms) | Size (KiB) | Size (MiB) | Δ size |"
        "\n"
        "|---|---|---|---|---|---|---|---|---|---|"
    )
    body_rows: list[str] = []
    for row in comparison.rows:
        delta_f1 = row.macro_f1 - base.macro_f1
        size_ratio = (
            base.size_kib / row.size_kib if row.size_kib > 0 else float("nan")
        )
        delta_f1_str = (
            "—"
            if row.variant == comparison.baseline_variant
            else f"{delta_f1:+.4f}"
        )
        size_ratio_str = (
            "—"
            if row.variant == comparison.baseline_variant
            else f"×{size_ratio:.2f}"
        )
        body_rows.append(
            "| "
            + " | ".join(
                [
                    f"`{row.variant}`",
                    f"{row.macro_f1:.4f}",
                    delta_f1_str,
                    f"{row.accuracy:.4f}",
                    f"{row.p50_ms:.3f}",
                    f"{row.p95_ms:.3f}",
                    f"{row.p99_ms:.3f}",
                    f"{row.size_kib:.2f}",
                    f"{row.size_mib:.4f}",
                    size_ratio_str,
                ]
            )
            + " |"
        )

    title = (
        f"# AgentPV Component 3 — `{comparison.system_type}` variant comparison\n\n"
        f"- **Split**: `{comparison.split}`\n"
        f"- **Baseline**: `{comparison.baseline_variant}`\n"
        f"- **N samples (test)**: {base.n_samples}\n\n"
    )

    notes = (
        "\n\n**Notes**\n"
        "- *Δ vs base* is `macro_f1(variant) - macro_f1(baseline)` "
        "(positive ⇒ variant is more accurate).\n"
        "- *Δ size* is `baseline_kib / variant_kib` "
        "(higher ⇒ variant is more compressed).\n"
        "- Latency is single-sample p50/p95/p99 in ms over ≥ 1000 timed CPU "
        "runs (Component 2 hard target: p95 ≤ 100 ms).\n"
        "- Size budget per Component 2: ≤ 50 MiB on disk.\n"
    )

    return title + header + "\n" + "\n".join(body_rows) + notes


def render_tradeoff_png(comparison: VariantComparison, output_path: Path) -> Path:
    """Render an accuracy-vs-size and accuracy-vs-latency tradeoff plot.

    Uses matplotlib (lazy import to keep startup fast). The plot is a
    1×2 grid: left = Macro-F1 vs size (KiB), right = Macro-F1 vs p95
    latency.
    """

    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    sizes = [r.size_kib for r in comparison.rows]
    p95s = [r.p95_ms for r in comparison.rows]
    f1s = [r.macro_f1 for r in comparison.rows]
    labels = [r.variant for r in comparison.rows]

    for ax, xs, xlabel, title in [
        (axes[0], sizes, "Model size on disk (KiB)", "Macro-F1 vs Size"),
        (axes[1], p95s, "p95 latency (ms, 1 sample, CPU)", "Macro-F1 vs p95 Latency"),
    ]:
        ax.scatter(xs, f1s, s=120, alpha=0.85, edgecolors="black")
        for x, y, label in zip(xs, f1s, labels, strict=True):
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(8, 6),
                fontsize=9,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Macro-F1 (test split)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"AgentPV Component 3 — {comparison.system_type} variant tradeoff",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info(
        "comparison_tradeoff_png_written",
        extra={
            "path": str(output_path),
            "system_type": comparison.system_type,
            "n_variants": len(comparison.rows),
        },
    )
    return output_path


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------


def compare_variants(
    *,
    summary_paths: list[Path],
    out_dir: Path,
    baseline_variant: str | None = None,
) -> tuple[Path, Path, Path]:
    """End-to-end: read variant summaries, write comparison artefacts.

    Returns a tuple ``(comparison_md, comparison_json, tradeoff_png)``.
    """

    ensure_dir(out_dir)
    comparison = build_comparison_rows(summary_paths, baseline_variant=baseline_variant)

    md_path = out_dir / "comparison.md"
    md_path.write_text(comparison_to_markdown(comparison), encoding="utf-8")

    json_path = out_dir / "comparison.json"
    json_path.write_text(
        json.dumps(comparison.to_json(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    png_path = out_dir / "comparison_tradeoff.png"
    render_tradeoff_png(comparison, png_path)

    log.info(
        "comparison_written",
        extra={
            "system_type": comparison.system_type,
            "out_dir": str(out_dir),
            "n_variants": len(comparison.rows),
            "baseline_variant": comparison.baseline_variant,
        },
    )
    return md_path, json_path, png_path
