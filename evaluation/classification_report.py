"""Per-class + aggregate classification report (assignment §4.3).

Produces a structured dataclass that can be serialised to JSON and a
human-readable Markdown table. Targeted at PV / BESS classifiers but is
fully system-agnostic — it accepts any :class:`EvaluationPredictions`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from evaluation.metrics import (
    EvaluationPredictions,
    PerClassRow,
    accuracy,
    macro_f1,
    per_class_metrics,
    to_dict,
)


@dataclass(frozen=True)
class ClassificationReport:
    """Aggregate + per-class metrics for a single split.

    Attributes mirror the fields scikit-learn's ``classification_report``
    text output produces, but the type is stable and JSON-serialisable so
    downstream tooling (Markdown writer, dashboard, agent_eval) can rely
    on it without parsing free text.
    """

    system_type: str  # "PV" / "BESS"
    split: str  # "train" / "val" / "test"
    n_samples: int
    n_classes: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: list[PerClassRow] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """Return a plain-dict representation suitable for ``json.dump``."""

        d = asdict(self)
        d["per_class"] = to_dict(self.per_class)
        # round floats for stable, human-friendly JSON
        d["accuracy"] = round(d["accuracy"], 6)
        d["macro_f1"] = round(d["macro_f1"], 6)
        d["weighted_f1"] = round(d["weighted_f1"], 6)
        return d

    def to_markdown(self) -> str:
        """Render a Markdown table of per-class + summary metrics."""

        lines: list[str] = []
        lines.append(
            f"### {self.system_type} classification report — split=`{self.split}` "
            f"(N={self.n_samples}, n_classes={self.n_classes})"
        )
        lines.append("")
        lines.append("| Class | Precision | Recall | F1 | Support |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in self.per_class:
            lines.append(
                f"| `{row.label}` | "
                f"{row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} | {row.support} |"
            )
        lines.append("")
        lines.append("| Aggregate | Value |")
        lines.append("|---|---:|")
        lines.append(f"| Accuracy | {self.accuracy:.4f} |")
        lines.append(f"| Macro-F1 | **{self.macro_f1:.4f}** |")
        lines.append(f"| Weighted-F1 | {self.weighted_f1:.4f} |")
        return "\n".join(lines)


def _weighted_f1(per_class: list[PerClassRow]) -> float:
    """Support-weighted average F1.

    Returns 0.0 when total support is 0 (empty split) — keeps callers
    branch-free on edge cases.
    """

    total_support = sum(r.support for r in per_class)
    if total_support == 0:
        return 0.0
    weighted = sum(r.f1 * r.support for r in per_class)
    return float(weighted / total_support)


def build_classification_report(
    predictions: EvaluationPredictions,
    *,
    system_type: str,
    split: str,
) -> ClassificationReport:
    """Compute every metric and pack into a :class:`ClassificationReport`."""

    rows = per_class_metrics(predictions)
    return ClassificationReport(
        system_type=system_type,
        split=split,
        n_samples=predictions.n_samples,
        n_classes=predictions.n_classes,
        accuracy=accuracy(predictions),
        macro_f1=macro_f1(predictions),
        weighted_f1=_weighted_f1(rows),
        per_class=rows,
    )
