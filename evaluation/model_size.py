"""Model size accounting (assignment §4.3 + §4.2).

A model artefact's *deployed* size on disk is what matters at the edge —
not parameter count. We measure the file size in bytes and convert into
KB / MB. The dataclass also carries a ``budget_mb`` so the runner can
mark a clear pass / fail in the report.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSizeReport:
    """Disk size metrics for a single model artefact."""

    path: str
    bytes: int
    kib: float
    mib: float
    budget_mib: float
    within_budget: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "bytes": self.bytes,
            "kib": round(self.kib, 4),
            "mib": round(self.mib, 4),
            "budget_mib": round(self.budget_mib, 4),
            "within_budget": self.within_budget,
        }


def measure_model_size(model_path: Path, *, budget_mib: float = 50.0) -> ModelSizeReport:
    """Stat ``model_path`` and return a :class:`ModelSizeReport`.

    Raises
    ------
    FileNotFoundError
        If ``model_path`` does not exist (let the caller handle it; we
        do not silently report 0 bytes).
    """

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if budget_mib <= 0:
        raise ValueError(f"budget_mib must be positive, got {budget_mib}")
    n_bytes = model_path.stat().st_size
    kib = n_bytes / 1024.0
    mib = kib / 1024.0
    return ModelSizeReport(
        path=str(model_path),
        bytes=int(n_bytes),
        kib=kib,
        mib=mib,
        budget_mib=budget_mib,
        within_budget=mib <= budget_mib,
    )
