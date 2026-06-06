"""Append-only JSONL writer for orchestrator events.

The dashboard tails this file (or reads it on demand) to render history.
Component 5 agent benchmark replays it for offline evaluation.

Design constraints:

* **Append-only.** Never rewrite earlier lines; downstream consumers can
  use ``stat().st_size`` as an idempotent cursor.
* **One JSON object per line.** Easy to ``cat | jq`` and to load with
  ``pandas.read_json(path, lines=True)``.
* **Atomic per-line write.** We open the file in append mode and rely on
  the OS for atomicity of single ``write()`` calls smaller than PIPE_BUF.
"""
from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

from api.schemas import OrchestratorEvent
from utils.logging_config import get_logger

log = get_logger(__name__)


class JsonlEventWriter:
    """Thread/asyncio-safe append-only JSONL writer.

    The lock guards against interleaved writes when multiple node coroutines
    flush simultaneously inside the same event loop.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Reset / create the file so each orchestrator run starts fresh.
        # If you want to preserve cross-run history, the polish phase should
        # rotate this file by timestamp instead of overwriting.
        self.path.touch(exist_ok=True)
        self._lock = Lock()

    def append(self, event: OrchestratorEvent) -> None:
        line = event.model_dump_json(by_alias=False, exclude_none=False) + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()

    def read_all(self) -> list[OrchestratorEvent]:
        """Convenience: read every event written so far (used by tests)."""

        if not self.path.exists():
            return []
        events: list[OrchestratorEvent] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                if not raw.strip():
                    continue
                events.append(OrchestratorEvent.model_validate_json(raw))
        return events

    def truncate(self) -> None:
        """Reset the log file. Used between test runs."""

        with self._lock:
            self.path.write_text("", encoding="utf-8")


def make_default_path(out_dir: Path | None = None) -> Path:
    """Default path for orchestrator events: ``data/orchestrator/events.jsonl``."""

    from utils.paths import DATA_DIR

    if out_dir is None:
        out_dir = DATA_DIR / "orchestrator"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "events.jsonl"


def summarize(events: list[OrchestratorEvent]) -> dict[str, Any]:
    """Compute a flat summary used by the orchestrator's status reporter."""

    n_total = len(events)
    n_alerts = sum(1 for e in events if e.alert is not None)
    n_recommendations = sum(1 for e in events if e.recommendation is not None)
    n_errors = sum(1 for e in events if e.error is not None)
    by_severity: dict[str, int] = {}
    by_fault_class: dict[str, int] = {}
    for e in events:
        if e.alert is None:
            continue
        sev = e.alert.severity.value
        by_severity[sev] = by_severity.get(sev, 0) + 1
        by_fault_class[e.alert.fault_class] = by_fault_class.get(e.alert.fault_class, 0) + 1
    return {
        "n_total": n_total,
        "n_alerts": n_alerts,
        "n_recommendations": n_recommendations,
        "n_errors": n_errors,
        "by_severity": by_severity,
        "by_fault_class": by_fault_class,
    }
