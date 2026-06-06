"""Helper: count Ollama / ReAct telemetry signals in an agent_eval log file."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "agent_eval" / "results" / "last_run_ollama.log"
    )
    # PowerShell ``Tee-Object`` writes UTF-16LE on Windows by default; try
    # UTF-8 first, fall back to UTF-16 when the body contains a BOM.
    raw = log_path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = raw.decode("utf-16", errors="ignore")
    else:
        text = raw.decode("utf-8", errors="ignore")
    counts = {
        "ollama_http_calls": len(re.findall(r"POST http://localhost:11434/api/chat", text)),
        "plan_fallback_warnings": len(re.findall(r"ollama_plan_fallback_mock", text)),
        "tool_validation_warnings": len(re.findall(r"tool_validation_failed", text)),
        "react_completed_events": len(re.findall(r"react_completed", text)),
        "alerts_escalated": len(re.findall(r"alert_escalated", text)),
        "log_path": str(log_path),
        "log_bytes": log_path.stat().st_size,
    }
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
