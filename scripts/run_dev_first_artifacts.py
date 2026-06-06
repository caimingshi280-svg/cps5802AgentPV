"""Produce first dev-oriented artifacts for reports (agent_eval + agent ASGI smoke).

Run from repo root::

    python scripts/run_dev_first_artifacts.py

Behaviour:

* ``APP_ENV=dev`` so paths / defaults match development overlay.
* Probes Ollama at ``AGENTPV_OLLAMA_BASE_URL`` (default ``http://127.0.0.1:11434``).
  If unreachable, sets ``AGENTPV_LLM_BACKEND=mock`` so the run still completes
  and the report documents the fallback.
* Writes a **3-scenario** mini benchmark to ``reports/dev_smoke_benchmark.json``,
  runs ``python -m agent_eval`` with ``full`` + ``no_reasoning_trace`` ablations,
  and saves JSON/Markdown under ``reports/``.
* Hits ``api.agent_service`` in-process via :class:`fastapi.testclient.TestClient`
  once (``/healthz`` + ``/recommend``) and writes ``reports/dev_agent_asgi_smoke.json``.

This script clears ``get_settings`` cache before importing the FastAPI app.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REPORTS = ROOT / "reports"
BENCH_SRC = ROOT / "agent_eval" / "benchmark.json"
BENCH_MINI = REPORTS / "dev_smoke_benchmark.json"
OUT_JSON = REPORTS / "dev_first_agent_eval.json"
OUT_MD = REPORTS / "dev_first_agent_eval.md"
META = REPORTS / "dev_run_meta.json"
ASGI_OUT = REPORTS / "dev_agent_asgi_smoke.json"


def _ollama_reachable(base: str) -> bool:
    base = base.rstrip("/")
    url = f"{base}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=2.0) as r:  # noqa: S310
            return r.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def _write_mini_benchmark(n: int = 3) -> None:
    raw = json.loads(BENCH_SRC.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) < n:
        raise RuntimeError("benchmark.json missing or too short")
    REPORTS.mkdir(parents=True, exist_ok=True)
    BENCH_MINI.write_text(
        json.dumps(raw[:n], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _asgi_smoke(llm_backend: str) -> None:
    os.environ["APP_ENV"] = "dev"
    os.environ["AGENTPV_LLM_BACKEND"] = llm_backend
    import importlib

    import configs.settings as cfg

    cfg.get_settings.cache_clear()
    cfg.settings = cfg.get_settings()

    from fastapi.testclient import TestClient

    import api.agent_service as agent_mod

    agent_mod = importlib.reload(agent_mod)
    alert = {
        "timestamp": datetime(2026, 5, 12, 12, 0, 0, tzinfo=UTC).isoformat(),
        "system_id": "PV-DEV-SMOKE-001",
        "system_type": "PV",
        "fault_class": "Inverter_fault",
        "severity": "critical",
        "confidence": 0.95,
        "sensor_snapshot": {"P_dc": 250.0, "P_ac": 0.0},
    }
    with TestClient(agent_mod.app) as client:
        hz = client.get("/healthz")
        rec = client.post("/recommend", json=alert)
    payload = {
        "healthz": {"status_code": hz.status_code, "body": hz.json()},
        "recommend": {
            "status_code": rec.status_code,
            "body": rec.json() if rec.headers.get("content-type", "").startswith("application/json") else rec.text[:2000],
        },
        "llm_backend": llm_backend,
    }
    ASGI_OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    if not BENCH_SRC.is_file():
        print("missing agent_eval/benchmark.json", file=sys.stderr)
        return 1

    _write_mini_benchmark(3)

    os.environ["APP_ENV"] = "dev"
    ollama_base = os.environ.get("AGENTPV_OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_ok = _ollama_reachable(ollama_base)
    llm_backend = "ollama" if ollama_ok else "mock"
    os.environ["AGENTPV_LLM_BACKEND"] = llm_backend

    import configs.settings as cfg

    cfg.get_settings.cache_clear()
    s = cfg.get_settings()

    REPORTS.mkdir(parents=True, exist_ok=True)
    META.write_text(
        json.dumps(
            {
                "app_env": "dev",
                "ollama_reachable": ollama_ok,
                "ollama_base_url": ollama_base,
                "llm_backend_effective": llm_backend,
                "note": (
                    "Ollama was used for agent_eval + ASGI smoke."
                    if ollama_ok
                    else "Ollama unreachable — LLM backend forced to mock for reproducible artifacts; start Ollama and re-run."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        "-m",
        "agent_eval",
        "--benchmark",
        str(BENCH_MINI),
        "--kb-dir",
        str(s.knowledge_base_dir),
        "--ablations",
        "full",
        "no_reasoning_trace",
        "--no-llm-judge",
        "--out-json",
        str(OUT_JSON),
        "--out-md",
        str(OUT_MD),
    ]
    env = os.environ.copy()
    env["APP_ENV"] = "dev"
    env["AGENTPV_LLM_BACKEND"] = llm_backend
    print("running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    if proc.returncode != 0:
        return proc.returncode

    _asgi_smoke(llm_backend)
    print(json.dumps({"wrote": [str(META), str(OUT_JSON), str(OUT_MD), str(ASGI_OUT)]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
