"""Integration tests for the FastAPI agent service.

We point the service at a tmp knowledge base via ``AGENTPV_KNOWLEDGE_BASE_DIR``
to avoid coupling tests to the on-disk knowledge base contents (which can
change without breaking the API contract).
"""
from __future__ import annotations

import importlib
from datetime import UTC, datetime

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from configs.settings import get_settings  # noqa: E402


def _alert_payload(severity: str = "critical", fault_class: str = "Inverter_fault") -> dict:
    return {
        "timestamp": datetime(2026, 5, 9, 12, 0, 0, tzinfo=UTC).isoformat(),
        "system_id": "PV_test_001",
        "system_type": "PV",
        "fault_class": fault_class,
        "severity": severity,
        "confidence": 0.95,
        "sensor_snapshot": {"P_dc": 250.0, "P_ac": 0.0, "eta": 0.0},
    }


@pytest.fixture
def kb_dir(tmp_path):
    """Tmp knowledge base with a couple of placeholder docs."""

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "placeholder_inverter.md").write_text(
        "# Inverter Fault Playbook (PLACEHOLDER)\n\n"
        "## Symptoms\nP_ac collapses to zero while P_dc remains.\n\n"
        "## Action\nDispatch technician within 4 hours.\n",
        encoding="utf-8",
    )
    (docs / "placeholder_safety.md").write_text(
        "# General Safety (PLACEHOLDER)\n\n"
        "Always LOTO before DC-side intervention.\n",
        encoding="utf-8",
    )
    return docs


@pytest.fixture
def app(kb_dir, monkeypatch):
    monkeypatch.setenv("AGENTPV_KNOWLEDGE_BASE_DIR", str(kb_dir))
    get_settings.cache_clear()  # type: ignore[attr-defined]
    from api import agent_service

    return importlib.reload(agent_service).app


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /healthz
# ---------------------------------------------------------------------------


def test_healthz_ok_when_kb_loaded(client) -> None:
    resp = client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"status": "ok", "service": "agent", "version": "0.1.0"}


# ---------------------------------------------------------------------------
# /recommend
# ---------------------------------------------------------------------------


def test_recommend_returns_structured_recommendation(client) -> None:
    resp = client.post("/recommend", json=_alert_payload(severity="critical"))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["urgency"] == "immediate"
    assert body["recommended_action"].startswith("[MOCK]")
    assert isinstance(body["reasoning_trace"], list)
    assert len(body["reasoning_trace"]) >= 5  # observe + reason + acts + reflect + report
    phases = [s["phase"] for s in body["reasoning_trace"]]
    assert phases[0] == "observe"
    assert phases[-1] == "report"


def test_recommend_critical_has_high_confidence_with_sources(client) -> None:
    resp = client.post("/recommend", json=_alert_payload(severity="critical"))
    assert resp.status_code == 200
    body = resp.json()
    assert body["confidence"] == "high"
    assert len(body["knowledge_sources"]) >= 1


def test_recommend_warning_has_lower_urgency(client) -> None:
    resp = client.post(
        "/recommend",
        json=_alert_payload(severity="warning", fault_class="Partial_shading"),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["urgency"] == "scheduled"


def test_recommend_rejects_invalid_alert(client) -> None:
    bad = _alert_payload()
    bad["confidence"] = 2.0  # out of [0, 1]
    resp = client.post("/recommend", json=bad)
    assert resp.status_code == 422


def test_recommend_rejects_extra_fields(client) -> None:
    bad = _alert_payload()
    bad["unexpected_key"] = "x"
    resp = client.post("/recommend", json=bad)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Degraded path: missing knowledge base
# ---------------------------------------------------------------------------


@pytest.fixture
def app_no_kb(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTPV_KNOWLEDGE_BASE_DIR", str(tmp_path / "missing"))
    get_settings.cache_clear()  # type: ignore[attr-defined]
    from api import agent_service

    return importlib.reload(agent_service).app


def test_healthz_degraded_when_kb_missing(app_no_kb) -> None:
    with TestClient(app_no_kb) as c:
        resp = c.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "degraded"


def test_recommend_returns_503_when_agent_uninitialized(app_no_kb) -> None:
    with TestClient(app_no_kb) as c:
        resp = c.post("/recommend", json=_alert_payload())
        assert resp.status_code == 503
