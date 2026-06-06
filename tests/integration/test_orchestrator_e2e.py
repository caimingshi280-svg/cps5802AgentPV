"""End-to-end integration test for the orchestrator.

We mount both FastAPI services in-process via :class:`httpx.ASGITransport`
and drive them with the real :class:`Orchestrator`. This validates the
contract chain ``SensorWindow → Alert → Recommendation`` without binding
real sockets.

These tests require:

* ONNX artefacts under ``quantization/artifacts/`` (produced in S07).
* A knowledge base directory (we point the agent service at a tmp one).

If artefacts are missing the test is skipped rather than failing — that
keeps the suite green on a fresh clone before a model is exported.
"""
from __future__ import annotations

import importlib

import httpx
import pytest

from configs.settings import get_settings
from orchestrator.clients import AgentClient, EdgeClient
from orchestrator.event_log import JsonlEventWriter
from orchestrator.node_simulator import NodeConfig
from orchestrator.orchestrator import Orchestrator, OrchestratorConfig
from utils.paths import ARTIFACTS_DIR

pytestmark = pytest.mark.skipif(
    not (ARTIFACTS_DIR / "cnn1d_pv.onnx").exists(),
    reason="ONNX artefacts not built; run S07 export first.",
)


@pytest.fixture
def kb_dir(tmp_path):
    docs = tmp_path / "kb"
    docs.mkdir()
    (docs / "placeholder_inverter.md").write_text(
        "# Inverter Fault (PLACEHOLDER)\n\n## Action\nDispatch within 4h.\n",
        encoding="utf-8",
    )
    (docs / "placeholder_safety.md").write_text(
        "# General Safety (PLACEHOLDER)\n\nLOTO before DC work.\n", encoding="utf-8"
    )
    return docs


@pytest.fixture
def edge_app():
    get_settings.cache_clear()  # type: ignore[attr-defined]
    from api import edge_service

    return importlib.reload(edge_service).app


@pytest.fixture
def agent_app(kb_dir, monkeypatch):
    monkeypatch.setenv("AGENTPV_KNOWLEDGE_BASE_DIR", str(kb_dir))
    get_settings.cache_clear()  # type: ignore[attr-defined]
    from api import agent_service

    return importlib.reload(agent_service).app


@pytest.mark.asyncio
async def test_orchestrator_drives_edge_and_agent_in_process(
    edge_app, agent_app, tmp_path
) -> None:
    """A 2-node, 0.6s run must produce ≥2 events and at least one Alert."""

    async with edge_app.router.lifespan_context(edge_app), agent_app.router.lifespan_context(agent_app):
        edge_http = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=edge_app),
            base_url="http://edge.test",
        )
        agent_http = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=agent_app),
            base_url="http://agent.test",
        )
        try:
            edge = EdgeClient(edge_http)
            agent = AgentClient(agent_http)
            writer = JsonlEventWriter(tmp_path / "events.jsonl")
            cfg = OrchestratorConfig(
                nodes=(
                    NodeConfig(
                        node_id="pv-it-1",
                        system_id="PV-IT-001",
                        system_type="PV",
                        seed=1,
                        fault_probability=1.0,  # 强制每步都 inject 故障
                        period_seconds=0.10,
                    ),
                    NodeConfig(
                        node_id="bess-it-1",
                        system_id="BESS-IT-001",
                        system_type="BESS",
                        seed=2,
                        fault_probability=1.0,
                        period_seconds=0.15,
                    ),
                ),
                duration_seconds=0.6,
            )
            orch = Orchestrator(cfg, edge=edge, agent=agent, writer=writer)
            await orch.run()
            summary = orch.summary()
        finally:
            await edge_http.aclose()
            await agent_http.aclose()

    # 至少每个节点跑了一步
    assert summary["n_nodes"] == 2
    for per in summary["per_node"]:
        assert per["n_steps"] >= 1

    events = writer.read_all()
    assert len(events) >= 2
    # 至少有一个 alert（任意 severity）
    n_alerts = sum(1 for e in events if e.alert is not None)
    assert n_alerts >= 1
    # 全部 fault_probability=1.0 → ground_truth 不会是 *_Normal
    for e in events:
        assert not e.ground_truth_label.endswith("_Normal")
