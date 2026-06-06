"""CLI entry point for the orchestrator.

Usage examples
--------------

Run 3 nodes (2 PV + 1 BESS) for 30 s, writing events to the default JSONL:

    python -m orchestrator --nodes pv2_bess1 --duration 30

Custom edge / agent base URLs:

    python -m orchestrator --edge http://edge:8000 --agent http://agent:8001 \
        --nodes pv2_bess1 --duration 60

The MVP picks node configs from a small built-in catalogue rather than a
YAML file (rule §27). Polish phase will add ``configs/orchestrator.yaml``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx

from api.schemas import OperatingCondition, SystemType
from orchestrator.clients import AgentClient, EdgeClient
from orchestrator.event_log import JsonlEventWriter, make_default_path
from orchestrator.node_simulator import NodeConfig
from orchestrator.orchestrator import Orchestrator, OrchestratorConfig
from utils.logging_config import get_logger

log = get_logger("orchestrator")


_PV6_BESS4_SPECS: tuple[
    tuple[str, str, SystemType, int, float, float, OperatingCondition, str], ...
] = (
    # (node_id,    system_id,        system_type, seed, fault_p, period_s, op_condition,             integration_mode)
    ("pv-001",  "PV-FARM-001",  SystemType.PV,   11, 0.30, 0.6, OperatingCondition.HIGH_IRRADIANCE,  "full"),
    ("pv-002",  "PV-FARM-002",  SystemType.PV,   22, 0.50, 0.7, OperatingCondition.HIGH_TEMPERATURE, "full"),
    ("pv-003",  "PV-FARM-003",  SystemType.PV,   31, 0.40, 0.8, OperatingCondition.LOW_IRRADIANCE,   "full"),
    ("pv-004",  "PV-FARM-004",  SystemType.PV,   47, 0.20, 1.0, OperatingCondition.HIGH_IRRADIANCE,  "edge_only"),
    ("pv-005",  "PV-FARM-005",  SystemType.PV,   53, 0.35, 0.9, OperatingCondition.HIGH_TEMPERATURE, "full"),
    ("pv-006",  "PV-FARM-006",  SystemType.PV,   67, 0.45, 1.1, OperatingCondition.LOW_IRRADIANCE,   "cloud_only"),
    ("bess-001", "BESS-RACK-001", SystemType.BESS, 33, 0.40, 0.8, OperatingCondition.HIGH_IRRADIANCE,  "full"),
    ("bess-002", "BESS-RACK-002", SystemType.BESS, 41, 0.30, 0.9, OperatingCondition.HIGH_TEMPERATURE, "full"),
    ("bess-003", "BESS-RACK-003", SystemType.BESS, 59, 0.50, 1.0, OperatingCondition.HIGH_IRRADIANCE,  "edge_only"),
    ("bess-004", "BESS-RACK-004", SystemType.BESS, 71, 0.25, 1.2, OperatingCondition.LOW_IRRADIANCE,   "cloud_only"),
)


def _catalog(name: str) -> tuple[NodeConfig, ...]:
    """Built-in node-set catalogues.

    Naming convention: ``pv<N>_bess<M>``. The presentation/integration runs
    consume :data:`_PV6_BESS4_SPECS` (10 nodes, all three integration modes
    represented) so a single orchestrator session can demonstrate the
    Component 6 graceful-degradation ablation.
    """

    if name == "pv6_bess4":
        return tuple(
            NodeConfig(
                node_id=spec[0],
                system_id=spec[1],
                system_type=spec[2],
                seed=spec[3],
                fault_probability=spec[4],
                period_seconds=spec[5],
                operating_condition=spec[6],
                integration_mode=spec[7],  # type: ignore[arg-type]
            )
            for spec in _PV6_BESS4_SPECS
        )
    if name == "pv2_bess1":
        return (
            NodeConfig(
                node_id="pv-001",
                system_id="PV-FARM-001",
                system_type=SystemType.PV,
                seed=11,
                fault_probability=0.30,
                period_seconds=1.0,
                operating_condition=OperatingCondition.HIGH_IRRADIANCE,
            ),
            NodeConfig(
                node_id="pv-002",
                system_id="PV-FARM-002",
                system_type=SystemType.PV,
                seed=22,
                fault_probability=0.50,
                period_seconds=1.5,
                operating_condition=OperatingCondition.HIGH_TEMPERATURE,
            ),
            NodeConfig(
                node_id="bess-001",
                system_id="BESS-RACK-001",
                system_type=SystemType.BESS,
                seed=33,
                fault_probability=0.40,
                period_seconds=2.0,
                operating_condition=OperatingCondition.HIGH_IRRADIANCE,
            ),
        )
    if name == "minimal":
        return (
            NodeConfig(
                node_id="pv-mini",
                system_id="PV-MINI-001",
                system_type=SystemType.PV,
                seed=7,
                fault_probability=0.50,
                period_seconds=1.0,
            ),
        )
    raise ValueError(
        f"Unknown nodes catalogue {name!r}; supported: "
        "['minimal', 'pv2_bess1', 'pv6_bess4']"
    )


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="orchestrator")
    parser.add_argument(
        "--edge",
        default="http://127.0.0.1:8000",
        help="Edge service base URL (default: %(default)s).",
    )
    parser.add_argument(
        "--agent",
        default="http://127.0.0.1:8001",
        help="Agent service base URL (default: %(default)s).",
    )
    parser.add_argument(
        "--nodes",
        choices=("minimal", "pv2_bess1", "pv6_bess4"),
        default="pv2_bess1",
        help="Built-in node-set name (default: %(default)s).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Total run time in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Path to JSONL event log (default: data/orchestrator/events.jsonl).",
    )
    parser.add_argument(
        "--http-timeout",
        type=float,
        default=10.0,
        help="HTTP request timeout (s) for both edge/agent calls.",
    )
    return parser.parse_args(argv)


async def _run_async(args: argparse.Namespace) -> dict:
    nodes = _catalog(args.nodes)
    out_path = args.out or make_default_path()
    writer = JsonlEventWriter(out_path)
    writer.truncate()  # 每次 CLI 跑都从头记录（rule §6 — 重现性优先）

    timeout = httpx.Timeout(args.http_timeout)
    limits = httpx.Limits(max_connections=32, max_keepalive_connections=16)
    async with (
        httpx.AsyncClient(base_url=args.edge, timeout=timeout, limits=limits) as edge_http,
        httpx.AsyncClient(base_url=args.agent, timeout=timeout, limits=limits) as agent_http,
    ):
        edge = EdgeClient(edge_http)
        agent = AgentClient(agent_http)
        orch = Orchestrator(
            OrchestratorConfig(nodes=nodes, duration_seconds=args.duration),
            edge=edge,
            agent=agent,
            writer=writer,
        )
        log.info(
            "orchestrator_cli_run",
            extra={
                "edge": args.edge,
                "agent": args.agent,
                "nodes": args.nodes,
                "duration_s": args.duration,
                "out": str(out_path),
            },
        )
        await orch.run()
        return orch.summary()


def main(argv: list[str] | None = None) -> None:
    args = _parse(argv)
    summary = asyncio.run(_run_async(args))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
