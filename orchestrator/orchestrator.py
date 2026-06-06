"""High-level orchestrator: drive N :class:`NodeRunner`s to completion.

Responsibilities:

* Own the shared :class:`JsonlEventWriter` that all nodes append to.
* Spawn one ``asyncio.Task`` per node and wait for them.
* Provide a :meth:`stop` method (signal handler / test driver entry).
* Provide a flat :meth:`summary` snapshot for the dashboard / CLI.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from orchestrator.clients import AgentClient, EdgeClient
from orchestrator.event_log import JsonlEventWriter, summarize
from orchestrator.node_simulator import NodeConfig, NodeRunner
from utils.logging_config import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class OrchestratorConfig:
    """Top-level configuration."""

    nodes: tuple[NodeConfig, ...]
    duration_seconds: float | None = None  # None → run until stop() called


class Orchestrator:
    """Manage many :class:`NodeRunner`s under a single asyncio event loop."""

    def __init__(
        self,
        config: OrchestratorConfig,
        edge: EdgeClient,
        agent: AgentClient,
        writer: JsonlEventWriter,
    ) -> None:
        if not config.nodes:
            raise ValueError("Orchestrator requires at least one node")
        self.config = config
        self.edge = edge
        self.agent = agent
        self.writer = writer
        self._stop = asyncio.Event()
        self.runners: list[NodeRunner] = [
            NodeRunner(cfg, edge=edge, agent=agent, writer=writer)
            for cfg in config.nodes
        ]
        self._tasks: list[asyncio.Task[Any]] = []

    async def run(self) -> None:
        """Start all node loops; wait for duration or external stop."""

        log.info(
            "orchestrator_starting",
            extra={
                "n_nodes": len(self.runners),
                "duration_s": self.config.duration_seconds,
            },
        )
        loop = asyncio.get_event_loop()
        del loop  # not used directly; documenting that we stay on the caller's loop
        self._tasks = [
            asyncio.create_task(
                runner.run_forever(self._stop),
                name=f"node-{runner.config.node_id}",
            )
            for runner in self.runners
        ]
        if self.config.duration_seconds is not None:
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=self.config.duration_seconds
                )
            except TimeoutError:
                pass
            await self.stop()
        else:
            await self._stop.wait()
            await self._stop_tasks()
        log.info(
            "orchestrator_stopped",
            extra={"summary": self.summary()},
        )

    async def stop(self) -> None:
        """Signal all node loops to exit; wait for them."""

        if self._stop.is_set():
            return
        self._stop.set()
        await self._stop_tasks()

    async def _stop_tasks(self) -> None:
        # Wait for the natural exit of each node loop.
        await asyncio.gather(*self._tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # Read-only views (used by CLI + tests)
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        per_node = [
            {
                "node_id": r.config.node_id,
                "system_id": r.config.system_id,
                "system_type": r.config.system_type.value,
                "n_steps": r.state.step_number,
                "n_alerts": r.state.n_alerts,
                "n_recommendations": r.state.n_recommendations,
                "n_errors": r.state.n_errors,
            }
            for r in self.runners
        ]
        events = self.writer.read_all()
        global_summary = summarize(events)
        return {
            "n_nodes": len(self.runners),
            "per_node": per_node,
            "global": global_summary,
        }
