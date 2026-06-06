"""Tool abstraction for AgentPV agents (project rule §11).

Every tool MUST:

* declare typed input and output Pydantic models;
* run with a timeout;
* return a structured :class:`api.schemas.ToolError` envelope on failure
  (never raise).

The :class:`ToolError` contract lives in :mod:`api.schemas` per project rule
§3 (single source of truth for shared contracts) and is re-exported here for
convenience.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from api.schemas import ToolError
from utils.logging_config import get_logger

__all__ = ["Tool", "ToolError"]

log = get_logger(__name__)

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)


class Tool(ABC, Generic[InputT, OutputT]):
    """Abstract base class for every AgentPV tool.

    Subclasses set the class-level attributes :attr:`name`, :attr:`description`,
    :attr:`input_model`, :attr:`output_model` and implement :meth:`_run`.
    """

    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    timeout_s: float = 5.0
    max_retries: int = 1

    @abstractmethod
    async def _run(self, inp: InputT) -> OutputT:
        """Execute the tool. Subclasses implement this method."""

    async def __call__(self, raw_input: dict[str, Any]) -> dict[str, Any]:
        """Validate, execute, and serialize.

        Returns the output model as a dict on success, or a :class:`ToolError`
        dict on failure. Never raises.
        """
        trace_id = uuid.uuid4().hex

        try:
            inp = self.input_model.model_validate(raw_input)  # type: ignore[assignment]
        except Exception as exc:  # broad: validation can raise many subclasses
            log.warning(
                "tool_validation_failed",
                extra={"tool": self.name, "trace_id": trace_id, "error": str(exc)},
            )
            return ToolError(
                error_code="VALIDATION",
                message=str(exc),
                tool_name=self.name,
                trace_id=trace_id,
            ).model_dump()

        last_error = ""
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                result = await asyncio.wait_for(
                    self._run(inp),  # type: ignore[arg-type]
                    timeout=self.timeout_s,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                log.info(
                    "tool_ok",
                    extra={
                        "tool": self.name,
                        "trace_id": trace_id,
                        "attempt": attempt,
                        "elapsed_ms": elapsed_ms,
                    },
                )
                return result.model_dump()
            except TimeoutError:
                last_error = f"timeout after {self.timeout_s}s"
                log.warning(
                    "tool_timeout",
                    extra={
                        "tool": self.name,
                        "trace_id": trace_id,
                        "attempt": attempt,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                log.exception(
                    "tool_internal_error",
                    extra={
                        "tool": self.name,
                        "trace_id": trace_id,
                        "attempt": attempt,
                    },
                )
                return ToolError(
                    error_code="INTERNAL",
                    message=str(exc),
                    tool_name=self.name,
                    trace_id=trace_id,
                ).model_dump()

        return ToolError(
            error_code="TIMEOUT",
            message=last_error,
            tool_name=self.name,
            trace_id=trace_id,
        ).model_dump()
