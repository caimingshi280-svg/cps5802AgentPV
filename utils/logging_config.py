"""Structured logging configuration for AgentPV (project rule §4).

Use ``get_logger(__name__)`` everywhere; never call :func:`print`.

中文说明
--------
统一控制台输出格式（文本或 JSON）。业务代码只取 logger，不在库外重复配置
``basicConfig``，避免多进程/多 uvicorn worker 下日志错乱。
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from configs.settings import settings


class JsonFormatter(logging.Formatter):
    """Render log records as one-line JSON objects (suitable for shipping)."""

    _RESERVED: frozenset[str] = frozenset(
        {
            "args", "asctime", "created", "exc_info", "exc_text", "filename",
            "funcName", "levelname", "levelno", "lineno", "message", "module",
            "msecs", "msg", "name", "pathname", "process", "processName",
            "relativeCreated", "stack_info", "thread", "threadName", "taskName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key in self._RESERVED or key.startswith("_"):
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)
        return json.dumps(payload, ensure_ascii=False)


def _build_handler() -> logging.Handler:
    handler = logging.StreamHandler(stream=sys.stdout)
    if settings.log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s :: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
    return handler


_INITIALIZED: bool = False


def configure_logging() -> None:
    """Idempotently configure the root logger using current ``settings``."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    root = logging.getLogger()
    root.setLevel(settings.log_level.upper())
    root.handlers.clear()
    root.addHandler(_build_handler())
    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    configure_logging()
    return logging.getLogger(name)
