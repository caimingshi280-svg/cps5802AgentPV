"""Timing utilities for benchmarks and structured logs.

中文说明
--------
``stopwatch`` 上下文用于记录推理耗时等；日志中输出毫秒级耗时便于与作业
P95 延迟指标对照。
"""
from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import ParamSpec, TypeVar

from utils.logging_config import get_logger

P = ParamSpec("P")
R = TypeVar("R")
log = get_logger(__name__)


@contextmanager
def stopwatch(label: str) -> Iterator[dict[str, float]]:
    """Context manager that measures elapsed wall time in milliseconds.

    Example::

        with stopwatch("predict") as result:
            run_inference(window)
        print(result["elapsed_ms"])
    """
    bucket: dict[str, float] = {"elapsed_ms": 0.0}
    start = time.perf_counter()
    try:
        yield bucket
    finally:
        bucket["elapsed_ms"] = (time.perf_counter() - start) * 1000.0
        log.debug(
            "stopwatch",
            extra={"label": label, "elapsed_ms": bucket["elapsed_ms"]},
        )


def timed(label: str | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that logs the elapsed wall time of a synchronous function."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        name = label or func.__qualname__

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            t0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                log.debug(
                    "timed",
                    extra={
                        "label": name,
                        "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
                    },
                )

        return wrapper

    return decorator
