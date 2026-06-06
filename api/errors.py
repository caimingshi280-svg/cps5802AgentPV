"""Helpers for constructing structured API error responses (project rule §12)."""
from __future__ import annotations

import uuid

from api.schemas import ErrorResponse


def make_error(
    error_code: str,
    message: str,
    *,
    trace_id: str | None = None,
    retry_after_s: int | None = None,
) -> ErrorResponse:
    """Build an :class:`ErrorResponse`, generating a trace_id when missing.

    Standard ``error_code`` values used across services:

    * ``VALIDATION``       — input failed Pydantic validation
    * ``MODEL_LOAD_FAIL``  — ONNX model could not be loaded
    * ``TIMEOUT``          — downstream call exceeded its budget
    * ``DOWNSTREAM_DOWN``  — required upstream service is unavailable
    * ``INTERNAL``         — unhandled exception
    """
    return ErrorResponse(
        error_code=error_code,
        message=message,
        trace_id=trace_id or uuid.uuid4().hex,
        retry_after_s=retry_after_s,
    )
