"""Predictor protocol used by :mod:`evaluation.runner`.

Both ONNX and PyTorch backends implement this protocol so the
evaluation runner can iterate over arbitrary "model variants" without
caring about the underlying runtime. This is the abstraction Component
3 needs to compare ≥ 2 variants per assignment §4.3.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from api.schemas import SystemType


@runtime_checkable
class Predictor(Protocol):
    """A minimal interface every model variant must expose.

    Attributes
    ----------
    system_type
        Whether this predictor classifies PV or BESS windows.
    label_classes
        Ordered class names — matches the integer labels produced by
        :meth:`run_logits`.
    in_channels
        Expected feature-channel count of input windows.

    Methods
    -------
    run_logits(x)
        Forward an ``(B, T, F)`` raw float32 batch and return
        ``(B, n_classes)`` float32 logits. Pre-processing
        (standardisation) is the predictor's responsibility — every
        variant should mimic the deployed contract where the caller
        feeds raw sensor readings.
    """

    system_type: SystemType
    label_classes: tuple[str, ...]
    in_channels: int

    def run_logits(self, x: np.ndarray) -> np.ndarray: ...
