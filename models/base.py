"""Abstract base class for AgentPV time-series classifiers.

Per project rule §2 (modular design) and §7 (time-series model design),
every classifier MUST:

* declare ``in_channels`` (number of sensor features per time step) and
  ``n_classes`` (output dimensionality) as instance attributes;
* expect input shape ``(B, T, F)`` and emit logits of shape ``(B, n_classes)``;
* report parameter count and disk footprint.

Concrete architectures (CNN1D, LSTMClassifier, ...) live in sibling files.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseClassifier(nn.Module, ABC):
    """Common interface for time-series classifiers."""

    in_channels: int
    n_classes: int

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape ``(B, n_classes)``.

        Subclasses must accept ``x`` of shape ``(B, T, F)``. Internal
        transposition to PyTorch's ``(B, F, T)`` Conv1d convention is the
        subclass's responsibility — keep the public contract uniform.
        """

    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
