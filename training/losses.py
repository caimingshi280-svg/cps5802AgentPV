"""Loss functions for AgentPV classifier training.

For the MVP we expose two well-understood options:

* :class:`WeightedCrossEntropyLoss` — standard CE with optional class weights
  (sufficient for moderate imbalance).
* :class:`FocalLoss` — for *severe* imbalance or hard-example mining; reserved
  for the polish phase.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812 — `F` is the PyTorch convention


class WeightedCrossEntropyLoss(nn.Module):
    """Thin wrapper around :class:`nn.CrossEntropyLoss` for a uniform API."""

    def __init__(self, class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, targets)


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017) for severe class imbalance.

    L = -α * (1 - p_t)^γ * log(p_t)

    Use only when class imbalance can't be addressed by sampler/weights.
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        focal = (1.0 - pt) ** self.gamma * (-log_pt)
        if self.alpha is not None:
            alpha_t = self.alpha.to(focal.device)[targets]
            focal = alpha_t * focal
        return focal.mean()
