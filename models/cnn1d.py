"""1D CNN classifier for AgentPV time-series.

Architecture rationale (project rule §7):

* **Input**: ``(B, T=60, F=8)`` sensor windows from :mod:`simulation`.
* **Why 1D CNN over LSTM/Transformer**:
    - Faults exhibit local temporal patterns (oscillations, step changes,
      slow drifts) that 1D convolutions extract well with few parameters.
    - Trains on CPU in seconds; inference < 5 ms — comfortable headroom
      under the 100 ms latency budget.
    - Quantization- and ONNX-friendly (no recurrent state, no attention).
* **Trade-off**: cannot model very long dependencies. Acceptable here
  because windows are 60 s long and faults are detectable within that span.

Tensor shapes (defended during oral exam — keep this comment up to date):

::

    x:                          (B,  60,  8)         input window
    .transpose(1, 2):           (B,   8, 60)         channels-first for Conv1d
    Conv1d(8 -> 32, k=5, p=2):  (B,  32, 60)
    BatchNorm1d + ReLU:         (B,  32, 60)
    Conv1d(32 -> 64, k=5, p=2): (B,  64, 60)
    BatchNorm1d + ReLU:         (B,  64, 60)
    MaxPool1d(2):               (B,  64, 30)
    Conv1d(64 -> 128, k=3, p=1):(B, 128, 30)
    BatchNorm1d + ReLU:         (B, 128, 30)
    AdaptiveAvgPool1d(1):       (B, 128,  1)
    Flatten:                    (B, 128)
    Linear(128 -> 64):          (B,  64)
    Dropout(p):                 (B,  64)
    Linear(64 -> n_classes):    (B,  n_classes)
"""
from __future__ import annotations

import torch
from torch import nn

from models.base import BaseClassifier


class CNN1D(BaseClassifier):
    """Compact 1D CNN time-series classifier."""

    def __init__(
        self,
        in_channels: int = 8,
        n_classes: int = 7,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        # 三层卷积负责局部时间特征；AdaptiveAvgPool1d(1) 把变长序列收成全局表征。
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Sensor windows shaped ``(B, T, F)``. The internal ``transpose``
            converts to Conv1d's expected ``(B, F, T)`` channels-first layout.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, n_classes)``.
        """
        if x.ndim != 3:
            raise ValueError(
                f"expected 3D input (B, T, F), got shape {tuple(x.shape)}"
            )
        if x.shape[-1] != self.in_channels:
            raise ValueError(
                f"expected F={self.in_channels} feature channels, "
                f"got F={x.shape[-1]}"
            )
        x = x.transpose(1, 2)  # (B, T, F) -> (B, F, T)
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits
