"""PyTorch baseline predictor for Component 3 multi-variant evaluation.

This wraps a ``.pt`` checkpoint produced by :mod:`training.train` so
that it satisfies the :class:`evaluation.predictor.Predictor` protocol
(matching the ONNX backend). The point is to verify that the ONNX
export pipeline did **not** introduce accuracy regression and to give
the assignment §4.3 "compare ≥ 2 variants" requirement a true
reference baseline.

Notes
-----
* Standardisation is applied **inside** :meth:`run_logits` using the
  ``feature_stats`` saved in the checkpoint, mirroring what the
  self-contained ONNX graph does. Callers therefore feed raw window
  values exactly as they do for the ONNX backend.
* ``model.eval()`` is enforced once at construction so BatchNorm uses
  running statistics (the bug that bit S07's numerical-parity test).
* Inference runs under ``torch.inference_mode()`` so we don't pay
  autograd cost when benchmarking latency.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from api.schemas import SystemType
from models.cnn1d import CNN1D
from training.data import FeatureStats
from utils.logging_config import get_logger

log = get_logger(__name__)


class PyTorchClassifier:
    """Run a PyTorch ``.pt`` checkpoint as a Predictor.

    Parameters
    ----------
    checkpoint_path
        Path to a ``.pt`` file written by ``training/trainer.py`` —
        must contain ``model_state_dict``, ``label_classes``,
        ``feature_stats``, ``system_type``, ``in_channels``,
        ``n_classes``, ``model_arch`` (currently only ``"CNN1D"``).
    device
        ``"cpu"`` is enforced unless overridden — rule §17 mandates
        CPU-only edge inference, and we want apples-to-apples
        comparison with the ONNX backend.
    """

    def __init__(self, checkpoint_path: Path, *, device: str = "cpu") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"PyTorch checkpoint missing: {checkpoint_path}")

        payload = torch.load(
            str(self.checkpoint_path), map_location=device, weights_only=False
        )

        # 1) Required fields — fail loudly if they're missing.
        for key in ("model_state_dict", "label_classes", "feature_stats", "system_type"):
            if key not in payload:
                raise KeyError(
                    f"checkpoint at {checkpoint_path} missing required field "
                    f"{key!r}; re-train with the current training/train.py."
                )

        arch = payload.get("model_arch", "CNN1D")
        if arch != "CNN1D":
            raise ValueError(
                f"unsupported model_arch={arch!r} in checkpoint {checkpoint_path}; "
                "extend PyTorchClassifier to dispatch to the new architecture."
            )

        # 2) Build + load the model.
        in_channels = int(payload.get("in_channels", 8))
        n_classes = int(payload["n_classes"])
        dropout = float(payload.get("dropout", 0.30))
        model = CNN1D(in_channels=in_channels, n_classes=n_classes, dropout=dropout)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        self._model = model
        self._device = torch.device(device)

        # 3) Feature stats → torch tensors broadcastable across (B, T, F).
        stats = FeatureStats.from_dict(payload["feature_stats"])
        self._mean = torch.from_numpy(stats.mean.astype(np.float32))
        self._std = torch.from_numpy(stats.std.astype(np.float32))

        # 4) Predictor-protocol attributes.
        self.system_type = SystemType(str(payload["system_type"]))
        self.label_classes: tuple[str, ...] = tuple(payload["label_classes"])
        self.in_channels = in_channels
        self.val_macro_f1 = payload.get("val_macro_f1")  # informational

        log.info(
            "pytorch_classifier_loaded",
            extra={
                "path": str(self.checkpoint_path),
                "system_type": self.system_type.value,
                "n_classes": len(self.label_classes),
                "device": str(self._device),
            },
        )

    def run_logits(self, x: np.ndarray) -> np.ndarray:
        """Run inference and return raw logits.

        Parameters
        ----------
        x
            Input tensor shaped ``(B, T, F)`` of **raw** sensor values
            (no caller-side standardisation; this method handles it).

        Returns
        -------
        np.ndarray
            ``(B, n_classes)`` float32 logits.
        """

        if x.ndim != 3:
            raise ValueError(f"expected (B, T, F), got {x.shape}")
        if x.shape[-1] != self.in_channels:
            raise ValueError(
                f"expected F={self.in_channels} feature channels, got {x.shape[-1]}"
            )

        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        with torch.inference_mode():
            x_t = torch.from_numpy(x).to(self._device)
            x_t = (x_t - self._mean) / self._std
            logits = self._model(x_t)
            return logits.detach().cpu().numpy().astype(np.float32, copy=False)


def measure_pytorch_state_dict_bytes(checkpoint_path: Path) -> int:
    """Return the deployable model-weight size in bytes.

    The training pipeline saves a richer dict (history, optimizer hints,
    feature stats). For a fair *deployed-binary* comparison against
    INT8 ONNX, we re-serialise just ``state_dict`` to a temp buffer
    and measure that.
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"PyTorch checkpoint missing: {checkpoint_path}")
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if "model_state_dict" not in payload:
        raise KeyError(f"checkpoint {checkpoint_path} has no model_state_dict")

    import io

    buf = io.BytesIO()
    torch.save(payload["model_state_dict"], buf)
    return buf.tell()
