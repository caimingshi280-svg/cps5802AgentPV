"""Export a trained PyTorch checkpoint to a self-contained ONNX file.

Self-contained means the resulting ``.onnx`` file:

* applies per-channel input standardization **inside the graph**
  (so any deployment can feed raw ``SensorWindow.values`` directly,
  without bundling a separate ``feature_stats.json``);
* records the label taxonomy in the model's ``metadata_props`` so
  downstream tooling can verify it matches the request's ``system_type``.

Usage
-----
::

    python -m quantization.onnx_export \
        --checkpoint quantization/artifacts/cnn1d_pv_best.pt \
        --output    quantization/artifacts/cnn1d_pv.onnx \
        --opset 17

Constraints (project rule §17 — edge inference budget):

* Single-sample latency < 100 ms on CPU
* Model file size < 50 MB
* Quantization-friendly architecture (no recurrent / attention)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn

from models.cnn1d import CNN1D
from training.data import FeatureStats
from utils.logging_config import get_logger
from utils.paths import ARTIFACTS_DIR, ensure_dir

log = get_logger(__name__)


class StandardizingClassifier(nn.Module):
    """Wrap a classifier so that ``(x - mean) / std`` is part of its forward.

    The mean / std are registered as buffers, which causes ONNX export to
    serialize them as graph constants (foldable into the first Conv).
    Downstream callers feed raw sensor values; no Python-side preprocessing
    is required.
    """

    def __init__(self, model: nn.Module, feature_stats: FeatureStats) -> None:
        super().__init__()
        self.model = model
        self.register_buffer(
            "mean",
            torch.from_numpy(np.asarray(feature_stats.mean, dtype=np.float32)),
        )
        self.register_buffer(
            "std",
            torch.from_numpy(np.asarray(feature_stats.std, dtype=np.float32)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) raw sensor values; broadcast (F,) buffers across (B, T).
        return self.model((x - self.mean) / self.std)


def _build_model_from_checkpoint(payload: dict) -> StandardizingClassifier:
    """Reconstruct the wrapped model from a checkpoint payload."""

    arch = payload.get("model_arch", "CNN1D")
    if arch != "CNN1D":
        raise ValueError(f"Unsupported model_arch in checkpoint: {arch!r}")

    in_channels = int(payload.get("in_channels", 8))
    n_classes = int(payload["n_classes"])
    dropout = float(payload.get("dropout", 0.30))
    base = CNN1D(in_channels=in_channels, n_classes=n_classes, dropout=dropout)
    base.load_state_dict(payload["model_state_dict"])
    base.eval()

    if "feature_stats" not in payload:
        raise KeyError(
            "Checkpoint missing 'feature_stats'; re-train with the current "
            "training/train.py to embed standardization in the ONNX graph."
        )
    feature_stats = FeatureStats.from_dict(payload["feature_stats"])

    wrapped = StandardizingClassifier(base, feature_stats).eval()
    return wrapped


def _attach_metadata(
    onnx_path: Path,
    *,
    label_classes: list[str],
    system_type: str,
    in_channels: int,
    val_macro_f1: float | None,
    feature_stats: FeatureStats,
) -> None:
    """Attach label taxonomy and provenance to the ONNX file's metadata_props."""

    model_proto = onnx.load(str(onnx_path))
    metadata = {
        "agentpv.system_type": system_type,
        "agentpv.label_classes": json.dumps(label_classes),
        "agentpv.in_channels": str(in_channels),
        "agentpv.feature_stats": json.dumps(feature_stats.to_dict()),
        "agentpv.training_val_macro_f1": (
            "" if val_macro_f1 is None else f"{val_macro_f1:.6f}"
        ),
        "agentpv.opset_version": "17",
    }
    # 移除已有同名 entry，再追加新版本，避免重复。
    existing_keys = {prop.key for prop in model_proto.metadata_props}
    keep = [prop for prop in model_proto.metadata_props if prop.key not in metadata]
    del model_proto.metadata_props[:]
    for prop in keep:
        model_proto.metadata_props.append(prop)
    for k, v in metadata.items():
        prop = model_proto.metadata_props.add()
        prop.key = k
        prop.value = v
    log.info(
        "onnx_metadata_attached",
        extra={
            "preserved_existing_keys": sorted(existing_keys - set(metadata.keys())),
            "added_keys": sorted(metadata.keys()),
        },
    )
    onnx.save(model_proto, str(onnx_path))


def export_checkpoint(
    *,
    checkpoint_path: Path,
    output_path: Path,
    opset: int = 17,
    sample_window_size: int = 60,
) -> Path:
    """Convert ``.pt`` to a self-contained ``.onnx`` file."""

    payload = torch.load(
        str(checkpoint_path), weights_only=False, map_location="cpu"
    )
    log.info(
        "checkpoint_loaded",
        extra={
            "path": str(checkpoint_path),
            "epoch": payload.get("epoch"),
            "val_macro_f1": payload.get("val_macro_f1"),
        },
    )

    wrapped = _build_model_from_checkpoint(payload)
    in_channels = int(payload.get("in_channels", 8))
    label_classes = list(payload["label_classes"])
    system_type = str(payload["system_type"])
    feature_stats = FeatureStats.from_dict(payload["feature_stats"])

    # Dummy input matches SensorWindow.values: (B, T, F).
    dummy = torch.randn(1, sample_window_size, in_channels)

    ensure_dir(output_path.parent)
    # 用 TorchScript exporter (dynamo=False)：更轻、不依赖 onnxscript。
    # PyTorch 2.5+ 默认会切到 dynamo exporter；这里显式回退到稳定路径。
    torch.onnx.export(
        wrapped,
        dummy,
        str(output_path),
        input_names=["sensor_window"],
        output_names=["logits"],
        dynamic_axes={
            "sensor_window": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    log.info("onnx_exported", extra={"path": str(output_path), "opset": opset})

    _attach_metadata(
        output_path,
        label_classes=label_classes,
        system_type=system_type,
        in_channels=in_channels,
        val_macro_f1=payload.get("val_macro_f1"),
        feature_stats=feature_stats,
    )

    onnx.checker.check_model(str(output_path))
    log.info("onnx_validated", extra={"path": str(output_path)})
    return output_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="agentpv-onnx-export")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to <ARTIFACTS_DIR>/<checkpoint stem>.onnx",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--window-size", type=int, default=60)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    output = args.output or ARTIFACTS_DIR / f"{args.checkpoint.stem}.onnx"
    out = export_checkpoint(
        checkpoint_path=args.checkpoint,
        output_path=output,
        opset=args.opset,
        sample_window_size=args.window_size,
    )
    print(json.dumps({"output": str(out)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
