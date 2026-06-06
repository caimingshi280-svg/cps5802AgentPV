"""Unit tests for :class:`evaluation.pytorch_runner.PyTorchClassifier`.

Builds a tiny CNN1D on the fly, saves a checkpoint in the format
:func:`training.train` would produce, then verifies the wrapper:
- enforces input shape & feature count,
- reports correct Predictor-protocol attributes,
- produces deterministic logits,
- measure_pytorch_state_dict_bytes returns a positive integer.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from api.schemas import SystemType
from evaluation.predictor import Predictor
from evaluation.pytorch_runner import (
    PyTorchClassifier,
    measure_pytorch_state_dict_bytes,
)
from models.cnn1d import CNN1D


def _save_synthetic_checkpoint(
    tmp_path: Path, *, in_channels: int = 4, n_classes: int = 3
) -> Path:
    """Mirror the layout produced by training/trainer.py."""

    model = CNN1D(in_channels=in_channels, n_classes=n_classes, dropout=0.30).eval()
    feature_stats = {
        "mean": [0.0] * in_channels,
        "std": [1.0] * in_channels,
        "feature_names": [f"f{i}" for i in range(in_channels)],
    }
    payload = {
        "model_state_dict": model.state_dict(),
        "label_classes": tuple(f"class_{i}" for i in range(n_classes)),
        "feature_stats": feature_stats,
        "system_type": SystemType.PV.value,
        "in_channels": in_channels,
        "n_classes": n_classes,
        "dropout": 0.30,
        "model_arch": "CNN1D",
        "val_macro_f1": 0.91,
    }
    ckpt = tmp_path / "tiny.pt"
    torch.save(payload, str(ckpt))
    return ckpt


def test_pytorch_classifier_satisfies_predictor_protocol(tmp_path: Path) -> None:
    ckpt = _save_synthetic_checkpoint(tmp_path)
    clf = PyTorchClassifier(ckpt)
    assert isinstance(clf, Predictor)
    assert clf.system_type is SystemType.PV
    assert clf.in_channels == 4
    assert clf.label_classes == ("class_0", "class_1", "class_2")


def test_pytorch_classifier_run_logits_shape(tmp_path: Path) -> None:
    ckpt = _save_synthetic_checkpoint(tmp_path)
    clf = PyTorchClassifier(ckpt)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 12, 4)).astype(np.float32)
    logits = clf.run_logits(x)
    assert logits.shape == (5, 3)
    assert logits.dtype == np.float32


def test_pytorch_classifier_is_deterministic(tmp_path: Path) -> None:
    ckpt = _save_synthetic_checkpoint(tmp_path)
    clf = PyTorchClassifier(ckpt)
    x = np.full((2, 8, 4), 0.5, dtype=np.float32)
    a = clf.run_logits(x)
    b = clf.run_logits(x)
    np.testing.assert_array_equal(a, b)


def test_pytorch_classifier_rejects_wrong_input_shape(tmp_path: Path) -> None:
    ckpt = _save_synthetic_checkpoint(tmp_path)
    clf = PyTorchClassifier(ckpt)
    with pytest.raises(ValueError, match=r"\(B, T, F\)"):
        clf.run_logits(np.zeros((10, 4), dtype=np.float32))


def test_pytorch_classifier_rejects_wrong_channel_count(tmp_path: Path) -> None:
    ckpt = _save_synthetic_checkpoint(tmp_path, in_channels=4)
    clf = PyTorchClassifier(ckpt)
    with pytest.raises(ValueError, match="feature channels"):
        clf.run_logits(np.zeros((1, 8, 7), dtype=np.float32))


def test_pytorch_classifier_rejects_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PyTorchClassifier(tmp_path / "missing.pt")


def test_pytorch_classifier_rejects_unknown_arch(tmp_path: Path) -> None:
    ckpt = _save_synthetic_checkpoint(tmp_path)
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    payload["model_arch"] = "MysteryModel"
    torch.save(payload, str(ckpt))
    with pytest.raises(ValueError, match="unsupported model_arch"):
        PyTorchClassifier(ckpt)


def test_pytorch_classifier_requires_required_fields(tmp_path: Path) -> None:
    ckpt = _save_synthetic_checkpoint(tmp_path)
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    del payload["feature_stats"]
    torch.save(payload, str(ckpt))
    with pytest.raises(KeyError, match="feature_stats"):
        PyTorchClassifier(ckpt)


def test_measure_state_dict_bytes_returns_positive_and_close_to_full(
    tmp_path: Path,
) -> None:
    """The state-dict-only blob should be roughly the same size as the
    full training checkpoint — they both carry the same tensor data
    and torch adds a constant per-save header. We assert it's positive
    and within ±5 KiB of the full checkpoint as a sanity guard.
    """

    ckpt = _save_synthetic_checkpoint(tmp_path)
    full_bytes = ckpt.stat().st_size
    state_bytes = measure_pytorch_state_dict_bytes(ckpt)
    assert state_bytes > 0
    assert abs(state_bytes - full_bytes) < 5 * 1024


def test_measure_state_dict_bytes_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        measure_pytorch_state_dict_bytes(tmp_path / "missing.pt")
