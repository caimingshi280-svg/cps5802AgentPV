"""Unit tests for :mod:`models.cnn1d`."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from models.cnn1d import CNN1D  # noqa: E402


def test_cnn1d_forward_shape_pv() -> None:
    model = CNN1D(in_channels=8, n_classes=7).eval()
    x = torch.randn(4, 60, 8)
    logits = model(x)
    assert logits.shape == (4, 7)


def test_cnn1d_forward_shape_bess() -> None:
    model = CNN1D(in_channels=8, n_classes=5).eval()
    x = torch.randn(2, 60, 8)
    logits = model(x)
    assert logits.shape == (2, 5)


def test_cnn1d_logits_are_finite() -> None:
    model = CNN1D(in_channels=8, n_classes=7).eval()
    x = torch.randn(8, 60, 8)
    logits = model(x)
    assert torch.isfinite(logits).all()


def test_cnn1d_rejects_wrong_dimensionality() -> None:
    model = CNN1D(in_channels=8, n_classes=7)
    with pytest.raises(ValueError, match="3D input"):
        model(torch.randn(60, 8))  # missing batch axis


def test_cnn1d_rejects_wrong_feature_count() -> None:
    model = CNN1D(in_channels=8, n_classes=7)
    with pytest.raises(ValueError, match="feature channels"):
        model(torch.randn(2, 60, 4))


def test_cnn1d_param_count_is_compact() -> None:
    """Compact parameter count is required for INT8 quantization later."""
    model = CNN1D(in_channels=8, n_classes=12)
    n = model.num_parameters()
    # Expect ~50k params; a sane upper bound keeps the architecture honest.
    assert 10_000 < n < 200_000, f"unexpected parameter count: {n}"


def test_cnn1d_supports_backprop() -> None:
    """Forward + backward path must produce non-None gradients."""
    model = CNN1D(in_channels=8, n_classes=7).train()
    x = torch.randn(4, 60, 8, requires_grad=False)
    y = torch.tensor([0, 1, 2, 3])
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    grads_present = [p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters() if p.requires_grad]
    assert all(grads_present), "every trainable parameter must receive a gradient"
