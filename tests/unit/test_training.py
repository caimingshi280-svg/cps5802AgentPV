"""Unit tests for :mod:`training.data` / :mod:`training.losses` / :mod:`training.trainer`.

These tests construct tiny synthetic datasets — they must NOT depend on
the real ``data/processed`` artifacts being present.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402
from torch import optim  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from api.schemas import PV_FAULT_CLASSES, SystemType  # noqa: E402
from models.cnn1d import CNN1D  # noqa: E402
from training.data import (  # noqa: E402
    BESS_LABEL_MAP,
    PV_LABEL_MAP,
    FeatureStats,
    LabelMap,
    TimeSeriesNpzDataset,
    class_weights,
    label_map_for,
)
from training.losses import FocalLoss, WeightedCrossEntropyLoss  # noqa: E402
from training.trainer import Trainer  # noqa: E402

# ---------------------------------------------------------------------------
# data.py
# ---------------------------------------------------------------------------


def test_label_map_for_pv() -> None:
    assert label_map_for(SystemType.PV) is PV_LABEL_MAP


def test_label_map_for_bess() -> None:
    assert label_map_for(SystemType.BESS) is BESS_LABEL_MAP


def test_label_map_round_trip() -> None:
    lm = LabelMap(classes=PV_FAULT_CLASSES)
    for idx, cls in enumerate(PV_FAULT_CLASSES):
        assert lm.to_id(cls) == idx
        assert lm.to_label(idx) == cls


def _make_dummy_arrays(n_per_class: int = 4, classes: tuple[str, ...] = PV_FAULT_CLASSES) -> tuple[np.ndarray, np.ndarray]:
    n = n_per_class * len(classes)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, 60, 8)).astype(np.float32)
    y = np.array([cls for cls in classes for _ in range(n_per_class)])
    return x, y


def test_timeseries_dataset_shapes() -> None:
    x, y = _make_dummy_arrays()
    ds = TimeSeriesNpzDataset(x, y, label_map=PV_LABEL_MAP)
    assert len(ds) == x.shape[0]
    sample, label = ds[0]
    assert sample.shape == (60, 8)
    assert sample.dtype == torch.float32
    assert label.dtype == torch.long


def test_timeseries_dataset_rejects_mismatched_lengths() -> None:
    x, _ = _make_dummy_arrays()
    with pytest.raises(ValueError, match="length mismatch"):
        TimeSeriesNpzDataset(x, np.array(["PV_Normal"]), label_map=PV_LABEL_MAP)


def test_timeseries_dataset_rejects_2d_input() -> None:
    with pytest.raises(ValueError, match="3D"):
        TimeSeriesNpzDataset(
            np.zeros((4, 60), dtype=np.float32),
            np.array(["PV_Normal"] * 4),
            label_map=PV_LABEL_MAP,
        )


def test_feature_stats_fit_centers_and_scales() -> None:
    rng = np.random.default_rng(0)
    # 3 channels with very different scales — the BESS-style failure mode.
    x = rng.standard_normal((100, 60, 3)).astype(np.float32) * np.array([1.0, 100.0, 0.001]) + np.array([5.0, 50.0, -0.01])
    stats = FeatureStats.fit(x)
    standardized = stats.apply(x)
    flat = standardized.reshape(-1, 3)
    np.testing.assert_allclose(flat.mean(axis=0), np.zeros(3), atol=1e-3)
    np.testing.assert_allclose(flat.std(axis=0), np.ones(3), atol=1e-3)


def test_feature_stats_round_trip_dict() -> None:
    stats = FeatureStats(
        mean=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        std=np.array([0.5, 1.0, 2.0], dtype=np.float32),
    )
    restored = FeatureStats.from_dict(stats.to_dict())
    np.testing.assert_array_equal(restored.mean, stats.mean)
    np.testing.assert_array_equal(restored.std, stats.std)


def test_feature_stats_handles_zero_variance_channel() -> None:
    """A constant feature must not divide by zero — eps must clamp std."""
    x = np.zeros((10, 60, 3), dtype=np.float32)
    x[..., 0] = 7.0  # constant channel
    stats = FeatureStats.fit(x, eps=1e-6)
    assert stats.std[0] >= 1e-6
    out = stats.apply(x)
    assert np.isfinite(out).all()


def test_dataset_applies_feature_stats() -> None:
    x = np.full((4, 60, 8), fill_value=10.0, dtype=np.float32)
    y = np.array(["PV_Normal"] * 4)
    stats = FeatureStats(
        mean=np.full(8, 10.0, dtype=np.float32),
        std=np.ones(8, dtype=np.float32),
    )
    ds = TimeSeriesNpzDataset(x, y, label_map=PV_LABEL_MAP, feature_stats=stats)
    sample, _ = ds[0]
    assert torch.allclose(sample, torch.zeros(60, 8))


def test_class_weights_inverse_frequency() -> None:
    # 4 classes with imbalance: [10, 5, 1, 4]
    labels = np.concatenate(
        [np.zeros(10), np.ones(5), np.full(1, 2), np.full(4, 3)]
    ).astype(int)
    w = class_weights(labels, n_classes=4)
    assert w.shape == (4,)
    # The rarer class should have a strictly higher weight.
    assert w[2] > w[0]
    assert w[1] > w[0]


# ---------------------------------------------------------------------------
# losses.py
# ---------------------------------------------------------------------------


def test_weighted_ce_runs() -> None:
    loss = WeightedCrossEntropyLoss(class_weights=torch.tensor([1.0, 2.0, 1.5]))
    logits = torch.randn(4, 3, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 1])
    out = loss(logits, targets)
    out.backward()
    assert torch.isfinite(out)
    assert out.item() > 0


def test_focal_loss_zero_when_perfect() -> None:
    """When p_t -> 1, focal loss -> 0."""
    targets = torch.tensor([0, 1])
    logits = torch.tensor([[20.0, -20.0], [-20.0, 20.0]])
    loss = FocalLoss(gamma=2.0)(logits, targets)
    assert loss.item() < 1e-3


def test_focal_loss_high_when_wrong() -> None:
    targets = torch.tensor([0, 1])
    logits = torch.tensor([[-20.0, 20.0], [20.0, -20.0]])  # wrong on both
    loss = FocalLoss(gamma=2.0)(logits, targets)
    assert loss.item() > 5.0


# ---------------------------------------------------------------------------
# trainer.py — integration smoke test
# ---------------------------------------------------------------------------


def test_trainer_overfits_tiny_separable_problem(tmp_path) -> None:
    """The trainer should drive macro F1 -> 1.0 on a tiny separable problem."""

    rng = np.random.default_rng(0)
    n_classes = 3
    samples_per_class = 32

    x_list = []
    y_list = []
    for cls in range(n_classes):
        # Each class is a constant offset across all features; trivially separable.
        offset = (cls - 1) * 5.0
        x_cls = rng.standard_normal((samples_per_class, 60, 8)).astype(np.float32) * 0.1 + offset
        y_cls = np.full(samples_per_class, cls, dtype=np.int64)
        x_list.append(x_cls)
        y_list.append(y_cls)
    x = np.concatenate(x_list, axis=0)
    y_int = np.concatenate(y_list, axis=0)
    # Convert to string labels using the first 3 PV classes.
    y_str = np.array([PV_FAULT_CLASSES[i] for i in y_int])

    full_ds = TimeSeriesNpzDataset(
        x, y_str, label_map=LabelMap(classes=PV_FAULT_CLASSES[:n_classes])
    )

    train_loader = DataLoader(full_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(full_ds, batch_size=32, shuffle=False)

    torch.manual_seed(0)
    model = CNN1D(in_channels=8, n_classes=n_classes, dropout=0.0)
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    trainer = Trainer(
        model=model,
        loss_fn=WeightedCrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=None,
        device="cpu",
        early_stop_patience=5,
    )

    result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=12,
        checkpoint_dir=tmp_path,
        run_name="overfit_smoke",
    )
    assert result.best_macro_f1 >= 0.95, f"trainer failed to overfit: {result}"
    assert result.best_checkpoint_path.exists()
    assert len(result.history) >= 1
