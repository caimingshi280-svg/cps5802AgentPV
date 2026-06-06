"""Dataset adapters for the npz/csv layout produced by Component 1.

Layout assumption (see ``simulation/README.md``):

::

    data/processed/X_pv.npz    arr "X" of shape (N, T, F)
    data/processed/y_pv.npz    arr "y" of shape (N,) dtype <U32 (label strings)
    data/processed/meta_pv.csv columns: local_idx, sample_idx, system_id,
                                        system_type, label, operating_condition
    data/splits/train.csv      columns: sample_idx, system_id, system_type,
                                        label, operating_condition
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from api.schemas import (
    BESS_FAULT_CLASSES,
    PV_FAULT_CLASSES,
    SplitName,
    SystemType,
)


@dataclass(frozen=True)
class LabelMap:
    """Bidirectional mapping between string labels and integer ids."""

    classes: tuple[str, ...]

    def to_id(self, label: str) -> int:
        return self.classes.index(label)

    def to_label(self, idx: int) -> str:
        return self.classes[idx]


PV_LABEL_MAP = LabelMap(classes=PV_FAULT_CLASSES)
BESS_LABEL_MAP = LabelMap(classes=BESS_FAULT_CLASSES)


def label_map_for(system_type: SystemType) -> LabelMap:
    """Pick the label map that matches the requested system type."""

    return PV_LABEL_MAP if system_type is SystemType.PV else BESS_LABEL_MAP


@dataclass(frozen=True)
class FeatureStats:
    """Per-channel mean/std for input standardization.

    Statistics are fit on the **training** split only and applied to all
    splits to avoid information leakage. The arrays are float32 with shape
    ``(F,)`` matching ``SensorWindow.feature_names`` order.
    """

    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        if self.mean.shape != self.std.shape or self.mean.ndim != 1:
            raise ValueError(
                f"FeatureStats expects matching 1-D mean/std, "
                f"got mean.shape={self.mean.shape}, std.shape={self.std.shape}"
            )

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Apply (x - mean) / std with broadcasting over (..., T, F)."""
        return ((x - self.mean) / self.std).astype(np.float32, copy=False)

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, list[float]]) -> FeatureStats:
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
        )

    @classmethod
    def fit(cls, x: np.ndarray, eps: float = 1e-6) -> FeatureStats:
        """Compute per-feature mean/std across all samples and time steps."""
        if x.ndim != 3:
            raise ValueError(f"expected 3-D X to fit stats, got {x.shape}")
        flat = x.reshape(-1, x.shape[-1])
        mean = flat.mean(axis=0).astype(np.float32)
        std = flat.std(axis=0).astype(np.float32)
        # 防 zero-variance 通道导致除零。
        std = np.maximum(std, eps).astype(np.float32)
        return cls(mean=mean, std=std)


class TimeSeriesNpzDataset(Dataset):
    """In-memory dataset backed by the npz files written by Component 1.

    Memory footprint for the assignment-scale dataset (51k × 60 × 8 × 4 B)
    is ~100 MB, which is acceptable for CPU training. If a larger dataset
    is later produced, swap in ``np.load(..., mmap_mode='r')``.

    Optional ``feature_stats`` standardize inputs per channel — required for
    physically heterogeneous features (e.g. BESS energy in Wh ≫ voltage in V).
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        label_map: LabelMap,
        feature_stats: FeatureStats | None = None,
    ) -> None:
        if x.ndim != 3:
            raise ValueError(f"X must be 3D (N, T, F), got {x.shape}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y length mismatch: {x.shape[0]} vs {y.shape[0]}"
            )
        x_arr = x.astype(np.float32, copy=False)
        if feature_stats is not None:
            x_arr = feature_stats.apply(x_arr)
        # 提前转 int 标签——避免训练循环里反复字符串查找。
        self._x = x_arr
        self._y = np.array([label_map.to_id(str(label)) for label in y], dtype=np.int64)
        self.label_map = label_map
        self.feature_stats = feature_stats

    def __len__(self) -> int:
        return self._x.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_t = torch.from_numpy(self._x[idx])  # shape (T, F)
        y_t = torch.tensor(self._y[idx], dtype=torch.long)
        return x_t, y_t

    @property
    def labels(self) -> np.ndarray:
        """Return the integer label vector — used for class weighting."""
        return self._y


def _filter_indices_by_split(
    meta_df: pd.DataFrame,
    split_df: pd.DataFrame,
) -> np.ndarray:
    """Return a boolean mask over ``meta_df`` selecting samples in ``split_df``."""

    split_ids = set(split_df["sample_idx"].astype(int))
    return meta_df["sample_idx"].astype(int).isin(split_ids).to_numpy()


def _load_split_arrays(
    processed_dir: Path,
    splits_dir: Path,
    system_type: SystemType,
    split: SplitName,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw ``(X, y)`` arrays for one split — no standardization."""

    suffix = "pv" if system_type is SystemType.PV else "bess"
    x_path = processed_dir / f"X_{suffix}.npz"
    y_path = processed_dir / f"y_{suffix}.npz"
    meta_path = processed_dir / f"meta_{suffix}.csv"
    split_path = splits_dir / f"{split.value}.csv"

    for path in (x_path, y_path, meta_path, split_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset artifact: {path}")

    x_full = np.load(x_path)["X"]
    y_full = np.load(y_path)["y"]
    meta_df = pd.read_csv(meta_path)
    split_df = pd.read_csv(split_path)

    # split_df 包含两种 system_type 的样本，需按 system_type 过滤。
    split_df = split_df[split_df["system_type"] == system_type.value].copy()
    mask = _filter_indices_by_split(meta_df, split_df)

    return x_full[mask], y_full[mask]


def load_split(
    processed_dir: Path,
    splits_dir: Path,
    system_type: SystemType,
    split: SplitName,
    feature_stats: FeatureStats | None = None,
) -> TimeSeriesNpzDataset:
    """Load one split (train/val/test) for one system type, optionally standardized."""

    x_split, y_split = _load_split_arrays(processed_dir, splits_dir, system_type, split)
    return TimeSeriesNpzDataset(
        x_split,
        y_split,
        label_map=label_map_for(system_type),
        feature_stats=feature_stats,
    )


def class_weights(labels: Sequence[int], n_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for weighted cross-entropy.

    Returns a 1-D tensor of length ``n_classes`` suitable for passing to
    :class:`torch.nn.CrossEntropyLoss(weight=...)`.
    """

    counts = np.bincount(np.asarray(labels), minlength=n_classes).astype(np.float64)
    counts = np.clip(counts, 1.0, None)  # 0 计数时给个 1 避免除零
    inv = counts.sum() / (n_classes * counts)
    return torch.from_numpy(inv.astype(np.float32))
