"""Dataset generator CLI for AgentPV (Component 1).

Examples
--------
Full assignment-scale dataset (~51,000 samples):

    python -m simulation.generate_dataset --seed 42

Small smoke run for tests:

    python -m simulation.generate_dataset --seed 0 --n-pv 80 --n-bess 40 \
        --out-dir data/_smoke

中文说明
--------
根据 PV/BESS 仿真与故障注入生成带标签时间序列，写出 parquet 与划分索引；
``--version-path`` 可在测试中指向临时文件，避免覆盖仓库 canonical 的
``data/version.txt``。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from api.schemas import (
    BESS_FAULT_CLASSES,
    PV_FAULT_CLASSES,
    DatasetMetadata,
    OperatingCondition,
    RawSample,
    SensorWindow,
    SplitName,
    SystemType,
)
from simulation.battery_simulator import BESS_FEATURE_NAMES, BatterySimulator
from simulation.fault_injector import inject_fault
from simulation.pv_simulator import PV_FEATURE_NAMES, PVSimulator
from utils.logging_config import get_logger
from utils.paths import (
    DATA_DIR,
    PROCESSED_DIR,
    SPLITS_DIR,
    ensure_dir,
)
from utils.seeds import set_global_seed

log = get_logger(__name__)

# 三种工况按 5:3:2 采样（高辐照最常见）
_CONDITION_WEIGHTS: dict[OperatingCondition, float] = {
    OperatingCondition.HIGH_IRRADIANCE: 0.5,
    OperatingCondition.LOW_IRRADIANCE: 0.3,
    OperatingCondition.HIGH_TEMPERATURE: 0.2,
}

DEFAULT_WINDOW_SIZE: int = 60
DEFAULT_SAMPLE_RATE_HZ: float = 1.0
DEFAULT_SCHEMA_VERSION: str = "0.1.0"


# ---------------------------------------------------------------------------
# Task list construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Task:
    """One generation task: which system_type, label, condition, and id."""

    sample_idx: int
    system_type: SystemType
    label: str
    condition: OperatingCondition
    system_id: str


def _build_pv_distribution(n_pv_normal: int, n_pv_total: int) -> dict[str, int]:
    """Allocate PV samples per fault class with Normal getting the surplus."""

    n_fault_classes = len(PV_FAULT_CLASSES) - 1
    per_fault = (n_pv_total - n_pv_normal) // n_fault_classes
    distribution: dict[str, int] = {"PV_Normal": n_pv_normal}
    for cls in PV_FAULT_CLASSES:
        if cls == "PV_Normal":
            continue
        distribution[cls] = per_fault
    leftover = n_pv_total - sum(distribution.values())
    distribution["PV_Normal"] += leftover  # 把整除余数挂到 Normal 上保证总数精确
    return distribution


def _build_bess_distribution(n_bess_normal: int, n_bess_total: int) -> dict[str, int]:
    """Allocate BESS samples per fault class."""

    n_fault_classes = len(BESS_FAULT_CLASSES) - 1
    per_fault = (n_bess_total - n_bess_normal) // n_fault_classes
    distribution: dict[str, int] = {"BESS_Normal": n_bess_normal}
    for cls in BESS_FAULT_CLASSES:
        if cls == "BESS_Normal":
            continue
        distribution[cls] = per_fault
    leftover = n_bess_total - sum(distribution.values())
    distribution["BESS_Normal"] += leftover
    return distribution


def _build_task_list(
    pv_dist: dict[str, int],
    bess_dist: dict[str, int],
    rng: np.random.Generator,
) -> list[_Task]:
    """Construct the shuffled list of generation tasks."""

    conditions = list(_CONDITION_WEIGHTS.keys())
    cond_p = np.array(list(_CONDITION_WEIGHTS.values()), dtype=np.float64)
    cond_p = cond_p / cond_p.sum()

    tasks: list[_Task] = []
    sample_idx = 0
    # PV
    for label, n in pv_dist.items():
        for _ in range(n):
            cond = conditions[rng.choice(len(conditions), p=cond_p)]
            sys_id = f"PV_{rng.integers(0, 200):04d}"
            tasks.append(
                _Task(sample_idx, SystemType.PV, label, cond, sys_id)
            )
            sample_idx += 1
    # BESS
    for label, n in bess_dist.items():
        for _ in range(n):
            cond = conditions[rng.choice(len(conditions), p=cond_p)]
            sys_id = f"BESS_{rng.integers(0, 200):04d}"
            tasks.append(
                _Task(sample_idx, SystemType.BESS, label, cond, sys_id)
            )
            sample_idx += 1

    rng.shuffle(tasks)
    return tasks


# ---------------------------------------------------------------------------
# Stratified split by system_id
# ---------------------------------------------------------------------------


def _stratified_split_by_system_id(
    meta_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    rng: np.random.Generator,
) -> dict[SplitName, list[int]]:
    """Split sample indices into train/val/test, keeping a system_id only in
    one split — prevents leakage (rule §6 prevent train/test leakage)."""

    sample_indices: dict[SplitName, list[int]] = {
        SplitName.TRAIN: [],
        SplitName.VAL: [],
        SplitName.TEST: [],
    }

    for _label, group in meta_df.groupby("label"):
        unique_ids = group["system_id"].unique().tolist()
        rng.shuffle(unique_ids)
        n_total = len(unique_ids)
        n_train = int(round(n_total * train_ratio))
        n_val = int(round(n_total * val_ratio))
        train_ids = set(unique_ids[:n_train])
        val_ids = set(unique_ids[n_train : n_train + n_val])

        for _, row in group.iterrows():
            sid = row["system_id"]
            if sid in train_ids:
                sample_indices[SplitName.TRAIN].append(int(row["sample_idx"]))
            elif sid in val_ids:
                sample_indices[SplitName.VAL].append(int(row["sample_idx"]))
            else:
                sample_indices[SplitName.TEST].append(int(row["sample_idx"]))

    for indices in sample_indices.values():
        indices.sort()
    return sample_indices


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def generate(
    out_dir: Path = PROCESSED_DIR,
    splits_dir: Path = SPLITS_DIR,
    n_pv_total: int = 21000,
    n_bess_total: int = 17000,
    n_pv_normal: int = 8000,
    n_bess_normal: int = 5000,
    window_size: int = DEFAULT_WINDOW_SIZE,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    validate_every: int = 500,
    version_path: Path | None = None,
) -> DatasetMetadata:
    """Generate the AgentPV dataset and write it to ``out_dir``.

    Parameters
    ----------
    version_path
        Destination for ``version.txt``. Defaults to ``DATA_DIR/version.txt``
        **only when** ``out_dir`` is the project's canonical
        ``data/processed`` directory; tests passing a ``tmp_path`` will get
        ``out_dir.parent / "version.txt"`` automatically so the project's
        committed metadata is never overwritten by a test run.

    Returns
    -------
    DatasetMetadata
        The metadata that has also been persisted to ``version_path``.
    """

    set_global_seed(seed)
    ensure_dir(out_dir)
    ensure_dir(splits_dir)
    rng = np.random.default_rng(seed)

    pv_sim = PVSimulator(seed=seed)
    bess_sim = BatterySimulator(seed=seed + 1)

    pv_dist = _build_pv_distribution(n_pv_normal, n_pv_total)
    bess_dist = _build_bess_distribution(n_bess_normal, n_bess_total)
    n_total = n_pv_total + n_bess_total

    log.info(
        "dataset_plan",
        extra={
            "n_total": n_total,
            "pv_distribution": pv_dist,
            "bess_distribution": bess_dist,
            "seed": seed,
        },
    )

    tasks = _build_task_list(pv_dist, bess_dist, rng)

    # 预分配 numpy 缓冲区——比逐样本拼接快几十倍。
    n_pv = sum(pv_dist.values())
    n_bess = sum(bess_dist.values())
    F_PV = len(PV_FEATURE_NAMES)
    F_BESS = len(BESS_FEATURE_NAMES)
    X_pv = np.zeros((n_pv, window_size, F_PV), dtype=np.float32)
    y_pv = np.empty(n_pv, dtype=np.dtype("U32"))
    X_bess = np.zeros((n_bess, window_size, F_BESS), dtype=np.float32)
    y_bess = np.empty(n_bess, dtype=np.dtype("U32"))

    pv_meta_rows: list[dict] = []
    bess_meta_rows: list[dict] = []

    pv_writer_idx = 0
    bess_writer_idx = 0
    cond_counter: Counter[OperatingCondition] = Counter()
    label_counter: Counter[str] = Counter()

    base_ts = datetime.now(UTC)

    for i, task in enumerate(tasks):
        if task.system_type is SystemType.PV:
            clean = pv_sim.simulate(task.condition, window_size=window_size)
        else:
            clean = bess_sim.simulate(task.condition, window_size=window_size)
        faulty = inject_fault(clean, task.label, rng)

        # 周期性强 schema 校验——抽样 1/validate_every 经过 RawSample 才保证
        # 我们没有偷偷输出无效数据；不每条都校验是为了速度。
        if i % validate_every == 0:
            feature_names = (
                list(PV_FEATURE_NAMES)
                if task.system_type is SystemType.PV
                else list(BESS_FEATURE_NAMES)
            )
            window = SensorWindow(
                timestamp_start=base_ts,
                system_id=task.system_id,
                system_type=task.system_type,
                sample_rate_hz=sample_rate_hz,
                window_size=window_size,
                feature_names=feature_names,
                values=faulty.tolist(),
                operating_condition=task.condition,
            )
            RawSample(window=window, label=task.label)

        # 存到对应缓冲区
        meta_row = {
            "sample_idx": task.sample_idx,
            "system_id": task.system_id,
            "system_type": task.system_type.value,
            "label": task.label,
            "operating_condition": task.condition.value,
        }
        if task.system_type is SystemType.PV:
            X_pv[pv_writer_idx] = faulty
            y_pv[pv_writer_idx] = task.label
            pv_meta_rows.append({"local_idx": pv_writer_idx, **meta_row})
            pv_writer_idx += 1
        else:
            X_bess[bess_writer_idx] = faulty
            y_bess[bess_writer_idx] = task.label
            bess_meta_rows.append({"local_idx": bess_writer_idx, **meta_row})
            bess_writer_idx += 1

        cond_counter[task.condition] += 1
        label_counter[task.label] += 1

    # 落盘
    np.savez_compressed(out_dir / "X_pv.npz", X=X_pv)
    np.savez_compressed(out_dir / "X_bess.npz", X=X_bess)
    np.savez_compressed(out_dir / "y_pv.npz", y=y_pv)
    np.savez_compressed(out_dir / "y_bess.npz", y=y_bess)

    pv_meta_df = pd.DataFrame(pv_meta_rows)
    bess_meta_df = pd.DataFrame(bess_meta_rows)
    pv_meta_df.to_csv(out_dir / "meta_pv.csv", index=False)
    bess_meta_df.to_csv(out_dir / "meta_bess.csv", index=False)

    # train/val/test 切分（按 system_id 分层）
    full_meta = pd.concat([pv_meta_df, bess_meta_df], ignore_index=True)
    split_indices = _stratified_split_by_system_id(
        full_meta, train_ratio=train_ratio, val_ratio=val_ratio, rng=rng
    )
    for split_name, indices in split_indices.items():
        split_df = full_meta[full_meta["sample_idx"].isin(indices)].copy()
        split_df.to_csv(splits_dir / f"{split_name.value}.csv", index=False)

    metadata = DatasetMetadata(
        schema_version=DEFAULT_SCHEMA_VERSION,
        generated_at=datetime.now(UTC),
        seed=seed,
        sample_rate_hz=sample_rate_hz,
        window_size=window_size,
        pv_feature_names=list(PV_FEATURE_NAMES),
        bess_feature_names=list(BESS_FEATURE_NAMES),
        n_samples=n_total,
        splits={SplitName(k.value): len(v) for k, v in split_indices.items()},
        class_distribution={k: int(v) for k, v in label_counter.items()},
        operating_condition_distribution={
            cond: int(cond_counter[cond]) for cond in OperatingCondition
        },
        notes=(
            f"AgentPV C1 generated dataset; "
            f"pv_total={n_pv_total}, bess_total={n_bess_total}, seed={seed}"
        ),
    )
    # When the caller passes a non-canonical ``out_dir`` (e.g. pytest's
    # ``tmp_path``) we must NOT overwrite the project's committed
    # ``data/version.txt`` — the metadata for the production dataset is a
    # report artifact. Falls back to the canonical project path only when
    # ``out_dir`` matches the project layout exactly.
    if version_path is None:
        if out_dir.resolve() == PROCESSED_DIR.resolve():
            target = DATA_DIR / "version.txt"
        else:
            target = out_dir.parent / "version.txt"
    else:
        target = version_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")

    log.info(
        "dataset_done",
        extra={
            "n_total": n_total,
            "splits": {k.value: len(v) for k, v in split_indices.items()},
            "out_dir": str(out_dir),
        },
    )
    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="agentpv-generate-dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-pv", type=int, default=21000, dest="n_pv_total")
    parser.add_argument("--n-bess", type=int, default=17000, dest="n_bess_total")
    parser.add_argument("--n-pv-normal", type=int, default=8000)
    parser.add_argument("--n-bess-normal", type=int, default=5000)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Output directory for npz/csv (default: data/processed)",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=SPLITS_DIR,
        help="Output directory for split csvs (default: data/splits)",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""

    args = _parse_args(argv)
    metadata = generate(
        out_dir=args.out_dir,
        splits_dir=args.splits_dir,
        n_pv_total=args.n_pv_total,
        n_bess_total=args.n_bess_total,
        n_pv_normal=args.n_pv_normal,
        n_bess_normal=args.n_bess_normal,
        window_size=args.window_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(json.dumps(metadata.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
