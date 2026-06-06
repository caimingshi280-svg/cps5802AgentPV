"""CLI entry point for training a single AgentPV classifier.

Examples
--------
Train PV model with defaults (CPU, 30 epochs):

    python -m training.train --system pv

Train BESS model with smaller batch and 10 epochs:

    python -m training.train --system bess --epochs 10 --batch-size 64
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch import optim
from torch.utils.data import DataLoader

from api.schemas import SplitName, SystemType
from models.cnn1d import CNN1D
from training.data import (
    BESS_LABEL_MAP,
    PV_LABEL_MAP,
    FeatureStats,
    TimeSeriesNpzDataset,
    _load_split_arrays,
    class_weights,
    load_split,
)
from training.losses import WeightedCrossEntropyLoss
from training.trainer import Trainer
from utils.logging_config import get_logger
from utils.paths import ARTIFACTS_DIR, PROCESSED_DIR, SPLITS_DIR
from utils.seeds import set_global_seed

log = get_logger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="agentpv-train")
    parser.add_argument(
        "--system",
        choices=["pv", "bess"],
        required=True,
        help="Which system's classifier to train.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--early-stop-patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory containing X_*.npz / y_*.npz / meta_*.csv",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=SPLITS_DIR,
        help="Directory containing train.csv / val.csv / test.csv",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ARTIFACTS_DIR,
        help="Where to save the best .pt checkpoint",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Inference target is CPU (rule §17), so default to cpu here too.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    set_global_seed(args.seed)

    system_type = SystemType.PV if args.system == "pv" else SystemType.BESS
    label_map = PV_LABEL_MAP if system_type is SystemType.PV else BESS_LABEL_MAP

    log.info(
        "train_start",
        extra={
            "system": args.system,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": args.device,
        },
    )

    # Fit per-channel mean/std on TRAIN ONLY — required because BESS/PV
    # features span multiple physical units (V, A, °C, Wh, ...). Using
    # train-only statistics avoids any val/test leakage (rule §6).
    x_train_raw, y_train_raw = _load_split_arrays(
        args.processed_dir, args.splits_dir, system_type, SplitName.TRAIN
    )
    feature_stats = FeatureStats.fit(x_train_raw)
    log.info(
        "feature_stats_fit",
        extra={
            "mean": [round(float(v), 4) for v in feature_stats.mean],
            "std": [round(float(v), 4) for v in feature_stats.std],
        },
    )

    train_ds = TimeSeriesNpzDataset(
        x_train_raw,
        y_train_raw,
        label_map=label_map,
        feature_stats=feature_stats,
    )
    val_ds = load_split(
        args.processed_dir,
        args.splits_dir,
        system_type,
        SplitName.VAL,
        feature_stats=feature_stats,
    )

    log.info(
        "datasets_loaded",
        extra={
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_classes": len(label_map.classes),
        },
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    n_classes = len(label_map.classes)
    weights = class_weights(train_ds.labels, n_classes=n_classes)
    loss_fn = WeightedCrossEntropyLoss(class_weights=weights)

    model = CNN1D(in_channels=8, n_classes=n_classes, dropout=args.dropout)
    log.info("model_built", extra={"n_params": model.num_parameters()})

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        early_stop_patience=args.early_stop_patience,
        extra_checkpoint_payload={
            "system_type": system_type.value,
            "n_classes": n_classes,
            "label_classes": list(label_map.classes),
            "feature_stats": feature_stats.to_dict(),
            "model_arch": "CNN1D",
            "in_channels": 8,
            "dropout": args.dropout,
        },
    )

    run_name = f"cnn1d_{args.system}"
    result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        checkpoint_dir=args.artifacts_dir,
        run_name=run_name,
    )

    summary = {
        "system": args.system,
        "n_classes": n_classes,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "best_epoch": result.best_epoch,
        "best_val_macro_f1": result.best_macro_f1,
        "checkpoint_path": str(result.best_checkpoint_path),
        "epochs_run": len(result.history),
        "history": [
            {
                "epoch": m.epoch,
                "train_loss": round(m.train_loss, 4),
                "val_loss": round(m.val_loss, 4),
                "val_macro_f1": round(m.val_macro_f1, 4),
                "val_accuracy": round(m.val_accuracy, 4),
                "elapsed_s": round(m.elapsed_s, 2),
            }
            for m in result.history
        ],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
