"""Generic Trainer for time-series classifiers (Component 2 MVP).

Single responsibility: orchestrate the train / validate loop. No CLI, no
data loading — that is delegated to :mod:`training.train` and
:mod:`training.data` respectively (rule §2 modular design).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.utils.data import DataLoader

from utils.logging_config import get_logger
from utils.paths import ensure_dir

log = get_logger(__name__)


@dataclass
class EpochMetrics:
    """Per-epoch metrics — emitted to logs and (later) TensorBoard."""

    epoch: int
    train_loss: float
    val_loss: float
    val_macro_f1: float
    val_accuracy: float
    elapsed_s: float


@dataclass
class TrainResult:
    """Aggregate result returned by :meth:`Trainer.fit`."""

    best_macro_f1: float
    best_epoch: int
    best_checkpoint_path: Path
    history: list[EpochMetrics] = field(default_factory=list)


class Trainer:
    """Train a :class:`models.base.BaseClassifier`.

    The trainer owns training state (epochs, best score, history) but holds
    no business logic — keeping ML concerns testable in isolation.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler | None = None,
        device: str = "cpu",
        early_stop_patience: int = 8,
        extra_checkpoint_payload: dict | None = None,
    ) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.early_stop_patience = early_stop_patience
        # Anything serializable (e.g. label_map, feature_stats) the caller
        # wants saved next to the model state dict for reproducible inference.
        self.extra_checkpoint_payload = extra_checkpoint_payload or {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        checkpoint_dir: Path,
        run_name: str,
    ) -> TrainResult:
        """Run the full train/val loop with early stopping and checkpointing."""

        ensure_dir(checkpoint_dir)
        ckpt_path = checkpoint_dir / f"{run_name}_best.pt"

        history: list[EpochMetrics] = []
        best_f1 = -1.0
        best_epoch = 0
        epochs_since_improve = 0

        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            train_loss = self._train_one_epoch(train_loader)
            val_loss, val_f1, val_acc = self._validate(val_loader)
            if self.scheduler is not None:
                self.scheduler.step()
            elapsed = time.perf_counter() - t0

            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_macro_f1=val_f1,
                val_accuracy=val_acc,
                elapsed_s=elapsed,
            )
            history.append(metrics)
            log.info(
                "epoch_done",
                extra={
                    "run": run_name,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_macro_f1": val_f1,
                    "val_accuracy": val_acc,
                    "elapsed_s": elapsed,
                },
            )

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch
                epochs_since_improve = 0
                payload = {
                    "model_state_dict": self.model.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": val_f1,
                    "val_accuracy": val_acc,
                    **self.extra_checkpoint_payload,
                }
                torch.save(payload, ckpt_path)
                log.info(
                    "checkpoint_saved",
                    extra={"path": str(ckpt_path), "val_macro_f1": val_f1},
                )
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= self.early_stop_patience:
                    log.info(
                        "early_stop",
                        extra={
                            "epoch": epoch,
                            "patience": self.early_stop_patience,
                            "best_epoch": best_epoch,
                        },
                    )
                    break

        return TrainResult(
            best_macro_f1=best_f1,
            best_epoch=best_epoch,
            best_checkpoint_path=ckpt_path,
            history=history,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_seen = 0
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            n_seen += x.size(0)
        return total_loss / max(n_seen, 1)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        n_seen = 0
        preds: list[np.ndarray] = []
        truths: list[np.ndarray] = []
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            total_loss += loss.item() * x.size(0)
            n_seen += x.size(0)
            preds.append(logits.argmax(dim=-1).cpu().numpy())
            truths.append(y.cpu().numpy())

        y_pred = np.concatenate(preds)
        y_true = np.concatenate(truths)
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        accuracy = float((y_pred == y_true).mean())
        return total_loss / max(n_seen, 1), macro_f1, accuracy
