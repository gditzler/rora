"""Lightning callbacks for progress tracking."""

import lightning as pl
from typing import Any


class EpochProgressCallback(pl.Callback):
    """Callback to print epoch progress information."""

    def _get_metric_value(self, metric):
        """Safely extract value from metric (handles tensor, float, or other types)."""
        if metric is None:
            return None
        if hasattr(metric, 'item'):
            return metric.item()
        if isinstance(metric, (int, float)):
            return float(metric)
        return None

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Print epoch summary at the end of each training epoch."""
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        # Try to get train metrics
        train_loss = None
        train_acc = None

        # Check various possible metric names
        for key in ["train_loss_epoch", "train_loss"]:
            if key in metrics:
                train_loss = self._get_metric_value(metrics[key])
                break

        for key in ["train_acc_epoch", "train_acc"]:
            if key in metrics:
                train_acc = self._get_metric_value(metrics[key])
                break

        if train_loss is not None:
            if train_acc is not None:
                print(f"  Epoch {epoch + 1}/{trainer.max_epochs}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc*100:.2f}%")
            else:
                print(f"  Epoch {epoch + 1}/{trainer.max_epochs}: Train Loss = {train_loss:.4f}")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Print validation summary at the end of each validation epoch."""
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        val_loss = None
        val_acc = None

        if "val_loss" in metrics:
            val_loss = self._get_metric_value(metrics["val_loss"])

        if "val_acc" in metrics:
            val_acc = self._get_metric_value(metrics["val_acc"])

        if val_loss is not None:
            if val_acc is not None:
                print(f"  Epoch {epoch + 1}/{trainer.max_epochs}: Val Loss = {val_loss:.4f}, Val Acc = {val_acc*100:.2f}%")
            else:
                print(f"  Epoch {epoch + 1}/{trainer.max_epochs}: Val Loss = {val_loss:.4f}")


class MultiTaskEpochProgressCallback(pl.Callback):
    """Callback to print epoch progress for multi-task training."""

    def _get_metric_value(self, metric):
        """Safely extract value from metric (handles tensor, float, or other types)."""
        if metric is None:
            return None
        if hasattr(metric, 'item'):
            return metric.item()
        if isinstance(metric, (int, float)):
            return float(metric)
        return None

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Print epoch summary at the end of each training epoch."""
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        train_loss = None
        train_acc_a = None
        train_acc_b = None

        # Check various possible metric names
        for key in ["train_loss_epoch", "train_loss"]:
            if key in metrics:
                train_loss = self._get_metric_value(metrics[key])
                break

        for key in ["train_acc_a_epoch", "train_acc_a"]:
            if key in metrics:
                train_acc_a = self._get_metric_value(metrics[key])
                break

        for key in ["train_acc_b_epoch", "train_acc_b"]:
            if key in metrics:
                train_acc_b = self._get_metric_value(metrics[key])
                break

        if train_loss is not None:
            if train_acc_a is not None and train_acc_b is not None:
                print(
                    f"  Epoch {epoch + 1}/{trainer.max_epochs}: "
                    f"Train Loss = {train_loss:.4f}, "
                    f"Acc A = {train_acc_a*100:.2f}%, Acc B = {train_acc_b*100:.2f}%"
                )
            else:
                print(f"  Epoch {epoch + 1}/{trainer.max_epochs}: Train Loss = {train_loss:.4f}")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Print validation summary at the end of each validation epoch."""
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        val_loss = None
        val_acc_a = None
        val_acc_b = None

        if "val_loss" in metrics:
            val_loss = self._get_metric_value(metrics["val_loss"])

        if "val_acc_a" in metrics:
            val_acc_a = self._get_metric_value(metrics["val_acc_a"])

        if "val_acc_b" in metrics:
            val_acc_b = self._get_metric_value(metrics["val_acc_b"])

        if val_loss is not None:
            if val_acc_a is not None and val_acc_b is not None:
                print(
                    f"  Epoch {epoch + 1}/{trainer.max_epochs}: "
                    f"Val Loss = {val_loss:.4f}, "
                    f"Acc A = {val_acc_a*100:.2f}%, Acc B = {val_acc_b*100:.2f}%"
                )
            else:
                print(f"  Epoch {epoch + 1}/{trainer.max_epochs}: Val Loss = {val_loss:.4f}")
