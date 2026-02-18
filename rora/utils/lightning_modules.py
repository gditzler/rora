"""PyTorch Lightning modules for training."""

import torch
import torch.nn as nn
import lightning as pl
from torchmetrics import Accuracy
from typing import Optional, Dict, Any


class BaseModelModule(pl.LightningModule):
    """Lightning module for training base models (all parameters trainable)."""

    def __init__(self, model: nn.Module, learning_rate: float = 0.001, num_classes: int = 10):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        self.train_acc(outputs, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(images))
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(images))
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        self.val_acc(outputs, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(images))
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(images))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class AdapterModule(pl.LightningModule):
    """Lightning module for training models with adapters (base frozen, adapters trainable)."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        num_classes: int = 10,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.freeze_base = freeze_base
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

        # Freeze base parameters
        if freeze_base:
            self._freeze_base_parameters()

    def _freeze_base_parameters(self):
        """Freeze base parameters, keep adapters and heads trainable."""
        for name, param in self.model.named_parameters():
            if (
                "adapter" not in name
                and "classifier" not in name
                and "head_a" not in name
                and "head_b" not in name
            ):
                param.requires_grad = False

        # Verify we have trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise ValueError(
                "No trainable parameters found. Make sure adapters or heads are not frozen. "
                f"Model parameters: {[name for name, _ in self.model.named_parameters()]}"
            )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        self.train_acc(outputs, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(images))
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(images))
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        self.val_acc(outputs, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(images))
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(images))
        return loss

    def configure_optimizers(self):
        # Only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.Adam(trainable_params, lr=self.learning_rate)


class MultiTaskAdapterModule(pl.LightningModule):
    """Lightning module for multi-task training with adapters."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.freeze_base = freeze_base
        self.criterion_a = nn.CrossEntropyLoss()
        self.criterion_b = nn.CrossEntropyLoss()
        # Use multiclass for 2-class classification
        # Note: For 2 classes, we can use either "multiclass" or "binary"
        # Using "multiclass" with num_classes=2 works correctly
        self.train_acc_a = Accuracy(task="multiclass", num_classes=2)
        self.train_acc_b = Accuracy(task="multiclass", num_classes=2)
        self.val_acc_a = Accuracy(task="multiclass", num_classes=2)
        self.val_acc_b = Accuracy(task="multiclass", num_classes=2)

        # Freeze base parameters
        if freeze_base:
            self._freeze_base_parameters()

    def _freeze_base_parameters(self):
        """Freeze base parameters, keep adapters and heads trainable."""
        # Freeze linears and activations explicitly
        if hasattr(self.model, "base_model"):
            for linear in self.model.base_model.linears:
                for param in linear.parameters():
                    param.requires_grad = False
            for activation in self.model.base_model.activations:
                for param in activation.parameters():
                    param.requires_grad = False

        # Verify we have trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise ValueError(
                "No trainable parameters found. Make sure adapters or heads are not frozen. "
                f"Model parameters: {[name for name, _ in self.model.named_parameters()]}"
            )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels_dict = batch
        outputs_a, outputs_b = self.model(images)

        # Convert labels from dict to tensors
        # labels_dict values are lists, convert to tensor and ensure correct dtype and device
        if isinstance(labels_dict["even_odd"], list):
            labels_a = torch.tensor(labels_dict["even_odd"], dtype=torch.long, device=outputs_a.device)
        else:
            labels_a = labels_dict["even_odd"].to(dtype=torch.long, device=outputs_a.device)

        if isinstance(labels_dict["bit_parity"], list):
            labels_b = torch.tensor(labels_dict["bit_parity"], dtype=torch.long, device=outputs_b.device)
        else:
            labels_b = labels_dict["bit_parity"].to(dtype=torch.long, device=outputs_b.device)

        loss_a = self.criterion_a(outputs_a, labels_a)
        loss_b = self.criterion_b(outputs_b, labels_b)
        loss = loss_a + loss_b

        self.train_acc_a(outputs_a, labels_a)
        self.train_acc_b(outputs_b, labels_b)

        # Log metrics with consistent arguments
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(images))
        self.log("train_loss_a", loss_a, on_step=True, on_epoch=True, batch_size=len(images))
        self.log("train_loss_b", loss_b, on_step=True, on_epoch=True, batch_size=len(images))
        self.log("train_acc_a", self.train_acc_a, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(images))
        self.log("train_acc_b", self.train_acc_b, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(images))

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels_dict = batch
        outputs_a, outputs_b = self.model(images)

        # Convert labels from dict to tensors
        # labels_dict values are lists, convert to tensor and ensure correct dtype and device
        if isinstance(labels_dict["even_odd"], list):
            labels_a = torch.tensor(labels_dict["even_odd"], dtype=torch.long, device=outputs_a.device)
        else:
            labels_a = labels_dict["even_odd"].to(dtype=torch.long, device=outputs_a.device)

        if isinstance(labels_dict["bit_parity"], list):
            labels_b = torch.tensor(labels_dict["bit_parity"], dtype=torch.long, device=outputs_b.device)
        else:
            labels_b = labels_dict["bit_parity"].to(dtype=torch.long, device=outputs_b.device)

        loss_a = self.criterion_a(outputs_a, labels_a)
        loss_b = self.criterion_b(outputs_b, labels_b)
        loss = loss_a + loss_b

        self.val_acc_a(outputs_a, labels_a)
        self.val_acc_b(outputs_b, labels_b)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(images))
        self.log("val_loss_a", loss_a, on_step=False, on_epoch=True, batch_size=len(images))
        self.log("val_loss_b", loss_b, on_step=False, on_epoch=True, batch_size=len(images))
        self.log("val_acc_a", self.val_acc_a, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(images))
        self.log("val_acc_b", self.val_acc_b, on_step=False, on_epoch=True, prog_bar=True, batch_size=len(images))

        return loss

    def configure_optimizers(self):
        # Only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.Adam(trainable_params, lr=self.learning_rate)
