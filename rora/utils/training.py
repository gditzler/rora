"""Training utilities."""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Dict, Optional, Callable


def train_base_model(
    model: nn.Module,
    train_loader,
    num_epochs: int = 15,
    lr: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: bool = True,
):
    """Train a base model (all parameters)."""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not verbose)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100*correct/total:.2f}%"})

        if verbose:
            print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}, Acc = {100*correct/total:.2f}%")

    return model


def train_adapter_model(
    model: nn.Module,
    train_loader,
    num_epochs: int = 15,
    lr: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    freeze_base: bool = True,
    verbose: bool = True,
    multitask: bool = False,
):
    """Train a model with adapters (freeze base, train adapters and heads)."""
    model = model.to(device)

    # Freeze base parameters
    if freeze_base:
        for name, param in model.named_parameters():
            # Keep adapters, heads (classifier, head_a, head_b), and task-specific heads trainable
            if (
                "adapter" not in name
                and "classifier" not in name
                and "head_a" not in name
                and "head_b" not in name
            ):
                param.requires_grad = False

    # Get trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if len(trainable_params) == 0:
        raise ValueError(
            "No trainable parameters found. Make sure adapters or heads are not frozen. "
            f"Model parameters: {[name for name, _ in model.named_parameters()]}"
        )

    optimizer = optim.Adam(trainable_params, lr=lr)

    if multitask:
        criterion_a = nn.CrossEntropyLoss()
        criterion_b = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct_a = 0
        correct_b = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not verbose)
        for batch in pbar:
            if multitask:
                images, labels_dict = batch
                images = images.to(device)
                # labels_dict is a dict of lists from custom collate function
                labels_a = torch.tensor(labels_dict["even_odd"]).to(device)
                labels_b = torch.tensor(labels_dict["bit_parity"]).to(device)

                optimizer.zero_grad()
                outputs_a, outputs_b = model(images)
                loss_a = criterion_a(outputs_a, labels_a)
                loss_b = criterion_b(outputs_b, labels_b)
                loss = loss_a + loss_b
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted_a = torch.max(outputs_a.data, 1)
                _, predicted_b = torch.max(outputs_b.data, 1)
                total += labels_a.size(0)
                correct_a += (predicted_a == labels_a).sum().item()
                correct_b += (predicted_b == labels_b).sum().item()

                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "acc_a": f"{100*correct_a/total:.2f}%",
                        "acc_b": f"{100*correct_b/total:.2f}%",
                    }
                )
            else:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct = (predicted == labels).sum().item()

                pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100*correct/total:.2f}%"})

        if verbose:
            if multitask:
                print(
                    f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}, "
                    f"Acc A = {100*correct_a/total:.2f}%, Acc B = {100*correct_b/total:.2f}%"
                )
            else:
                print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}, Acc = {100*correct/total:.2f}%")

    return model


def evaluate_model(
    model: nn.Module,
    test_loader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    multitask: bool = False,
):
    """Evaluate a model on test set."""
    model = model.to(device)
    model.eval()

    correct_a = 0
    correct_b = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            if multitask:
                images, labels_dict = batch
                images = images.to(device)
                # labels_dict is a dict of lists from custom collate function
                labels_a = torch.tensor(labels_dict["even_odd"]).to(device)
                labels_b = torch.tensor(labels_dict["bit_parity"]).to(device)

                outputs_a, outputs_b = model(images)
                _, predicted_a = torch.max(outputs_a.data, 1)
                _, predicted_b = torch.max(outputs_b.data, 1)
                total += labels_a.size(0)
                correct_a += (predicted_a == labels_a).sum().item()
                correct_b += (predicted_b == labels_b).sum().item()
            else:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                if multitask:
                    # This shouldn't happen, but handle it
                    pass
                else:
                    correct_a = (predicted == labels).sum().item()

    if multitask:
        acc_a = 100.0 * correct_a / total
        acc_b = 100.0 * correct_b / total
        return {"task_a": acc_a, "task_b": acc_b}
    else:
        acc = 100.0 * correct_a / total
        return {"accuracy": acc}
