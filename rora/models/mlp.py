"""MLP model for MNIST experiments."""

import torch
import torch.nn as nn
from typing import Optional, List


class MLPWithAdapter(nn.Module):
    """
    Multi-layer perceptron with adapter support for MNIST classification.

    For RoRA: y = W(R^T x) where R is learned rotation
    For LoRA: y = Wx + (alpha/r) A(B^T x)

    Args:
        input_dim: Input dimension (784 for MNIST)
        hidden_dims: List of hidden layer dimensions
        num_classes: Number of output classes
        use_adapter: Whether to use adapters (RoRA/LoRA)
        adapter_type: Type of adapter ('rora', 'lora', or None)
        adapter_rank: Rank of adapters
        adapter_layers: List of layer indices to add adapters to
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: List[int] = [512, 512],
        num_classes: int = 10,
        use_adapter: bool = False,
        adapter_type: Optional[str] = None,
        adapter_rank: int = 8,
        adapter_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.use_adapter = use_adapter
        self.adapter_type = adapter_type
        self.adapter_rank = adapter_rank

        if adapter_layers is None:
            adapter_layers = list(range(len(hidden_dims)))

        # Build layers
        self.linears = nn.ModuleList()
        self.adapters = nn.ModuleList()
        self.activations = nn.ModuleList()

        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            linear = nn.Linear(prev_dim, hidden_dim)
            self.linears.append(linear)

            # Adapter (if enabled and this layer should have one)
            if use_adapter and i in adapter_layers:
                if adapter_type == "rora":
                    from rora import RoRA

                    adapter = RoRA(d=prev_dim, r=adapter_rank)  # RoRA operates on input
                    self.adapters.append(adapter)
                elif adapter_type == "lora":
                    from rora import LoRA

                    adapter = LoRA(m=hidden_dim, d=prev_dim, r=adapter_rank)  # LoRA operates on input
                    self.adapters.append(adapter)
                else:
                    self.adapters.append(None)
            else:
                self.adapters.append(None)

            # Activation
            self.activations.append(nn.ReLU())

            prev_dim = hidden_dim

        # Classification head
        self.classifier = nn.Linear(prev_dim, num_classes)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run all hidden layers and return pre-classifier feature vector."""
        x = x.view(x.size(0), -1)
        for linear, adapter, activation in zip(self.linears, self.adapters, self.activations):
            if adapter is not None:
                if self.adapter_type == "rora":
                    # RoRA: y = W(R^T x)
                    x = linear(adapter(x))
                elif self.adapter_type == "lora":
                    # LoRA: y = Wx + adapter(x)
                    x = linear(x) + adapter(x)
            else:
                x = linear(x)
            x = activation(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.classifier(self.get_features(x))


class MultiTaskMLP(nn.Module):
    """
    Multi-task MLP with two classification heads.
    Used for Experiment 1: Even/Odd and Bit-Parity tasks.
    """

    def __init__(
        self,
        base_model: nn.Module,
        use_adapter: bool = False,
        adapter_type: Optional[str] = None,
    ):
        super().__init__()
        # Remove the original classifier
        self.base_trunk = nn.Sequential(*list(base_model.children())[:-1])

        # Create two task-specific heads
        # Get the output dimension from the base model's classifier
        if hasattr(base_model, "classifier"):
            head_dim = base_model.classifier.in_features
        else:
            # Fallback: assume 512
            head_dim = 512

        self.head_a = nn.Linear(head_dim, 2)  # Even/Odd binary classification
        self.head_b = nn.Linear(head_dim, 2)  # Bit-Parity binary classification

        # Freeze base trunk
        for param in self.base_trunk.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor):
        """Forward pass returning outputs for both tasks."""
        x = x.view(x.size(0), -1)
        if isinstance(self.base_trunk, nn.Sequential):
            for module in self.base_trunk:
                x = module(x)
        else:
            x = self.base_trunk(x)
        return self.head_a(x), self.head_b(x)
