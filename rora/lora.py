"""LoRA (Low-Rank Adaptation) module implementation for comparison."""

import torch
import torch.nn as nn


class LoRA(nn.Module):
    """
    Low-Rank Adaptation (LoRA) module.

    Implements additive low-rank updates: y = Wx + (alpha/r) * A(B^T x)

    Args:
        m: Output dimension
        d: Input dimension
        r: Rank of the adapter
        alpha: Scaling factor
    """

    def __init__(self, m: int, d: int, r: int = 8, alpha: float = 1.0):
        super().__init__()
        self.m = m
        self.d = d
        self.r = r
        self.alpha = alpha

        # Low-rank matrices A and B
        self.A = nn.Parameter(torch.randn(m, r) * 0.02)
        self.B = nn.Parameter(torch.randn(d, r) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: y = (alpha/r) * A(B^T x)

        Args:
            x: Input tensor of shape (..., d)

        Returns:
            Output tensor of shape (..., m)
        """
        x_shape = x.shape
        x_flat = x.view(-1, self.d)  # (batch, d)
        output = (self.alpha / self.r) * (x_flat @ self.B @ self.A.T)
        return output.view(*x_shape[:-1], self.m)
