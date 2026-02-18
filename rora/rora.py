"""Rotational Rank Adaptation (RoRA) module implementation."""

import torch
import torch.nn as nn
from torch.linalg import qr
from scipy.linalg import logm
import numpy as np


class RoRA(nn.Module):
    """
    Rotational Rank Adaptation (RoRA) module.

    Implements orthogonal adaptation by right-multiplying frozen base weights
    with a rotation matrix parameterized via low-rank skew-symmetric generators
    in the Lie algebra so(d).

    Args:
        d: Feature dimension
        r: Rank of the adapter (effective subspace dimension is 2r)
        alpha: Scaling factor (for compatibility with LoRA interface)
    """

    def __init__(self, d: int, r: int = 8, alpha: float = 1.0):
        super().__init__()
        self.d = d
        self.r = r
        self.alpha = alpha

        # Low-rank parameters U, V for skew-symmetric generator B = UV^T - VU^T
        self.U = nn.Parameter(torch.randn(d, r) * 0.02)
        self.V = nn.Parameter(torch.randn(d, r) * 0.02)

        # Cache for (Q, Rcore_T) during eval mode; invalidated on train/eval switch
        self._cached_rotation = None

    def train(self, mode: bool = True) -> "RoRA":
        super().train(mode)
        self._cached_rotation = None
        return self

    def _compute_core_rotation(self, dtype: torch.dtype, device: torch.device):
        """
        Compute (Q, Rcore_T) where Q ∈ R^{d×2r} is the subspace basis and
        Rcore_T = exp(-M) ∈ SO(2r) is the core rotation transpose.

        Caches the result during eval mode to avoid redundant QR + matrix_exp per batch.
        """
        if not self.training and self._cached_rotation is not None:
            return self._cached_rotation

        # Thin QR factorization: Q ∈ R^{d×2r} orthonormal basis for span(U, V)
        Q, _ = qr(torch.cat([self.U, self.V], dim=1))  # (d, 2r)

        # Project U, V to subspace
        A = Q.T @ self.U  # (2r, r)
        C = Q.T @ self.V  # (2r, r)

        # Core skew-symmetric matrix M ∈ R^{2r×2r}
        M = A @ C.T - C @ A.T

        # exp(-M) via torch — no CPU round-trip, no device synchronization
        Rcore_T = torch.linalg.matrix_exp(-M.detach().to(device=device, dtype=dtype))

        if not self.training:
            self._cached_rotation = (Q, Rcore_T)

        return Q, Rcore_T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing Algorithm 1 from the paper.

        Args:
            x: Input tensor of shape (..., d)

        Returns:
            Rotated input tensor of same shape
        """
        Q, Rcore_T = self._compute_core_rotation(x.dtype, x.device)
        Delta_R = Rcore_T - torch.eye(2 * self.r, device=x.device, dtype=x.dtype)

        x_shape = x.shape
        x_flat = x.view(-1, self.d)   # (batch, d)
        x_proj = x_flat @ Q           # (batch, 2r)
        x_rot = x_proj @ Delta_R.T    # (batch, 2r) — stays in (batch, features) layout
        x_out = x_flat + x_rot @ Q.T  # (batch, d)

        return x_out.view(x_shape)

    def get_rotation_matrix(self) -> torch.Tensor:
        """
        Get the full rotation matrix R ∈ SO(d).

        Returns:
            Rotation matrix of shape (d, d)
        """
        dtype, device = self.U.dtype, self.U.device
        Q, _ = qr(torch.cat([self.U, self.V], dim=1))
        A = Q.T @ self.U
        C = Q.T @ self.V
        M = A @ C.T - C @ A.T
        Rcore = torch.linalg.matrix_exp(M.detach().to(dtype=dtype))
        I_d = torch.eye(self.d, device=device, dtype=dtype)
        I_2r = torch.eye(2 * self.r, device=device, dtype=dtype)
        return I_d + Q @ (Rcore - I_2r) @ Q.T

    @staticmethod
    def merge(rora1: "RoRA", rora2: "RoRA", w1: float = 0.5, w2: float = 0.5) -> "RoRA":
        """
        Merge two RoRA modules using Algorithm 2 from the paper.

        Args:
            rora1: First RoRA module
            rora2: Second RoRA module
            w1: Weight for first module (default: 0.5)
            w2: Weight for second module (default: 0.5)

        Returns:
            Merged RoRA module
        """
        assert rora1.d == rora2.d, "RoRA modules must have same dimension"
        assert rora1.r == rora2.r, "RoRA modules must have same rank"
        assert abs(w1 + w2 - 1.0) < 1e-6, "Weights must sum to 1"

        # Get subspace bases
        Q1, _ = qr(torch.cat([rora1.U, rora1.V], dim=1))
        Q2, _ = qr(torch.cat([rora2.U, rora2.V], dim=1))

        # Get core rotations via torch (no CPU round-trip needed)
        A1 = Q1.T @ rora1.U
        C1 = Q1.T @ rora1.V
        Rc1_np = torch.linalg.matrix_exp((A1 @ C1.T - C1 @ A1.T).detach()).cpu().numpy()

        A2 = Q2.T @ rora2.U
        C2 = Q2.T @ rora2.V
        Rc2_np = torch.linalg.matrix_exp((A2 @ C2.T - C2 @ A2.T).detach()).cpu().numpy()

        # Step 1: Subspace alignment
        Q1T_Q2_np = (Q1.T @ Q2).detach().cpu().numpy()
        US, Sigma, VS_T = np.linalg.svd(Q1T_Q2_np, full_matrices=False)
        VS = VS_T.T

        # Ensure det(S) = 1
        det_adjust = np.linalg.det(US @ VS.T)
        S_np = US @ np.diag([1.0] * (2 * rora1.r - 1) + [det_adjust]) @ VS.T
        S = torch.from_numpy(S_np).to(Q1.device, dtype=Q1.dtype)

        # Step 2: Transport module 2 to module 1's frame
        Q2_tilde = Q2 @ S
        Rc2_tilde_np = S_np.T @ Rc2_np @ S_np

        # Step 3: Merge subspaces
        Q_bar, _ = qr(Q1 + Q2_tilde)

        # Step 4: Merge core rotations in Lie algebra
        M1_log_np = logm(Rc1_np)
        M2_tilde_log_np = logm(Rc2_tilde_np)
        M_bar_np = w1 * M1_log_np + w2 * M2_tilde_log_np
        # Take real part in case logm returns near-zero imaginary components
        M_bar = torch.from_numpy(np.real(M_bar_np).astype(np.float64)).to(Q1.device, dtype=Q1.dtype)
        Rcore_bar = torch.linalg.matrix_exp(M_bar)

        # Extract low-rank representation from merged rotation
        I_2r = torch.eye(2 * rora1.r, device=Q1.device, dtype=Q1.dtype)
        Delta_Rcore = Rcore_bar - I_2r
        U_svd, S_svd, Vh_svd = torch.linalg.svd(Delta_Rcore)
        # Top-r left and right singular vectors
        U_bar_proj = U_svd[:, :rora1.r] @ torch.diag(torch.sqrt(S_svd[:rora1.r]))
        V_bar_proj = Vh_svd[:rora1.r, :].T @ torch.diag(torch.sqrt(S_svd[:rora1.r]))

        # Map back to full space
        U_bar = Q_bar @ U_bar_proj
        V_bar = Q_bar @ V_bar_proj

        # Create merged module
        merged = RoRA(rora1.d, rora1.r, alpha=rora1.alpha)
        merged.U.data = U_bar
        merged.V.data = V_bar

        return merged
