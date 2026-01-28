"""Rotational Rank Adaptation (RoRA) module implementation."""

import torch
import torch.nn as nn
from torch.linalg import qr
from scipy.linalg import expm, logm
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementing Algorithm 1 from the paper.

        Args:
            x: Input tensor of shape (..., d)

        Returns:
            Rotated input tensor of same shape
        """
        # Step 1: Thin QR factorization
        # Q ∈ R^{d×2r} is orthonormal basis for span(U, V)
        Q, _ = qr(torch.cat([self.U, self.V], dim=1))
        # Q shape: (d, 2r)

        # Step 2: Project U, V to subspace
        A = Q.T @ self.U  # (2r, r)
        C = Q.T @ self.V  # (2r, r)

        # Step 3: Compute core skew-symmetric matrix M ∈ R^{2r×2r}
        M = A @ C.T - C @ A.T  # (2r, 2r)

        # Step 4: Compute R^T in core space via matrix exponential
        # Convert to numpy for scipy.linalg.expm
        M_np = M.detach().cpu().numpy()
        Rcore_T_np = expm(-M_np)  # exp(-M) gives R^T
        Rcore_T = torch.from_numpy(Rcore_T_np).to(x.device, dtype=x.dtype)

        # Step 5: Compute delta rotation
        Delta_R = Rcore_T - torch.eye(2 * self.r, device=x.device, dtype=x.dtype)

        # Step 6: Project input to subspace
        x_shape = x.shape
        x_flat = x.view(-1, self.d)  # (batch, d)
        x_proj = x_flat @ Q  # (batch, 2r)

        # Step 7: Apply rotation in subspace
        # According to Algorithm 1: xrot = Delta_R @ xproj
        # Delta_R is (2r, 2r), x_proj is (batch, 2r)
        # We need to apply Delta_R from the left: (batch, 2r) -> (2r, batch) -> apply -> (2r, batch) -> (batch, 2r)
        x_rot = torch.matmul(Delta_R, x_proj.T).T  # (batch, 2r)

        # Step 8: Map back and add residual
        x_rot_full = x_rot @ Q.T  # (batch, d)
        x_out = x_flat + x_rot_full

        return x_out.view(x_shape)

    def get_rotation_matrix(self) -> torch.Tensor:
        """
        Get the full rotation matrix R ∈ SO(d).

        Returns:
            Rotation matrix of shape (d, d)
        """
        Q, _ = qr(torch.cat([self.U, self.V], dim=1))
        A = Q.T @ self.U
        C = Q.T @ self.V
        M = A @ C.T - C @ A.T

        M_np = M.detach().cpu().numpy()
        Rcore_np = expm(M_np)
        Rcore = torch.from_numpy(Rcore_np).to(self.U.device, dtype=self.U.dtype)

        # Full rotation: R = I + Q(Rcore - I)Q^T
        I_d = torch.eye(self.d, device=self.U.device, dtype=self.U.dtype)
        I_2r = torch.eye(2 * self.r, device=self.U.device, dtype=self.U.dtype)
        R = I_d + Q @ (Rcore - I_2r) @ Q.T

        return R

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

        # Get core rotations
        A1 = Q1.T @ rora1.U
        C1 = Q1.T @ rora1.V
        M1_np = (A1 @ C1.T - C1 @ A1.T).detach().cpu().numpy()
        Rc1_np = expm(M1_np)

        A2 = Q2.T @ rora2.U
        C2 = Q2.T @ rora2.V
        M2_np = (A2 @ C2.T - C2 @ A2.T).detach().cpu().numpy()
        Rc2_np = expm(M2_np)

        # Step 1: Subspace alignment
        Q1T_Q2 = Q1.T @ Q2
        Q1T_Q2_np = Q1T_Q2.detach().cpu().numpy()
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
        Rcore_bar_np = expm(M_bar_np)

        # Create merged module by finding U, V that generate the merged rotation
        # We need to find U_bar, V_bar such that the generated rotation matches
        # This is a bit tricky - we'll use the merged Q and Rcore to reconstruct
        Rcore_bar = torch.from_numpy(Rcore_bar_np).to(Q1.device, dtype=Q1.dtype)

        # For simplicity, we'll extract a low-rank approximation
        # The merged rotation is R = I + Q_bar (Rcore_bar - I) Q_bar^T
        # We want to find U_bar, V_bar that approximate this
        I_2r = torch.eye(2 * rora1.r, device=Q1.device, dtype=Q1.dtype)
        Delta_Rcore = Rcore_bar - I_2r

        # Use SVD to get low-rank representation
        U_svd, S_svd, V_svd_T = torch.linalg.svd(Delta_Rcore)
        # Take top r singular vectors
        U_bar_proj = U_svd[:, :rora1.r] @ torch.diag(torch.sqrt(S_svd[:rora1.r]))
        V_bar_proj = V_svd[:, :rora1.r] @ torch.diag(torch.sqrt(S_svd[:rora1.r]))

        # Map back to full space
        U_bar = Q_bar @ U_bar_proj
        V_bar = Q_bar @ V_bar_proj

        # Create merged module
        merged = RoRA(rora1.d, rora1.r, alpha=rora1.alpha)
        merged.U.data = U_bar
        merged.V.data = V_bar

        return merged
