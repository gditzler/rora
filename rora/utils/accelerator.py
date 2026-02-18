"""Utility functions for accelerator detection."""

import torch


def get_accelerator():
    """
    Detect and return the best available accelerator.
    Priority: CUDA > CPU
    Note: MPS is not supported due to missing operations (e.g., linalg.qr) required by RoRA.

    Returns:
        str: Accelerator name ('gpu' or 'cpu')
    """
    if torch.cuda.is_available():
        return "gpu"
    else:
        # Use CPU instead of MPS because MPS doesn't support torch.linalg.qr
        # which is required by the RoRA implementation
        return "cpu"
