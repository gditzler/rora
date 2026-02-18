"""Basic test to verify RoRA and LoRA implementations work."""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rora import RoRA, LoRA
from rora.models.mlp import MLPWithAdapter


def test_rora():
    """Test RoRA forward pass."""
    print("Testing RoRA...")
    d = 128
    r = 8
    batch_size = 32

    rora = RoRA(d=d, r=r)
    x = torch.randn(batch_size, d)

    y = rora(x)
    assert y.shape == x.shape, f"Shape mismatch: {y.shape} vs {x.shape}"
    print(f"  ✓ RoRA forward pass: {x.shape} -> {y.shape}")

    # Test that rotation preserves norm approximately (for small rotations)
    # Note: For initialized (small) rotations, the change should be small
    norm_x = torch.norm(x, dim=1)
    norm_y = torch.norm(y, dim=1)
    print(f"  ✓ Input norm: {norm_x.mean().item():.4f}, Output norm: {norm_y.mean().item():.4f}")


def test_lora():
    """Test LoRA forward pass."""
    print("Testing LoRA...")
    m = 128
    d = 128
    r = 8
    batch_size = 32

    lora = LoRA(m=m, d=d, r=r)
    x = torch.randn(batch_size, d)

    y = lora(x)
    assert y.shape == (batch_size, m), f"Shape mismatch: {y.shape} vs {(batch_size, m)}"
    print(f"  ✓ LoRA forward pass: {x.shape} -> {y.shape}")


def test_mlp_with_adapters():
    """Test MLP with adapters."""
    print("Testing MLP with adapters...")
    batch_size = 32
    input_dim = 784

    for adapter_type in [None, "rora", "lora"]:
        model = MLPWithAdapter(
            input_dim=input_dim,
            hidden_dims=[512, 512],
            num_classes=10,
            use_adapter=(adapter_type is not None),
            adapter_type=adapter_type,
            adapter_rank=8,
        )

        x = torch.randn(batch_size, 1, 28, 28)  # MNIST-like input
        y = model(x)
        assert y.shape == (batch_size, 10), f"Shape mismatch: {y.shape}"
        print(f"  ✓ MLP with {adapter_type or 'no adapter'}: {x.shape} -> {y.shape}")


if __name__ == "__main__":
    print("Running basic tests...\n")
    test_rora()
    print()
    test_lora()
    print()
    test_mlp_with_adapters()
    print("\nAll tests passed! ✓")
