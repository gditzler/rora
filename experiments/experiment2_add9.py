"""
Experiment 2: MNIST Novel Class Adaptation (Add-9)
Train a base model on digits 0-8 only (9-way classification).
Then adapt to recognize all digits 0-9, where digit 9 is entirely novel.
Compare RoRA vs LoRA.
"""

import torch
import torch.nn as nn
import numpy as np
import lightning as pl
from pathlib import Path
import sys
import pickle
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rora.utils.accelerator import get_accelerator

# Configuration: Set to 2 for fast debugging, 15 for full training
NUM_EPOCHS = 10

from rora.models.mlp import MLPWithAdapter
from rora.utils.data import get_mnist_loaders
from rora.utils.lightning_modules import BaseModelModule, AdapterModule
from rora.utils.callbacks import EpochProgressCallback


def _make_trainer(num_epochs: int, accelerator: str, callbacks: list) -> pl.Trainer:
    """Construct a Trainer with standard experiment settings."""
    return pl.Trainer(
        max_epochs=num_epochs,
        accelerator=accelerator,
        devices=1,
        enable_progress_bar=True,
        logger=False,
        log_every_n_steps=50,
        callbacks=callbacks,
    )


def run_experiment2(ranks=[4, 8, 16], num_seeds=5, accelerator="auto"):
    """Run Experiment 2: MNIST Novel Class Adaptation (Add-9)."""
    print("=" * 80)
    print("Experiment 2: MNIST Novel Class Adaptation (Add-9)")
    print("=" * 80)

    results = {}

    # Step 1: Train base model on digits 0-8 only (single reference run for reporting)
    print("\n" + "=" * 80)
    print("Step 1: Training base model on digits 0-8 only")
    print("=" * 80)
    print("Loading MNIST dataset (digits 0-8)...")
    train_subset = list(range(9))  # Digits 0-8
    train_loader_base, test_loader_base = get_mnist_loaders(
        batch_size=64, train_subset=train_subset, test_subset=train_subset
    )
    print(f"  Training batches: {len(train_loader_base)}, Test batches: {len(test_loader_base)}")

    base_model = MLPWithAdapter(
        input_dim=784, hidden_dims=[512, 512], num_classes=9, use_adapter=False
    )

    print(f"  Initializing base model training (digits 0-8)...")
    print(f"  Training for {NUM_EPOCHS} epochs with learning rate 0.001")
    base_module = BaseModelModule(base_model, learning_rate=0.001, num_classes=9)
    trainer = _make_trainer(NUM_EPOCHS, accelerator, [EpochProgressCallback()])
    print(f"  Starting training...")
    trainer.fit(base_module, train_loader_base, test_loader_base)
    print(f"  ✓ Base model training completed!")

    print(f"  Evaluating base model on test set (digits 0-8)...")
    val_results = trainer.validate(base_module, test_loader_base)
    base_acc = val_results[0]["val_acc"] * 100
    print(f"  ✓ Base model accuracy (digits 0-8): {base_acc:.2f}%")
    results["base_0_8"] = base_acc

    # Step 2: Adapt to all digits 0-9 using RoRA and LoRA
    print("\n" + "=" * 80)
    print("Step 2: Adapting to all digits 0-9")
    print("=" * 80)
    print("Loading full MNIST dataset (digits 0-9)...")
    train_loader_full, test_loader_full = get_mnist_loaders(batch_size=64)
    print(f"  Training batches: {len(train_loader_full)}, Test batches: {len(test_loader_full)}")

    # Collect per-(adapter_type, rank) results across seeds
    collected = {at: {f"rank_{r}": [] for r in ranks} for at in ["rora", "lora"]}

    for seed in range(num_seeds):
        print(f"\nSeed {seed + 1}/{num_seeds}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        pl.seed_everything(seed)

        # Train base model ONCE per seed — reused for all (adapter_type, rank) combos
        print(f"  Training base model (digits 0-8, {NUM_EPOCHS} epochs)...")
        base_model_seed = MLPWithAdapter(
            input_dim=784, hidden_dims=[512, 512], num_classes=9, use_adapter=False
        )
        base_module_seed = BaseModelModule(base_model_seed, learning_rate=0.001, num_classes=9)
        trainer = _make_trainer(NUM_EPOCHS, accelerator, [EpochProgressCallback()])
        trainer.fit(base_module_seed, train_loader_base, test_loader_base)
        print(f"  ✓ Base model training completed")

        # Save linear layer weights for reuse across all (adapter_type, rank) combinations
        saved_linears = [(l.weight.data.clone(), l.bias.data.clone()) for l in base_model_seed.linears]

        for adapter_type in ["rora", "lora"]:
            for rank in ranks:
                print(f"\n  {adapter_type.upper()} rank={rank}:")

                # Create adapter model with 10 output classes
                print(f"    Creating {adapter_type.upper()} adapter model (rank {rank})...")
                adapter_model = MLPWithAdapter(
                    input_dim=784,
                    hidden_dims=[512, 512],
                    num_classes=10,
                    use_adapter=True,
                    adapter_type=adapter_type,
                    adapter_rank=rank,
                )

                # Load saved base weights
                for i, (w, b) in enumerate(saved_linears):
                    adapter_model.linears[i].weight.data = w.clone()
                    adapter_model.linears[i].bias.data = b.clone()

                trainable_params = sum(p.numel() for p in adapter_model.parameters() if p.requires_grad)
                print(f"    Trainable parameters: {trainable_params:,}")

                # Train adapters and new classifier
                print(f"    Training {adapter_type.upper()} adapter ({NUM_EPOCHS} epochs)...")
                adapter_module = AdapterModule(
                    adapter_model, learning_rate=0.001, num_classes=10, freeze_base=True
                )
                trainer = _make_trainer(NUM_EPOCHS, accelerator, [EpochProgressCallback()])
                trainer.fit(adapter_module, train_loader_full, test_loader_full)
                print(f"    ✓ Adapter training completed")

                print(f"    Evaluating on test set (digits 0-9)...")
                val_results = trainer.validate(adapter_module, test_loader_full)
                acc = val_results[0]["val_acc"] * 100
                print(f"    ✓ Accuracy: {acc:.2f}%")
                collected[adapter_type][f"rank_{rank}"].append(acc)

    # Compute statistics and build results dict
    print("\n" + "=" * 80)
    print("Results by method")
    print("=" * 80)
    for adapter_type in ["rora", "lora"]:
        results[adapter_type] = {}
        print(f"\n--- {adapter_type.upper()} ---")
        for rank in ranks:
            rank_results = collected[adapter_type][f"rank_{rank}"]
            mean_acc = np.mean(rank_results)
            std_acc = np.std(rank_results)
            print(f"  Rank {rank}: {mean_acc:.2f} ± {std_acc:.2f}%")
            results[adapter_type][f"rank_{rank}"] = (mean_acc, std_acc)

    # Print summary table
    print("\n" + "=" * 80)
    print("Summary Table")
    print("=" * 80)
    print(f"{'Method':<20} {'Rank 4':<15} {'Rank 8':<15} {'Rank 16':<15}")
    print("-" * 80)
    print(f"{'Base (0-8 only)':<20} {base_acc:>13.2f}% {'N/A':<15} {'N/A':<15}")

    for adapter_type in ["rora", "lora"]:
        adapter_name = adapter_type.upper()
        r4 = results[adapter_type]["rank_4"]
        r8 = results[adapter_type]["rank_8"]
        r16 = results[adapter_type]["rank_16"]
        print(
            f"{adapter_name:<20} "
            f"{r4[0]:>6.2f} ± {r4[1]:>4.2f}%  "
            f"{r8[0]:>6.2f} ± {r8[1]:>4.2f}%  "
            f"{r16[0]:>6.2f} ± {r16[1]:>4.2f}%"
        )

    # Save results to pickle file
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"experiment2_results_{timestamp}.pkl"

    save_data = {
        "experiment": "experiment2_add9",
        "timestamp": timestamp,
        "config": {
            "ranks": ranks,
            "num_seeds": num_seeds,
            "num_epochs": NUM_EPOCHS,
            "accelerator": accelerator,
        },
        "results": results,
    }

    with open(output_file, "wb") as f:
        pickle.dump(save_data, f)

    print(f"\n✓ Results saved to: {output_file}")

    return results


if __name__ == "__main__":
    accelerator = get_accelerator()
    print(f"Using accelerator: {accelerator}")

    results = run_experiment2(ranks=[4, 8, 16], num_seeds=5, accelerator=accelerator)
