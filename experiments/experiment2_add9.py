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

from utils.accelerator import get_accelerator

# Configuration: Set to 2 for fast debugging, 15 for full training
NUM_EPOCHS = 2

from models.mlp import MLPWithAdapter
from utils.data import get_mnist_loaders
from utils.lightning_modules import BaseModelModule, AdapterModule
from utils.callbacks import EpochProgressCallback


def run_experiment2(ranks=[4, 8, 16], num_seeds=5, accelerator="auto"):
    """Run Experiment 2: MNIST Novel Class Adaptation (Add-9)."""
    print("=" * 80)
    print("Experiment 2: MNIST Novel Class Adaptation (Add-9)")
    print("=" * 80)

    results = {}

    # Step 1: Train base model on digits 0-8 only
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

    # Train with Lightning
    print(f"  Initializing base model training (digits 0-8)...")
    print(f"  Training for {NUM_EPOCHS} epochs with learning rate 0.001")
    base_module = BaseModelModule(base_model, learning_rate=0.001, num_classes=9)
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator=accelerator,
        devices=1,
        enable_progress_bar=True,
        logger=False,
        log_every_n_steps=50,
        callbacks=[EpochProgressCallback()],  # Add callback for epoch summaries
    )
    print(f"  Starting training...")
    trainer.fit(base_module, train_loader_base, test_loader_base)
    print(f"  ✓ Base model training completed!")

    # Evaluate base model on 0-8
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
    train_loader_full, test_loader_full = get_mnist_loaders(batch_size=64)  # All digits 0-9
    print(f"  Training batches: {len(train_loader_full)}, Test batches: {len(test_loader_full)}")

    # Modify base model to have 10 classes instead of 9
    base_model.classifier = nn.Linear(512, 10)

    for adapter_type in ["rora", "lora"]:
        print(f"\n--- {adapter_type.upper()} ---")
        adapter_results = {}

        for rank in ranks:
            print(f"\nRank {rank}:")
            rank_results = []

            for seed in range(num_seeds):
                print(f"  Seed {seed + 1}/{num_seeds}")
                torch.manual_seed(seed)
                np.random.seed(seed)
                pl.seed_everything(seed)

                # Create base model (same architecture, trained on 0-8)
                print(f"      Creating base model...")
                base_model_seed = MLPWithAdapter(
                    input_dim=784, hidden_dims=[512, 512], num_classes=9, use_adapter=False
                )

                # Train base on 0-8
                print(f"      Training base model (digits 0-8, {NUM_EPOCHS} epochs)...")
                base_module_seed = BaseModelModule(base_model_seed, learning_rate=0.001, num_classes=9)
                trainer = pl.Trainer(
                    max_epochs=NUM_EPOCHS,
                    accelerator=accelerator,
                    devices=1,
                    enable_progress_bar=True,  # Enable for visibility
                    logger=False,
                    log_every_n_steps=50,
                    callbacks=[EpochProgressCallback()],  # Add callback for epoch summaries
                )
                trainer.fit(base_module_seed, train_loader_base, test_loader_base)
                print(f"      ✓ Base model training completed")

                # Change classifier to 10 classes
                print(f"      Updating classifier to 10 classes...")
                base_model_seed.classifier = nn.Linear(512, 10)

                # Create model with adapter
                print(f"      Creating model with {adapter_type.upper()} adapter (rank {rank})...")
                adapter_model = MLPWithAdapter(
                    input_dim=784,
                    hidden_dims=[512, 512],
                    num_classes=10,
                    use_adapter=True,
                    adapter_type=adapter_type,
                    adapter_rank=rank,
                )

                # Copy base weights (except classifier)
                print(f"      Copying base weights to adapter model...")
                adapter_model.linears[0].weight.data = base_model_seed.linears[0].weight.data.clone()
                adapter_model.linears[0].bias.data = base_model_seed.linears[0].bias.data.clone()
                adapter_model.linears[1].weight.data = base_model_seed.linears[1].weight.data.clone()
                adapter_model.linears[1].bias.data = base_model_seed.linears[1].bias.data.clone()
                trainable_params = sum(p.numel() for p in adapter_model.parameters() if p.requires_grad)
                print(f"      Trainable parameters: {trainable_params:,}")

                # Train adapters and new classifier with Lightning
                print(f"      Training {adapter_type.upper()} adapter (Rank {rank}, {NUM_EPOCHS} epochs)...")
                adapter_module = AdapterModule(
                    adapter_model, learning_rate=0.001, num_classes=10, freeze_base=True
                )
                trainer = pl.Trainer(
                    max_epochs=NUM_EPOCHS,
                    accelerator=accelerator,
                    devices=1,
                    enable_progress_bar=True,  # Enable for visibility
                    logger=False,
                    log_every_n_steps=50,
                    callbacks=[EpochProgressCallback()],  # Add callback for epoch summaries
                )
                trainer.fit(adapter_module, train_loader_full, test_loader_full)
                print(f"      ✓ Adapter training completed")

                # Evaluate on full test set (0-9)
                print(f"      Evaluating on test set (digits 0-9)...")
                val_results = trainer.validate(adapter_module, test_loader_full)
                acc = val_results[0]["val_acc"] * 100
                print(f"      ✓ Accuracy: {acc:.2f}%")
                rank_results.append(acc)

            # Compute statistics
            mean_acc = np.mean(rank_results)
            std_acc = np.std(rank_results)
            print(f"  Mean: {mean_acc:.2f} ± {std_acc:.2f}%")
            adapter_results[f"rank_{rank}"] = (mean_acc, std_acc)

        results[adapter_type] = adapter_results

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
