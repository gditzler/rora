"""
Experiment 1: MNIST Multi-Task Learning
Train a base MLP on standard 10-way MNIST digit classification.
Then add RoRA modules and two classification heads for simultaneous learning of:
- Task A (Even/Odd): Binary label yA(d) = I[d is odd]
- Task B (Bit-Parity): Binary label yB(d) = popcount(d) mod 2
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
NUM_EPOCHS = 2

from rora.models.mlp import MLPWithAdapter
from rora.utils.data import get_mnist_loaders, get_multitask_mnist_loaders
from rora.utils.lightning_modules import BaseModelModule, MultiTaskAdapterModule
from rora.utils.callbacks import EpochProgressCallback, MultiTaskEpochProgressCallback


class MultiTaskModel(nn.Module):
    """Multi-task model with two classification heads."""

    def __init__(self, base_model, head_dim=512):
        super().__init__()
        # Keep reference to base model to preserve adapter structure
        self.base_model = base_model
        self.head_a = nn.Linear(head_dim, 2)  # Even/Odd
        self.head_b = nn.Linear(head_dim, 2)  # Bit-Parity

        # Freeze base trunk (linears and activations), but keep adapters trainable
        # Freeze linears and activations explicitly
        for linear in self.base_model.linears:
            for param in linear.parameters():
                param.requires_grad = False
        for activation in self.base_model.activations:
            for param in activation.parameters():
                param.requires_grad = False
        # Keep adapters trainable (they're in self.base_model.adapters)
        # Keep heads trainable (they're self.head_a and self.head_b)

    def forward(self, x):
        features = self.base_model.get_features(x)
        return self.head_a(features), self.head_b(features)


def run_experiment1(ranks=[4, 8, 16], num_seeds=5, accelerator="auto"):
    """Run Experiment 1: MNIST Multi-Task Learning."""
    print("=" * 80)
    print("Experiment 1: MNIST Multi-Task Learning")
    print("=" * 80)

    results = {}

    # Step 1: Train base model on standard 10-way MNIST classification
    print("\n" + "=" * 80)
    print("Step 1: Training base model on 10-way MNIST classification")
    print("=" * 80)
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_loaders(batch_size=64)
    print(f"  Training batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    base_model = MLPWithAdapter(
        input_dim=784, hidden_dims=[512, 512], num_classes=10, use_adapter=False
    )

    # Train with Lightning
    print(f"  Initializing base model training...")
    print(f"  Training for {NUM_EPOCHS} epochs with learning rate 0.001")
    base_module = BaseModelModule(base_model, learning_rate=0.001, num_classes=10)
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator=accelerator,
        devices=1,
        enable_progress_bar=True,
        logger=False,
        log_every_n_steps=50,  # Log every 50 steps
        callbacks=[EpochProgressCallback()],  # Add callback for epoch summaries
    )
    print(f"  Starting training...")
    trainer.fit(base_module, train_loader, test_loader)
    print(f"  ✓ Base model training completed!")

    # Evaluate base model
    print(f"  Evaluating base model on test set...")
    val_results = trainer.validate(base_module, test_loader)
    base_acc = val_results[0]["val_acc"]
    print(f"  ✓ Base model accuracy: {base_acc*100:.2f}%")
    results["base"] = base_acc * 100

    # Step 2: Create multi-task models with RoRA adapters
    print("\n" + "=" * 80)
    print("Step 2: Training multi-task models with RoRA adapters")
    print("=" * 80)
    print("Loading multi-task MNIST dataset...")
    train_multitask_loader, test_multitask_loader = get_multitask_mnist_loaders(batch_size=64)
    print(f"  Training batches: {len(train_multitask_loader)}, Test batches: {len(test_multitask_loader)}")

    for rank in ranks:
        print(f"\n--- RoRA Rank {rank} ---")
        rank_results = []

        for seed in range(num_seeds):
            print(f"\nSeed {seed + 1}/{num_seeds}")
            torch.manual_seed(seed)
            np.random.seed(seed)
            pl.seed_everything(seed)

            # Create base model (same architecture)
            print(f"    Creating base model and copying weights...")
            base_model_seed = MLPWithAdapter(
                input_dim=784, hidden_dims=[512, 512], num_classes=10, use_adapter=False
            )
            # Load the trained weights (copy from original base)
            base_model_seed.load_state_dict(base_model.state_dict())
            print(f"    Base weights copied successfully")

            # Create model with RoRA adapters
            print(f"    Creating model with RoRA adapters (rank {rank})...")
            base_with_rora = MLPWithAdapter(
                input_dim=784,
                hidden_dims=[512, 512],
                num_classes=10,
                use_adapter=True,
                adapter_type="rora",
                adapter_rank=rank,
            )

            # Copy base weights
            print(f"    Copying base weights to adapter model...")
            base_with_rora.linears[0].weight.data = base_model_seed.linears[0].weight.data.clone()
            base_with_rora.linears[0].bias.data = base_model_seed.linears[0].bias.data.clone()
            base_with_rora.linears[1].weight.data = base_model_seed.linears[1].weight.data.clone()
            base_with_rora.linears[1].bias.data = base_model_seed.linears[1].bias.data.clone()

            # Create multi-task model
            print(f"    Creating multi-task model with two classification heads...")
            multitask_model = MultiTaskModel(base_with_rora, head_dim=512)
            trainable_params = sum(p.numel() for p in multitask_model.parameters() if p.requires_grad)
            print(f"    Trainable parameters: {trainable_params:,}")

            # Train adapters and heads with Lightning
            print(f"    Initializing multi-task training (Rank {rank}, Seed {seed + 1})...")
            multitask_module = MultiTaskAdapterModule(
                multitask_model, learning_rate=0.001, freeze_base=True
            )
            trainer = pl.Trainer(
                max_epochs=NUM_EPOCHS,
                accelerator=accelerator,
                devices=1,
                enable_progress_bar=True,  # Enable progress bar for visibility
                logger=False,
                log_every_n_steps=50,  # Log every 50 steps
                callbacks=[MultiTaskEpochProgressCallback()],  # Add callback for epoch summaries
            )
            print(f"    Starting multi-task training ({NUM_EPOCHS} epochs)...")
            trainer.fit(multitask_module, train_multitask_loader, test_multitask_loader)
            print(f"    ✓ Multi-task training completed!")

            # Evaluate
            print(f"    Evaluating on test set...")
            val_results = trainer.validate(multitask_module, test_multitask_loader)
            acc_a = val_results[0]["val_acc_a"] * 100
            acc_b = val_results[0]["val_acc_b"] * 100
            print(f"    ✓ Task A (Even/Odd): {acc_a:.2f}%")
            print(f"    ✓ Task B (Bit-Parity): {acc_b:.2f}%")
            rank_results.append((acc_a, acc_b))

        # Compute statistics
        accs_a = [r[0] for r in rank_results]
        accs_b = [r[1] for r in rank_results]
        mean_a = np.mean(accs_a)
        std_a = np.std(accs_a)
        mean_b = np.mean(accs_b)
        std_b = np.std(accs_b)

        print(f"\nRank {rank} Results:")
        print(f"  Task A: {mean_a:.2f} ± {std_a:.2f}%")
        print(f"  Task B: {mean_b:.2f} ± {std_b:.2f}%")

        results[f"rora_rank_{rank}"] = {"task_a": (mean_a, std_a), "task_b": (mean_b, std_b)}

    # Print summary table
    print("\n" + "=" * 80)
    print("Summary Table")
    print("=" * 80)
    print(f"{'Method':<20} {'Task A Accuracy':<20} {'Task B Accuracy':<20}")
    print("-" * 80)
    print(f"{'Base (frozen)':<20} {results['base']:>18.2f}% {'N/A':>20}")
    for rank in ranks:
        r = results[f"rora_rank_{rank}"]
        print(
            f"{'RoRA Rank ' + str(rank):<20} "
            f"{r['task_a'][0]:>6.2f} ± {r['task_a'][1]:>5.2f}%  "
            f"{r['task_b'][0]:>6.2f} ± {r['task_b'][1]:>5.2f}%"
        )

    # Save results to pickle file
    output_dir = Path(__file__).parent.parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"experiment1_results_{timestamp}.pkl"

    save_data = {
        "experiment": "experiment1_multitask",
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

    results = run_experiment1(ranks=[4, 8, 16], num_seeds=5, accelerator=accelerator)
