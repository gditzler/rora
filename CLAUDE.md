# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research codebase for **RoRA (Rotational Rank Adaptation)**, a parameter-efficient fine-tuning method that adapts frozen model weights using orthogonal rotations from the Lie group SO(d). The paper compares RoRA against LoRA on MNIST experiments.

## Environment Setup

This project uses `uv` for dependency management:

```bash
uv sync          # Install dependencies
uv run python    # Run Python in the project environment
```

Dependencies: PyTorch, PyTorch Lightning, torchvision, scipy, numpy, tqdm.

## Running Tests

```bash
uv run python test_basic.py
```

## Running Experiments

```bash
# Experiment 1: Multi-task learning (Even/Odd and Bit-Parity on MNIST)
uv run python experiments/experiment1_multitask.py

# Experiment 2: Novel class adaptation (train on digits 0-8, adapt to 0-9)
uv run python experiments/experiment2_add9.py

# Generate LaTeX tables/plots from saved experiment results
uv run python experiments/generate_latex.py
```

Both experiments save timestamped pickle files to `outputs/`. The `NUM_EPOCHS` constant at the top of each experiment file controls training length (2 for fast debugging, 15 for full runs). Experiments auto-detect available accelerator (CPU/GPU/MPS) via `rora/utils/accelerator.py`.

## Code Style

- Line length: 100 characters (black + ruff)
- Target: Python 3.11+
- Remove trailing whitespace from all files (enforced by Cursor rule)

Lint/format:
```bash
uv run ruff check .
uv run black .
```

## Architecture

### Core Adapters (`rora/`)
- `rora.py` — `RoRA` module: parameterizes a rotation matrix R ∈ SO(d) via low-rank skew-symmetric generators (U, V) in the Lie algebra so(d). Forward pass implements Algorithm 1: thin QR → project to subspace → matrix exponential via scipy → apply delta rotation. `RoRA.merge()` implements Algorithm 2: geodesic interpolation of two rotation modules in SO(d) via Lie algebra averaging.
- `lora.py` — `LoRA` module: standard additive low-rank update `y = (alpha/r) * A(B^T x)`, used as a baseline.

### Models (`rora/models/`)
- `mlp.py` — `MLPWithAdapter`: MLP backbone supporting pluggable RoRA or LoRA adapters per hidden layer. For RoRA: `y = W(R^T x)`; for LoRA: `y = Wx + adapter(x)`. The `adapters` ModuleList is parallel to `linears` and `activations`.

### Utilities (`rora/utils/`)
- `lightning_modules.py` — Three Lightning modules: `BaseModelModule` (all params trainable), `AdapterModule` (base frozen, adapters + classifier trainable), `MultiTaskAdapterModule` (base frozen, two-head multi-task training).
- `data.py` — MNIST data loaders with support for digit subsets and multi-task labels (even/odd, bit-parity).
- `callbacks.py` — `EpochProgressCallback` and `MultiTaskEpochProgressCallback` for epoch-level summary logging.
- `accelerator.py` — Auto-detects best available accelerator (MPS, CUDA, or CPU).

### Experiments (`experiments/`)
- `experiment1_multitask.py` — Trains base MLP on 10-way MNIST, then adds RoRA adapters with two heads for Even/Odd and Bit-Parity tasks simultaneously.
- `experiment2_add9.py` — Trains base MLP on digits 0-8, then compares RoRA vs LoRA at adapting to all 10 digits.
- `generate_latex.py` — Reads latest pickle outputs and generates LaTeX tables + TikZ plots → `outputs/*.tex`.

### Key Design Patterns
- `AdapterModule._freeze_base_parameters()` uses name-based freezing: parameters whose names contain `"adapter"`, `"classifier"`, `"head_a"`, or `"head_b"` stay trainable; all others are frozen.
- RoRA's matrix exponential is computed via `scipy.linalg.expm` (off-graph), so gradients flow through U and V only, not through the expm itself.
- Experiment results are always saved as timestamped `.pkl` files; `generate_latex.py` picks the most recent one per experiment.
