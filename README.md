# Rotational Rank Adaptation (RoRA)

Implementation of experiments from the paper "Rotational Rank Adaptation (RoRA): Spectral–Orthogonal Geometry for Robust Model Merging".

## Setup

This project uses `uv` for dependency management and execution. If you don't have `uv` installed, you can install it from [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv).

### Create Virtual Environment and Install Dependencies

```bash
# Create virtual environment
uv venv

# Install project and dependencies
uv pip install -e .
```

The virtual environment will be created in `.venv/` and all dependencies will be installed automatically.

## Running Experiments

All experiments can be run using `uv run`, which automatically uses the virtual environment:

### Experiment 1: MNIST Multi-Task Learning

Train a base MLP on standard 10-way MNIST digit classification, then add RoRA modules and two classification heads for simultaneous learning of:
- Task A (Even/Odd): Binary label yA(d) = I[d is odd]
- Task B (Bit-Parity): Binary label yB(d) = popcount(d) mod 2

```bash
uv run python experiments/experiment1_multitask.py
```

### Experiment 2: MNIST Novel Class Adaptation (Add-9)

Train a base model on digits 0–8 only, then adapt to recognize all digits 0–9, where digit 9 is entirely novel. Compares RoRA vs LoRA.

```bash
uv run python experiments/experiment2_add9.py
```

## Testing

Run the basic functionality tests:

```bash
uv run python test_basic.py
```

## Alternative: Using the Virtual Environment Directly

If you prefer to activate the virtual environment manually:

```bash
# Activate the environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# Then run scripts normally
python experiments/experiment1_multitask.py
python experiments/experiment2_add9.py
python test_basic.py
```

## Project Structure

```
paper-rora/
├── rora/              # RoRA and LoRA implementations
├── models/            # Model architectures (MLP)
├── utils/             # Data loading and training utilities
├── experiments/       # Experiment scripts
├── .venv/             # Virtual environment (created by uv)
└── pyproject.toml     # Project configuration and dependencies
```

## Implementation Details

- **RoRA**: Implements orthogonal adaptation via low-rank skew-symmetric generators in the Lie algebra so(d)
- **LoRA**: Standard additive low-rank adaptation for comparison
- **MLP**: Multi-layer perceptron with adapter support for MNIST experiments
- **Lightning Modules**: Training is implemented using PyTorch Lightning for better code organization and reusability

## Architecture

The codebase is organized into several key components:

- **`rora/`**: Core adapter implementations (RoRA and LoRA)
- **`models/`**: Model architectures (MLP with adapter support)
- **`utils/`**: 
  - `data.py`: Data loading utilities
  - `lightning_modules.py`: PyTorch Lightning modules for training
- **`experiments/`**: Experiment scripts using Lightning

### Lightning Modules

The training code uses PyTorch Lightning modules for clean separation of concerns:

- **`BaseModelModule`**: For training base models (all parameters trainable)
- **`AdapterModule`**: For training with adapters (base frozen, adapters trainable)
- **`MultiTaskAdapterModule`**: For multi-task training with adapters

These modules handle:
- Automatic optimization configuration
- Training/validation step logic
- Metric tracking (accuracy, loss)
- Parameter freezing logic

## Dependencies

All dependencies are specified in `pyproject.toml`:
- PyTorch (>=2.0.0)
- torchvision (>=0.15.0)
- PyTorch Lightning (>=2.6.0) - Latest stable version
- numpy (>=1.24.0)
- scipy (>=1.10.0)
- tqdm (>=4.65.0)
