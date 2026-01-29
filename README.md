# Neural Equilibria for Long-Term Prediction of Nonlinear Conservation Laws  
**Efficient Implementation and Experimental Code Repository**

This repository contains the code for implementing [**neural equilibria models for long-term prediction of nonlinear conservation laws (arXiv)**](https://arxiv.org/abs/2501.06933).  
These models are designed to predict complex nonlinear dynamics governed by conservation laws, ensuring long-term stability and accuracy.

---

## Citation  
If you use this repository, please cite the paper using the following BibTeX:

```bibtex
@article{benitez2025neural,
  title={Neural equilibria for long-term prediction of nonlinear conservation laws},
  author={Benitez, J and Guo, Junyi and Hegazy, Kareem and Dokmani{\'c}, Ivan and Mahoney, Michael W and de Hoop, Maarten V},
  journal={arXiv preprint arXiv:2501.06933},
  year={2025}
}
```

## Getting Started

### Prerequisites
Ensure you have the following prerequisites installed:

* Python >= 3.12
* PyTorch >= 2.9.1 (with CUDA support recommended)
* Required libraries (provided in pyproject.toml)

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/JALB-epsilon/NeurDE.git
    cd NeurDE
    ```

2.  **Install dependencies:**

    Using uv (recommended):
    ```bash
    uv pip install -e .
    ```

    Or using pip:
    ```bash
    pip install -e .
    ```

3. **Verify installation:**

    ```bash
    python -c "import torch; import hydra; print('Installation successful')"
    ```

## Repository Structure

The repository has been refactored into a unified structure that supports all three experimental cases (Cylinder, Cylinder_faster, and SOD_shock_tube) through a single codebase with Hydra-based configuration management.

### Directory Structure
```
NeurDE/
├── configs/                 # Hydra configuration files
│   ├── case/               # Case-specific configs (cylinder, cylinder_faster, sod_shock_tube)
│   ├── model/              # Model architecture configs
│   ├── optimizer/          # Optimizer configs
│   ├── scheduler/          # Learning rate scheduler configs
│   ├── dataset/            # Dataset configs (stage1, stage2)
│   ├── training/          # Training configs (stage1, stage2)
│   ├── logging/            # Experiment tracking (wandb, mlflow)
│   └── config.yaml        # Main configuration file
├── model/                  # Unified model architecture
│   └── model.py           # NeurDE model definition
├── training/              # Training infrastructure
│   ├── base.py            # Base trainer class
│   ├── stage1_trainer.py  # Stage 1 training class
│   └── stage2_trainer.py  # Stage 2 training class
├── utils/                 # Unified utilities
│   ├── core.py            # Core utilities (seeding, tensor operations)
│   ├── data_io.py         # Data loading utilities
│   ├── datasets.py        # PyTorch Dataset classes
│   ├── loss.py            # Loss functions
│   ├── optimizer.py       # Optimizer and scheduler dispatch
│   ├── plotting.py        # Plotting utilities
│   ├── case_specific.py   # Case-specific utilities (TVD scheduler)
│   ├── solver/            # Unified solver implementations
│   │   ├── __init__.py    # Solver factory (create_solver)
│   │   ├── base.py        # Base solver class
│   │   ├── cylinder.py    # Cylinder solver
│   │   └── sod.py         # SOD solver
│   └── phys/              # Physics computation modules
│       ├── getFeq.py      # F equilibrium distribution
│       ├── getGeq.py      # G equilibrium distribution
│       ├── getPolyGeq.py  # Polynomial G equilibrium
│       └── multinv.py     # Matrix inversion utilities
├── train_stage1.py        # Stage 1 training entry point
├── train_stage2.py        # Stage 2 training entry point
├── eval.py                # Evaluation entry point
├── pyproject.toml         # Project dependencies
└── README.md
```

### Case Descriptions

| Case | Description |
|------|-------------|
| `cylinder` | Implements the Cylinder case with dense matrix inversion for obstacles and boundary conditions. Uses CSR sparse format for main computations. |
| `cylinder_faster` | Faster alternative to the Cylinder case using sparse solvers throughout (CSC format). May exhibit reduced accuracy and stability in rollouts compared to the standard Cylinder case. |
| `sod_shock_tube` | Contains the code for the Sod shock tube experiment. Supports two cases (case_number: 1 or 2), with optional TVD loss for case 2. |

## Running the Code

The code uses Hydra for configuration management, allowing easy parameter selection and overrides via command-line arguments.

### Configuration Overview

All parameters are configured through Hydra configs in the `configs/` directory:
- **Case configs** (`configs/case/`): Physics parameters and solver settings
- **Model configs** (`configs/model/`): Architecture parameters
- **Training configs** (`configs/training/`): Training hyperparameters
- **Optimizer/Scheduler configs**: Optimization settings
- **Logging configs** (`configs/logging/`): Experiment tracking (WandB or MLflow) and terminal log intervals

### Data Generation

Before training, you need to generate simulation data. The `generate_data.py` script runs simulations using the unified solvers and saves the data to HDF5 files.

**Basic usage:**
```bash
# Generate data for Cylinder case (1000 steps by default)
python generate_data.py case=cylinder

# Generate data for Cylinder_faster case
python generate_data.py case=cylinder_faster

# Generate data for SOD case 1
python generate_data.py case=sod_shock_tube case_number=1

# Generate data for SOD case 2
python generate_data.py case=sod_shock_tube case_number=2
```

**With custom number of steps:**
```bash
# Generate 2000 steps of data
python generate_data.py case=cylinder steps=2000

# Generate data with compilation enabled (faster)
python generate_data.py case=cylinder steps=1000 compile=true

# Adjust GPU batch size for better memory utilization (default: 100)
python generate_data.py case=cylinder steps=1000 transfer_batch_size=200
```

The generated data will be saved to:
- `data_base/cylinder_case.h5` for Cylinder cases
- `data_base/SOD_case1.h5` or `data_base/SOD_case2.h5` for SOD cases

**Note:** The data generation process can take a significant amount of time depending on the number of steps and grid size. For testing purposes, you may want to use fewer steps initially. The script uses batched GPU-to-CPU transfers to improve GPU memory utilization - adjust `transfer_batch_size` based on your GPU memory (higher values = better utilization but more GPU memory).

### Stage 1 Training

Stage 1 trains the model to predict equilibrium G distribution function (Geq) from macroscopic variables (rho, ux, uy, T).

**Basic usage:**
```bash
# Uses default training=stage1 and dataset=stage1 from config
python train_stage1.py case=cylinder
python train_stage1.py case=cylinder_faster
python train_stage1.py case=sod_shock_tube case_number=1
```

**With parameter overrides:**
```bash
# Override device, batch size, and number of samples
python train_stage1.py case=cylinder device=0 num_samples=1000 batch_size=64

# Override model architecture
python train_stage1.py case=cylinder model.hidden_dim=64 model.num_layers=5

# Override optimizer and scheduler
python train_stage1.py case=cylinder optimizer=adamw scheduler=onecyclelr
```

### Stage 2 Training

Stage 2 trains the model on rollout sequences with solver integration.

**Basic usage:**
```bash
# Must specify training=stage2 and dataset=stage2
python train_stage2.py case=cylinder training=stage2 dataset=stage2
python train_stage2.py case=cylinder_faster training=stage2 dataset=stage2
python train_stage2.py case=sod_shock_tube case_number=1 training=stage2 dataset=stage2
```

**With TVD loss (SOD case 2):**
```bash
python train_stage2.py case=sod_shock_tube case_number=2 training=stage2 dataset=stage2 training.tvd.enabled=true
```

**SOD case:**
```bash
python train_stage2.py case=sod_shock_tube case_number=1 training=stage2 dataset=stage2
```

**With pretrained model:**
```bash
python train_stage2.py case=cylinder training=stage2 dataset=stage2 pretrained_path=results/stage1/best_model_epoch_100_top_1_loss_0.001234.pt
```

**With parameter overrides:**
```bash
# Override rollout steps and learning rate
python train_stage2.py case=cylinder training=stage2 dataset=stage2 training.N=25 training.lr=1e-4

# Override device and compilation
python train_stage2.py case=cylinder training=stage2 dataset=stage2 device=0 compile=true
```

### Configuration Examples

**Cylinder case with custom settings:**
```bash
python train_stage1.py \
    case=cylinder \
    training=stage1 \
    dataset=stage1 \
    device=0 \
    num_samples=500 \
    batch_size=32 \
    model.hidden_dim=64 \
    model.num_layers=4 \
    optimizer.optimizer_type=AdamW \
    optimizer.lr=1e-3 \
    scheduler.scheduler_type=CosineAnnealingWarmRestarts \
    compile=true
```

**SOD case 2 with TVD:**
```bash
python train_stage2.py \
    case=sod_shock_tube \
    case_number=2 \
    training=stage2 \
    dataset=stage2 \
    training.tvd.enabled=true \
    training.tvd.weight=15.0 \
    device=0
```

### Key Configuration Parameters

**Case Selection:**
- `case`: One of `cylinder`, `cylinder_faster`, or `sod_shock_tube`
- `case_number`: For SOD case, specify `1` or `2`

**Training Stage:**
- `training=stage1`: Equilibrium state prediction (default for train_stage1.py)
- `training=stage2`: Rollout training with solver integration (required for train_stage2.py)
- `dataset=stage1`: Dataset config for Stage 1 (default)
- `dataset=stage2`: Dataset config for Stage 2 (required for train_stage2.py)

**Device and Performance:**
- `device`: GPU index (e.g., `0`, `1`, `2`) or `-1` for CPU
- `compile`: Enable PyTorch compilation for faster execution (`true`/`false`)

**Data:**
- `num_samples`: Number of training samples
- `batch_size`: Batch size for training

**Model:**
- `model.hidden_dim`: Hidden dimension size
- `model.num_layers`: Number of layers
- `model.activation`: Activation function (`relu` or `tanh`)

**Training:**
- `training.epochs`: Number of training epochs
- `training.lr`: Learning rate
- `training.N`: Number of rollout steps (Stage 2 only)
- `training.tvd.enabled`: Enable TVD loss (SOD case 2, Stage 2 only)
- `training.tvd.weight`: TVD loss weight
- `training.validation.dataset_size`: Size of validation dataset (Stage 2 only, always enabled)
- `training.detach_after_streaming`: Detach gradients after streaming (SOD, Stage 2 only)

**Dataset:**
- `dataset.batch_size`: Batch size for DataLoader
- `dataset.shuffle`: Whether to shuffle data
- `dataset.num_workers`: Number of DataLoader workers
- `dataset.pin_memory`: Whether to pin memory for faster GPU transfer

### Experiment Tracking

Training can log metrics and hyperparameters to an experiment-tracking backend. Two backends are supported: **Weights & Biases (WandB)** and **MLflow**. The backend and its options are controlled by the `logging` config group in `configs/logging/`.

**Selecting a backend**

- Use the `logging` default in `configs/config.yaml`, or override from the command line:
  ```bash
  python train_stage1.py case=cylinder logging=wandb
  python train_stage1.py case=cylinder logging=mlflow
  ```
- Shared options for all backends:
  - `log_to_screen`: whether to print epoch/batch summaries to the terminal
  - `terminal_batch_log_interval`: log to terminal every N batches (e.g. `50`)
  - `tracker.enabled`: set to `false` to disable sending metrics to the backend
  - `tracker.batch_log_interval`: send metrics to the backend every N batches (can differ from terminal interval)
  - `tracker.project`: experiment/project name
  - `tracker.run_name`, `tracker.run_tag`: resolved from global `run_name` and `run_tag` in the main config

**WandB**

- Config file: `configs/logging/wandb.yaml`
- Install: `uv sync --group wandb` (or use the project’s optional dependency).
- Set in the logging config’s `tracker` section:
  - `entity`: your WandB username or team
  - `project`: WandB project name
- Log in once (e.g. `wandb login`) or set `WANDB_API_KEY` in the environment.

**MLflow**

- Config file: `configs/logging/mlflow.yaml`.
- Install: `uv sync --group mlflow`.
- **Tracking URI** (required for a remote server): set `tracker.tracking_uri` in the logging config, e.g.:
  - `http://localhost:5000` for a local server started with `mlflow server --port 5000`
  - `https://your-server:5000` for a remote MLflow server
- For a **remote server with HTTPS and self-signed certificates**, set in the logging config:
  - `tracker.insecure_tls: true` so the client skips TLS verification (use only in controlled environments).
- **Authentication**: username can be set in the config under `tracker.username`. The password must be provided via the environment variable `MLFLOW_TRACKING_PASSWORD` (never put the password in the config file). The trainer loads a `.env` file at startup (using `python-dotenv` with `override=False`), so you can put `MLFLOW_TRACKING_PASSWORD=your_password` in a `.env` file in the project root instead of exporting it in the shell.
- See [MLflow self-hosting](https://mlflow.org/docs/latest/self-hosting/) for running your own tracking server.

**Disabling experiment tracking**

- Set `tracker.enabled=false` in your logging config, e.g. `python train_stage1.py logging=wandb logging.tracker.enabled=false`. Terminal logging is independent and controlled by `log_to_screen` and `terminal_batch_log_interval`.

## Important Notes

### Implementation Details

1. **Streaming Operation**: The code uses pre-computed indices for streaming operations instead of `torch.roll` to avoid loops over discrete velocities.

2. **Matrix Inversion**: 
   - **Cylinder**: Uses dense numpy inversion (`np.linalg.inv`) for boundary conditions and obstacles, sparse CSR solver for main computations
   - **Cylinder_faster**: Uses sparse CSC solver throughout for better performance
   - **SOD**: Uses sparse CSR solver (no obstacle/BC handling needed)

3. **Gradient Detaching**: 
   - **Cylinder/Cylinder_faster**: Gradients are not detached after streaming
   - **SOD**: Gradients are detached after streaming by default

5. **TVD Loss**: Available for SOD case 2, with configurable weight and milestone-based scheduling.

6. **Validation**: Optional validation loop available for SOD cases, using separate validation dataset.

## Testing

### Evaluation

Run inference/rollout with a trained model and generate visualization plots.

**Basic usage:**
```bash
python eval.py case=cylinder trained_path=results/stage2/best_model_epoch_100_top_1_loss_0.001234.pt
python eval.py case=cylinder_faster trained_path=results/stage2/best_model_epoch_100_top_1_loss_0.001234.pt
python eval.py case=sod_shock_tube case_number=1 trained_path=results/case1/stage2/best_model_1_epoch_100_top_1_val_loss_0.001234.pt
```

**With parameter overrides:**
```bash
# Override device, number of steps, and initial condition
python eval.py case=cylinder trained_path=results/stage2/best_model.pt device=0 num_steps=1000 init_cond=500

# Disable obstacle handling (Cylinder cases)
python eval.py case=cylinder trained_path=results/stage2/best_model.pt with_obs=false

# Enable compilation
python eval.py case=cylinder trained_path=results/stage2/best_model.pt compile=true

# Custom output directory
python eval.py case=cylinder trained_path=results/stage2/best_model.pt output_dir=my_images
```

**Key Configuration Parameters:**
- `trained_path`: Path to trained model checkpoint (required)
- `num_steps`: Number of evaluation steps (default: 500)
- `init_cond`: Initial condition index in dataset (default: 500)
- `with_obs`: Enable obstacle handling for Cylinder cases (default: true)
- `device`: GPU index or -1 for CPU
- `compile`: Enable PyTorch compilation
- `output_dir`: Base directory for output images (default: images)

**Note:** When using TVD (as in SOD case 2), the latest saved model file may yield more accurate results than the one with the lowest validation loss. We recommend experimenting with different saved files.

## **Important Notes:** 
* In the paper, we use `torch.roll` for streaming. In this code, we removed the use of this function by defining the indices of the streaming directly. otherwise we have to make a for loop in the number of discrete velocities.
* We also use a sparse solver for the matrix inversion required in Newton's method for all the cases: for the cylinder, the BC and Obstacle uses a numpy inversion while the faster cylinder use a sparse solver for all the cases in the cylinder test. 

## Contact and Support
* The original code used for the paper can be provided upon request. Please open a GitHub issue or contact [antonio.lara@rice.edu] to request the original code---it is slower.*


## Contributing
Contributions are welcome! 
