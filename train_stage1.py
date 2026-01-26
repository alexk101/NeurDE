"""
Stage 1 training entry point.

Uses Hydra for configuration management.
Example usage:
    python train_stage1.py case=cylinder
    python train_stage1.py case=cylinder_faster
    python train_stage1.py case=sod_shock_tube case_number=1
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np

from model.model import NeurDE
from training import Stage1Trainer, create_basis
from utils.datasets import EquilibriumDataset
from utils.data_io import load_equilibrium_state
from utils.optimizer import dispatch_optimizer, get_scheduler
from utils.core import set_seed
from torch.utils.data import DataLoader


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training function for Stage 1.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object
    """
    # Set seed for reproducibility
    set_seed(0)

    # Override stage to 1
    cfg.stage = 1

    # Setup device
    device = torch.device(
        "cuda" if cfg.device >= 0 and torch.cuda.is_available() else "cpu"
    )

    # Load case configuration
    case_cfg = cfg.case
    physics = case_cfg.physics

    # Calculate velocity shift if needed
    if case_cfg.velocity_shift.calculate:
        # Cylinder case: calculate from physics params
        cs0 = np.sqrt(physics.vuy * physics.T0)
        U0 = physics.Ma0 * cs0
        Uax = U0 * physics.Ns
        Uay = 0.0
    else:
        # SOD case: use direct values
        Uax = case_cfg.velocity_shift.Uax
        Uay = case_cfg.velocity_shift.Uay

    # Create basis
    basis = create_basis(Uax, Uay, device)

    # Load data
    data_path = case_cfg.data.stage1
    all_rho, all_ux, all_uy, all_T, all_Geq = load_equilibrium_state(data_path)

    # Create dataset
    dataset = EquilibriumDataset(
        all_rho[: cfg.num_samples],
        all_ux[: cfg.num_samples],
        all_uy[: cfg.num_samples],
        all_T[: cfg.num_samples],
        all_Geq[: cfg.num_samples],
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.dataset.shuffle,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
    )

    # Create model
    model = NeurDE(
        alpha_layer=[4] + [cfg.model.hidden_dim] * cfg.model.num_layers,
        phi_layer=[2] + [cfg.model.hidden_dim] * cfg.model.num_layers,
        activation=cfg.model.activation,
    ).to(device)

    # Compile model if requested
    if cfg.compile:
        model = torch.compile(model)
        print("Model compiled.")

    # Create optimizer
    optimizer = dispatch_optimizer(
        model=model,
        lr=cfg.training.lr,
        optimizer_type=cfg.optimizer.optimizer_type,
    )

    # Create scheduler
    total_steps = len(dataloader) * cfg.training.epochs
    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_type=cfg.scheduler.scheduler_type,
        total_steps=total_steps,
        config=OmegaConf.to_container(cfg.scheduler, resolve=True),
    )

    # Get case name for model saving
    case_name = None
    if case_cfg.case_type == "sod_shock_tube":
        case_name = str(case_cfg.case_number)

    # Create trainer
    trainer = Stage1Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=dataloader,
        device=device,
        model_dir=cfg.training.model_dir,
        basis=basis,
        save_model=cfg.save_model,
        save_frequency=cfg.save_frequency,
        case_name=case_name,
    )

    # Print configuration
    print(f"Training {case_cfg.case_type} on {device}")
    print(f"Epochs: {cfg.training.epochs}, Samples: {cfg.num_samples}")
    print(f"Model: {cfg.model.hidden_dim} hidden dim, {cfg.model.num_layers} layers")

    # Train
    trainer.train(cfg.training.epochs)

    print("Training complete.")


if __name__ == "__main__":
    main()
