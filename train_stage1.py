"""
Stage 1 training entry point.

Uses Hydra for configuration management.
Example usage:
    python train_stage1.py case=cylinder
    python train_stage1.py case=cylinder_faster
    python train_stage1.py case=sod_shock_tube case_number=1
"""

import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np

from hydra.core.hydra_config import HydraConfig

from model.model import NeurDE
from training import Stage1Trainer, create_basis
from utils.datasets import EquilibriumDataset
from utils.data_io import load_equilibrium_state
from utils.optimizer import dispatch_optimizer, get_scheduler
from utils.core import set_seed, get_device
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
    device = get_device()

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

    val_size = cfg.training.get("validation", {}).get("dataset_size", 0)
    train_end = cfg.num_samples
    val_end = train_end + val_size
    if val_end > len(all_rho):
        val_size = max(0, len(all_rho) - train_end)
        val_end = train_end + val_size

    # Training dataset and dataloader
    train_dataset = EquilibriumDataset(
        all_rho[:train_end],
        all_ux[:train_end],
        all_uy[:train_end],
        all_T[:train_end],
        all_Geq[:train_end],
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.dataset.shuffle,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
    )

    # Validation dataset and dataloader (optional)
    val_dataloader = None
    if val_size > 0:
        val_dataset = EquilibriumDataset(
            all_rho[train_end:val_end],
            all_ux[train_end:val_end],
            all_uy[train_end:val_end],
            all_T[train_end:val_end],
            all_Geq[train_end:val_end],
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
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
    sched_cfg = cfg.training.scheduler
    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_type=sched_cfg.scheduler_type,
        total_steps=total_steps,
        config=OmegaConf.to_container(sched_cfg, resolve=True),
        total_epochs=cfg.training.epochs,
    )

    # Resolve model_dir under Hydra output dir so checkpoints live in outputs/<run>/
    try:
        hydra_out = HydraConfig.get().runtime.output_dir
        model_dir = os.path.join(hydra_out, cfg.training.model_dir)
    except Exception:
        model_dir = cfg.training.model_dir

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
        model_dir=model_dir,
        basis=basis,
        save_model=cfg.save_model,
        save_frequency=cfg.save_frequency,
        checkpoint_frequency=cfg.get("checkpoint_frequency", 0),
        keep_checkpoints=cfg.get("keep_checkpoints", 5),
        resume_from=cfg.get("resume_from"),
        case_name=case_name,
        val_dataloader=val_dataloader,
        case_type=case_cfg.case_type,
        cfg=cfg,
    )

    # Print configuration
    print(f"Training {case_cfg.case_type} on {device}")
    print(
        f"Epochs: {cfg.training.epochs}, Train samples: {train_end}, Val samples: {val_size}"
    )
    print(f"Model: {cfg.model.hidden_dim} hidden dim, {cfg.model.num_layers} layers")

    # Train
    trainer.train(cfg.training.epochs)

    print("Training complete.")


if __name__ == "__main__":
    main()
