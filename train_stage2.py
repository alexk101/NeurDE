"""
Stage 2 training entry point.

Uses Hydra for configuration management.
Example usage:
    python train_stage2.py case=cylinder stage=2
    python train_stage2.py case=cylinder_faster stage=2
    python train_stage2.py case=sod_shock_tube case_number=2 training=stage2 training.tvd.enabled=true
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np

from model.model import NeurDE
from training import Stage2Trainer, create_basis
from utils.datasets import RolloutBatchDataset, Stage2Dataset
from utils.data_io import load_data_stage_2
from utils.optimizer import dispatch_optimizer, get_scheduler
from utils.solver import create_solver
from utils.core import set_seed, adapt_checkpoint_keys
from torch.utils.data import DataLoader


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training function for Stage 2.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object
    """
    # Set seed for reproducibility
    set_seed(0)

    # Override stage to 2
    cfg.stage = 2

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

    # Create solver
    solver_kwargs = {
        "X": physics.X,
        "Y": physics.Y,
        "Qn": physics.Qn,
        "device": device,
    }

    if case_cfg.case_type in ["cylinder", "cylinder_faster"]:
        solver_kwargs.update(
            {
                "radius": physics.radius,
                "Ma0": physics.Ma0,
                "Re": physics.Re,
                "rho0": physics.rho0,
                "T0": physics.T0,
                "alpha1": physics.alpha1,
                "alpha01": physics.alpha01,
                "vuy": physics.vuy,
                "Pr": physics.Pr,
                "Ns": physics.Ns,
            }
        )
    elif case_cfg.case_type == "sod_shock_tube":
        solver_kwargs.update(
            {
                "alpha1": physics.alpha1,
                "alpha01": physics.alpha01,
                "vuy": physics.vuy,
                "Pr": physics.Pr,
                "muy": physics.muy,
                "Uax": Uax,
                "Uay": Uay,
            }
        )

    solver = create_solver(case_cfg.solver_type, **solver_kwargs)

    # Compile solver if requested
    if cfg.compile:
        solver.collision = torch.compile(
            solver.collision, dynamic=True, fullgraph=False
        )
        solver.streaming = torch.compile(
            solver.streaming, dynamic=True, fullgraph=False
        )
        solver.shift_operator = torch.compile(
            solver.shift_operator, dynamic=True, fullgraph=False
        )
        solver.get_macroscopic = torch.compile(
            solver.get_macroscopic, dynamic=True, fullgraph=False
        )
        solver.get_Feq = torch.compile(solver.get_Feq, dynamic=True, fullgraph=False)
        solver.get_temp_from_energy = torch.compile(
            solver.get_temp_from_energy, dynamic=True, fullgraph=False
        )
        print("Solver compiled.")

    # Load data
    data_path = case_cfg.data.stage2
    all_F, all_G, all_Feq, all_Geq = load_data_stage_2(data_path)

    # Create training dataset
    train_dataset = RolloutBatchDataset(
        all_Fi=all_F[: cfg.num_samples],
        all_Gi=all_G[: cfg.num_samples],
        all_Feq=all_Feq[: cfg.num_samples],
        all_Geq=all_Geq[: cfg.num_samples],
        number_of_rollout=cfg.training.N,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=cfg.dataset.shuffle,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
    )

    # Create validation dataset if enabled
    val_dataset = None
    if cfg.training.validation.enabled:
        val_dataset = Stage2Dataset(
            F=all_F[
                cfg.num_samples : cfg.num_samples
                + cfg.training.validation.dataset_size
            ],
            G=all_G[
                cfg.num_samples : cfg.num_samples
                + cfg.training.validation.dataset_size
            ],
            Feq=all_Feq[
                cfg.num_samples : cfg.num_samples
                + cfg.training.validation.dataset_size
            ],
            Geq=all_Geq[
                cfg.num_samples : cfg.num_samples
                + cfg.training.validation.dataset_size
            ],
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

    # Load pretrained model if path provided
    if cfg.get("pretrained_path") and cfg.pretrained_path:
        checkpoint_path = cfg.pretrained_path
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Adapt checkpoint keys to match model format (handles torch.compile prefix)
        state_dict = adapt_checkpoint_keys(checkpoint, model)
        model.load_state_dict(state_dict)

        print(f"Pretrained model loaded from {checkpoint_path}")

    # Create optimizer
    optimizer = dispatch_optimizer(
        model=model,
        lr=cfg.training.lr,
        optimizer_type=cfg.optimizer.optimizer_type,
    )

    # Create scheduler
    total_steps = len(train_dataloader) * cfg.training.epochs
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

    # TVD configuration
    tvd_enabled = cfg.training.tvd.enabled
    tvd_weight = cfg.training.tvd.weight
    tvd_milestones = cfg.training.tvd.weight_scheduler.milestones or []
    tvd_weights = cfg.training.tvd.weight_scheduler.weights or [tvd_weight]

    # Detach after streaming (SOD case)
    detach_after_streaming = cfg.training.detach_after_streaming
    if case_cfg.case_type == "sod_shock_tube":
        detach_after_streaming = True  # SOD always detaches

    # Create trainer
    trainer = Stage2Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=train_dataloader,
        solver=solver,
        device=device,
        model_dir=cfg.training.model_dir,
        basis=basis,
        num_rollout=cfg.training.N,
        save_model=cfg.save_model,
        save_frequency=cfg.save_frequency,
        checkpoint_frequency=cfg.get("checkpoint_frequency", 0),
        keep_checkpoints=cfg.get("keep_checkpoints", 5),
        resume_from=cfg.get("resume_from"),
        case_name=case_name,
        tvd_enabled=tvd_enabled,
        tvd_weight=tvd_weight,
        tvd_milestones=tvd_milestones,
        tvd_weights=tvd_weights,
        validation_enabled=cfg.training.validation.enabled,
        val_dataset=val_dataset,
        detach_after_streaming=detach_after_streaming,
        ema_alpha=cfg.training.ema_alpha,
    )

    # Print configuration
    print(f"Training {case_cfg.case_type} on {device}")
    print(f"Epochs: {cfg.training.epochs}, Samples: {cfg.num_samples}")
    print(f"Rollout steps: {cfg.training.N}")
    print(f"TVD enabled: {tvd_enabled}")
    print(f"Validation enabled: {cfg.training.validation.enabled}")
    print(f"Detach after streaming: {detach_after_streaming}")

    # Train
    trainer.train(cfg.training.epochs)

    print("Training complete.")


if __name__ == "__main__":
    main()
