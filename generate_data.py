"""
Data generation script for NeurDE training data.

This script runs simulations using the unified solvers and saves the data
to HDF5 files for training. Uses Hydra for configuration management.

Example usage:
    python generate_data.py case=cylinder steps=1000
    python generate_data.py case=cylinder_faster steps=1000
    python generate_data.py case=sod_shock_tube case_number=1 steps=1000
"""

import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import h5py
import os

from utils.solver import create_solver
from utils.core import detach


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Generate training data by running simulations.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object
    """
    # Setup device
    device = torch.device(
        "cuda" if cfg.device >= 0 and torch.cuda.is_available() else "cpu"
    )

    # Load case configuration
    case_cfg = cfg.case
    physics = case_cfg.physics

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
                # Note: sparse_format and use_dense_inv are set by create_solver
                # based on solver_type, so don't pass them here
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
                "Uax": case_cfg.velocity_shift.Uax,
                "Uay": case_cfg.velocity_shift.Uay,
            }
        )

    # Use solver_type from config (cylinder_base, cylinder_faster, or sod_solver)
    solver_type = case_cfg.solver_type
    solver = create_solver(solver_type, **solver_kwargs)

    # Compile solver methods if requested
    if cfg.compile:
        solver.collision = torch.compile(
            solver.collision, dynamic=True, fullgraph=False
        )
        solver.streaming = torch.compile(
            solver.streaming, dynamic=True, fullgraph=False
        )
        solver.get_macroscopic = torch.compile(
            solver.get_macroscopic, dynamic=True, fullgraph=False
        )
        solver.get_Feq = torch.compile(solver.get_Feq, dynamic=True, fullgraph=False)
        print("Solver methods compiled.")

    # Initialize distributions
    if case_cfg.case_type in ["cylinder", "cylinder_faster"]:
        Fi0, Gi0, khi0, zetax0, zetay0 = solver.initial_conditions()
    elif case_cfg.case_type == "sod_shock_tube":
        if case_cfg.case_number == 1:
            Fi0, Gi0, khi0, zetax0, zetay0 = solver.case_1_initial_conditions()
        elif case_cfg.case_number == 2:
            Fi0, Gi0, khi0, zetax0, zetay0 = solver.case_2_initial_conditions()
        else:
            raise ValueError(f"Invalid case_number: {case_cfg.case_number}")

    # Run simulation with batched GPU memory utilization
    steps = cfg.get("steps", 1000)
    batch_size = cfg.get("transfer_batch_size", 100)  # Transfer to CPU every N steps
    
    print(f"Running {steps} simulation steps for {case_cfg.case_type}...")
    print(f"Using GPU batch size: {batch_size} steps per CPU transfer")

    # Pre-allocate GPU tensors for batching (reduces memory fragmentation)
    Y, X = physics.Y, physics.X
    Qn = physics.Qn
    
    # Storage for data (will accumulate on GPU, then transfer in batches)
    all_rho = []
    all_ux = []
    all_uy = []
    all_T = []
    all_Feq = []
    all_Geq = []
    all_Fi0 = []
    all_Gi0 = []

    # Additional storage for Cylinder cases
    if case_cfg.case_type in ["cylinder", "cylinder_faster"]:
        all_Fi_obs_cyl = []
        all_Gi_obs_cyl = []
        all_Fi_obs_Inlet = []
        all_Gi_obs_Inlet = []

    # GPU-side batch buffers (accumulate before CPU transfer)
    # Note: Obstacle distributions are not full grid, so we don't batch them
    if device.type == "cuda":
        batch_rho = torch.zeros((batch_size, Y, X), device=device, dtype=torch.float32)
        batch_ux = torch.zeros((batch_size, Y, X), device=device, dtype=torch.float32)
        batch_uy = torch.zeros((batch_size, Y, X), device=device, dtype=torch.float32)
        batch_T = torch.zeros((batch_size, Y, X), device=device, dtype=torch.float32)
        batch_Feq = torch.zeros((batch_size, Qn, Y, X), device=device, dtype=torch.float32)
        batch_Geq = torch.zeros((batch_size, Qn, Y, X), device=device, dtype=torch.float32)
        batch_Fi0 = torch.zeros((batch_size, Qn, Y, X), device=device, dtype=torch.float32)
        batch_Gi0 = torch.zeros((batch_size, Qn, Y, X), device=device, dtype=torch.float32)
    else:
        # CPU mode: no batching needed
        batch_rho = batch_ux = batch_uy = batch_T = None
        batch_Feq = batch_Geq = batch_Fi0 = batch_Gi0 = None

    with torch.no_grad():
        batch_idx = 0
        for i in range(steps):
            # Store initial distributions in GPU batch buffer
            if device.type == "cuda":
                batch_Fi0[batch_idx] = Fi0
                batch_Gi0[batch_idx] = Gi0
            else:
                all_Fi0.append(detach(Fi0))
                all_Gi0.append(detach(Gi0))

            # Get macroscopic variables
            rho, ux, uy, E = solver.get_macroscopic(Fi0, Gi0)
            T = solver.get_temp_from_energy(ux, uy, E)

            # Store macroscopic variables in GPU batch buffer
            if device.type == "cuda":
                batch_rho[batch_idx] = rho
                batch_ux[batch_idx] = ux
                batch_uy[batch_idx] = uy
                batch_T[batch_idx] = T
            else:
                all_rho.append(detach(rho))
                all_ux.append(detach(ux))
                all_uy.append(detach(uy))
                all_T.append(detach(T))

            # Get equilibrium distributions
            Feq = solver.get_Feq(rho, ux, uy, T)
            if case_cfg.case_type in ["cylinder", "cylinder_faster"]:
                # Use solver's sparse_format attribute (set by create_solver)
                Geq, khi, zetax, zetay = solver.get_Geq_Newton_solver(
                    rho, ux, uy, T, khi0, zetax0, zetay0,
                    sparse_format=solver.sparse_format
                )
            else:
                Geq, khi, zetax, zetay = solver.get_Geq_Newton_solver(
                    rho, ux, uy, T, khi0, zetax0, zetay0
                )

            if device.type == "cuda":
                batch_Feq[batch_idx] = Feq
                batch_Geq[batch_idx] = Geq
            else:
                all_Feq.append(detach(Feq))
                all_Geq.append(detach(Geq))

            # Collision
            Fi0, Gi0 = solver.collision(Fi0, Gi0, Feq, Geq, rho, ux, uy, T)

            # Streaming
            Fi, Gi = solver.streaming(Fi0, Gi0)

            # Handle obstacle and BC for Cylinder cases
            # Note: Obstacle distributions are not full grid (only obstacle points),
            # so we store them directly without batching
            if case_cfg.case_type in ["cylinder", "cylinder_faster"]:
                Fi_obs_cyl, Gi_obs_cyl, Fi_obs_Inlet, Gi_obs_Inlet = (
                    solver.get_obs_distribution(rho, ux, uy, T, khi, zetax, zetay)
                )

                # Store directly (obstacle distributions are smaller, not full grid)
                all_Fi_obs_cyl.append(detach(Fi_obs_cyl))
                all_Gi_obs_cyl.append(detach(Gi_obs_cyl))
                all_Fi_obs_Inlet.append(detach(Fi_obs_Inlet))
                all_Gi_obs_Inlet.append(detach(Gi_obs_Inlet))

                Fi_new, Gi_new = solver.enforce_Obs_and_BC(
                    Fi, Gi, Fi_obs_cyl, Gi_obs_cyl, Fi_obs_Inlet, Gi_obs_Inlet
                )

                Fi0 = Fi_new
                Gi0 = Gi_new
            else:
                # SOD handles BC inline during streaming
                Fi0 = Fi
                Gi0 = Gi

            # Update relaxation variables
            khi0 = khi
            zetax0 = zetax
            zetay0 = zetay

            # Transfer batch to CPU when full
            batch_idx += 1
            if device.type == "cuda" and batch_idx >= batch_size:
                # Transfer entire batch to CPU at once (more efficient than per-step)
                all_rho.extend(batch_rho.cpu().numpy())
                all_ux.extend(batch_ux.cpu().numpy())
                all_uy.extend(batch_uy.cpu().numpy())
                all_T.extend(batch_T.cpu().numpy())
                all_Feq.extend(batch_Feq.cpu().numpy())
                all_Geq.extend(batch_Geq.cpu().numpy())
                all_Fi0.extend(batch_Fi0.cpu().numpy())
                all_Gi0.extend(batch_Gi0.cpu().numpy())
                
                batch_idx = 0
                torch.cuda.empty_cache()  # Free unused GPU memory

        # Transfer remaining batch if any
        if device.type == "cuda" and batch_idx > 0:
            all_rho.extend(batch_rho[:batch_idx].cpu().numpy())
            all_ux.extend(batch_ux[:batch_idx].cpu().numpy())
            all_uy.extend(batch_uy[:batch_idx].cpu().numpy())
            all_T.extend(batch_T[:batch_idx].cpu().numpy())
            all_Feq.extend(batch_Feq[:batch_idx].cpu().numpy())
            all_Geq.extend(batch_Geq[:batch_idx].cpu().numpy())
            all_Fi0.extend(batch_Fi0[:batch_idx].cpu().numpy())
            all_Gi0.extend(batch_Gi0[:batch_idx].cpu().numpy())

    # Save data to HDF5
    os.makedirs("data_base", exist_ok=True)

    if case_cfg.case_type == "sod_shock_tube":
        filename = f"data_base/SOD_case{case_cfg.case_number}.h5"
    else:
        filename = "data_base/cylinder_case.h5"

    print(f"Saving data to {filename}...")

    with h5py.File(filename, "w") as f:
        f.create_dataset("rho", data=np.array(all_rho))
        f.create_dataset("ux", data=np.array(all_ux))
        f.create_dataset("uy", data=np.array(all_uy))
        f.create_dataset("T", data=np.array(all_T))
        f.create_dataset("Feq", data=np.array(all_Feq))
        f.create_dataset("Geq", data=np.array(all_Geq))
        f.create_dataset("Fi0", data=np.array(all_Fi0))
        f.create_dataset("Gi0", data=np.array(all_Gi0))

        if case_cfg.case_type in ["cylinder", "cylinder_faster"]:
            f.create_dataset("Fi_obs_cyl", data=np.array(all_Fi_obs_cyl))
            f.create_dataset("Gi_obs_cyl", data=np.array(all_Gi_obs_cyl))
            f.create_dataset("Fi_obs_Inlet", data=np.array(all_Fi_obs_Inlet))
            f.create_dataset("Gi_obs_Inlet", data=np.array(all_Gi_obs_Inlet))

    print(f"Data generation complete! Saved {steps} steps to {filename}")


if __name__ == "__main__":
    main()
