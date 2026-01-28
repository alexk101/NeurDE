"""
Evaluation entry point.

Runs inference/rollout with a trained model and generates visualization plots.
Uses Hydra for configuration management.

Example usage:
    python eval.py case=cylinder trained_path=results/stage2/best_model_epoch_100_top_1_loss_0.001234.pt
    python eval.py case=sod_shock_tube case_number=1 trained_path=results/case1/stage2/best_model_1_epoch_100_top_1_val_loss_0.001234.pt
"""

import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import h5py
import os
from tqdm import tqdm

from model.model import NeurDE
from training import create_basis
from utils.solver import create_solver
from utils.loss import l2_error
from utils.core import set_seed, detach, adapt_checkpoint_keys
from utils.plotting import plot_cylinder_results, plot_sod_results


@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    """
    Main evaluation function.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object
    """
    # Set seed for reproducibility
    set_seed(0)

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
        if hasattr(solver, "get_temp_from_energy"):
            solver.get_temp_from_energy = torch.compile(
                solver.get_temp_from_energy, dynamic=True, fullgraph=False
            )
        print("Solver compiled.")

    # Load model architecture config (from training config if available, else from model config)
    # For eval, we need to infer from checkpoint or use defaults
    hidden_dim = cfg.model.hidden_dim if hasattr(cfg, "model") else 32
    num_layers = cfg.model.num_layers if hasattr(cfg, "model") else 4
    activation = cfg.model.activation if hasattr(cfg, "model") else "relu"

    # Create model
    model = NeurDE(
        alpha_layer=[4] + [hidden_dim] * num_layers,
        phi_layer=[2] + [hidden_dim] * num_layers,
        activation=activation,
    ).to(device)

    # Compile model if requested
    if cfg.compile:
        model = torch.compile(model)
        print("Model compiled.")

    # Load trained model checkpoint
    if cfg.trained_path:
        checkpoint = torch.load(cfg.trained_path, map_location=device)

        # Adapt checkpoint keys to match model format (handles torch.compile prefix)
        state_dict = adapt_checkpoint_keys(checkpoint, model)
        model.load_state_dict(state_dict)

        print(f"Trained model loaded from {cfg.trained_path}")
    else:
        raise ValueError("trained_path must be provided in config or via command line")

    # Load data for ground truth comparison
    data_path = case_cfg.data.stage2
    with h5py.File(data_path, "r") as f:
        all_rho = f["rho"][:]
        all_ux = f["ux"][:]
        all_uy = f["uy"][:]
        all_T = f["T"][:]
        all_Fi0 = f["Fi0"][:]
        all_Gi0 = f["Gi0"][:]

    # Initialize from dataset
    Fi0 = torch.tensor(all_Fi0[cfg.init_cond], device=device)
    Gi0 = torch.tensor(all_Gi0[cfg.init_cond], device=device)

    # Prepare output directory
    if case_cfg.case_type == "sod_shock_tube":
        case_name = f"SOD_case{case_cfg.case_number}"
    else:
        case_name = case_cfg.case_type.capitalize()

    image_dir = os.path.join(cfg.output_dir, case_name, "test_NN")
    os.makedirs(image_dir, exist_ok=True)

    print(f"Evaluating {case_cfg.case_type} on {device}")
    print(f"Steps: {cfg.num_steps}, Initial condition: {cfg.init_cond}")
    if case_cfg.case_type in ["cylinder", "cylinder_faster"]:
        print(f"With obstacle: {cfg.with_obs}")

    # Evaluation loop
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for i in tqdm(range(cfg.num_steps), desc="Evaluation"):
            # Get macroscopic variables
            rho, ux, uy, E = solver.get_macroscopic(Fi0.squeeze(0), Gi0.squeeze(0))
            T = solver.get_temp_from_energy(ux, uy, E)

            # Get equilibrium F
            Feq = solver.get_Feq(rho, ux, uy, T)

            # Prepare model input
            inputs = torch.stack(
                [
                    rho.unsqueeze(0),
                    ux.unsqueeze(0),
                    uy.unsqueeze(0),
                    T.unsqueeze(0),
                ],
                dim=1,
            ).to(device)

            # Model prediction
            Geq_pred = model(inputs, basis)

            # Compute loss against ground truth
            Geq_target = torch.tensor(all_Gi0[cfg.init_cond], device=device).unsqueeze(
                0
            )

            inner_loss = l2_error(
                Geq_pred, Geq_target.permute(0, 2, 3, 1).reshape(-1, 9)
            )
            total_loss += inner_loss.item()

            # Collision
            Geq_pred_reshaped = Geq_pred.permute(1, 0).reshape(
                solver.Qn, solver.Y, solver.X
            )
            Fi0, Gi0 = solver.collision(
                Fi0.squeeze(0),
                Gi0.squeeze(0),
                Feq,
                Geq_pred_reshaped,
                rho,
                ux,
                uy,
                T,
            )

            # Streaming
            Fi, Gi = solver.streaming(Fi0, Gi0)

            # Handle obstacle and BC (Cylinder cases)
            if case_cfg.case_type in ["cylinder", "cylinder_faster"] and cfg.with_obs:
                khi = detach(torch.zeros_like(ux))
                zetax = detach(torch.zeros_like(ux))
                zetay = detach(torch.zeros_like(ux))

                Fi_obs_cyl, Gi_obs_cyl, Fi_obs_Inlet, Gi_obs_Inlet = (
                    solver.get_obs_distribution(rho, ux, uy, T, khi, zetax, zetay)
                )

                Fi_new, Gi_new = solver.enforce_Obs_and_BC(
                    Fi, Gi, Fi_obs_cyl, Gi_obs_cyl, Fi_obs_Inlet, Gi_obs_Inlet
                )

                Fi0 = Fi_new.detach().unsqueeze(0)
                Gi0 = Gi_new.detach().unsqueeze(0)
            else:
                Fi0 = Fi.detach().unsqueeze(0)
                Gi0 = Gi.detach().unsqueeze(0)

            # Plotting
            step_idx = i + cfg.init_cond

            if case_cfg.case_type in ["cylinder", "cylinder_faster"]:
                # Cylinder: Plot Mach number
                Ma_NN = solver.get_local_Mach(ux, uy, T)
                Ma_GT = np.sqrt(all_ux**2 + all_uy**2) / np.sqrt(all_T * physics.vuy)
                plot_cylinder_results(
                    Ma_NN.cpu().numpy(),
                    Ma_GT[cfg.init_cond + i],
                    step_idx,
                    case_cfg.case_type,
                    output_dir=image_dir,
                )
            elif case_cfg.case_type == "sod_shock_tube":
                # SOD: Plot density, temperature, velocity, pressure
                rho_np = detach(rho)
                ux_np = detach(ux)
                T_np = detach(T)
                P_np = rho_np * T_np

                plot_sod_results(
                    rho_np,
                    ux_np,
                    T_np,
                    P_np,
                    all_rho[cfg.init_cond + i],
                    all_ux[cfg.init_cond + i],
                    all_T[cfg.init_cond + i],
                    all_rho[cfg.init_cond + i] * all_T[cfg.init_cond + i],
                    step_idx,
                    case_cfg.case_number,
                    output_dir=image_dir,
                )

    avg_loss = total_loss / cfg.num_steps
    print(f"\nEvaluation complete. Average loss: {avg_loss:.6f}")
    print(f"Images saved to: {image_dir}")


if __name__ == "__main__":
    main()
