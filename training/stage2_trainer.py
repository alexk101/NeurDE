"""
Stage 2 training class.

This module provides the Stage2Trainer class for training the model on
rollout sequences with solver integration.
"""

import math
import os
from omegaconf import DictConfig
import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, Optional
from torch.utils.data import DataLoader
from logging import getLogger
import matplotlib.pyplot as plt

from .base import BaseTrainer
from utils.datasets import Stage2Dataset
from utils.loss import l2_error, TVD_norm
from utils.case_specific import tvd_weight_scheduler
from utils.core import detach
from utils.plotting import plot_cylinder_results, plot_sod_results, fig_to_image


class Stage2Trainer(BaseTrainer):
    """
    Trainer for Stage 2 (rollout training with solver integration).

    Trains the model on rollout sequences, integrating with the solver
    for collision and streaming operations.

    Usage by case:
    -------------
    Cylinder/Cylinder_faster:
        - TVD: False (default)
        - Detach after streaming: False (default)

    SOD_shock_tube:
        - TVD: Optional (configurable, used in case 2)
        - Detach after streaming: True (default for SOD)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        dataloader: DataLoader,
        solver: Any,  # Cylinder_base or SODSolver
        device: torch.device,
        model_dir: str,
        basis: torch.Tensor,
        num_rollout: int,
        val_dataset: Stage2Dataset,
        save_model: bool = True,
        save_frequency: int = 1,
        checkpoint_frequency: int = 0,
        keep_checkpoints: int = 5,
        resume_from: Optional[str] = None,
        case_name: Optional[str] = None,
        # Optional features
        tvd_enabled: bool = False,
        tvd_weight: float = 15.0,
        tvd_milestones: Optional[list] = None,
        tvd_weights: Optional[list] = None,
        detach_after_streaming: bool = False,
        cfg: DictConfig = None,
    ):
        """
        Initialize Stage 2 trainer.

        Parameters
        ----------
        model : nn.Module
            Neural network model
        optimizer : torch.optim.Optimizer
            Optimizer for training
        scheduler : Optional[torch.optim.lr_scheduler._LRScheduler]
            Learning rate scheduler
        dataloader : DataLoader
            DataLoader for training data (rollout sequences)
        solver : Any
            Solver instance (Cylinder_base or SODSolver)
        device : torch.device
            Device for computation
        model_dir : str
            Directory to save models
        basis : torch.Tensor
            Basis vectors for lattice Boltzmann method
        num_rollout : int
            Number of rollout steps per sequence
        val_dataset : Stage2Dataset
            Validation dataset (required)
        save_model : bool, optional
            Whether to save model checkpoints (default: True)
        save_frequency : int, optional
            Minimum epochs between saves (default: 1)
        checkpoint_frequency : int, optional
            Save full checkpoint every N epochs (0 = disabled, default: 0)
        keep_checkpoints : int, optional
            Number of periodic checkpoints to keep (default: 5)
        resume_from : Optional[str], optional
            Path to checkpoint to resume from (default: None)
        case_name : Optional[str], optional
            Case name for model save naming
        tvd_enabled : bool, optional
            Whether to use TVD loss (default: False)
        tvd_weight : float, optional
            Base weight for TVD loss (default: 15.0)
        tvd_milestones : Optional[list], optional
            Epoch milestones for TVD weight scheduling
        tvd_weights : Optional[list], optional
            TVD weights corresponding to milestones
        detach_after_streaming : bool, optional
            Whether to detach gradients after streaming (default: False)
        cfg : DictConfig
            Hydra configuration object
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            model_dir=model_dir,
            save_model=save_model,
            save_frequency=save_frequency,
            checkpoint_frequency=checkpoint_frequency,
            keep_checkpoints=keep_checkpoints,
            resume_from=resume_from,
            cfg=cfg,
        )
        self.dataloader = dataloader
        self.solver = solver
        self.basis = basis
        self.num_rollout = num_rollout
        self.case_name = case_name
        self.log = getLogger(__name__)

        # TVD settings
        self.tvd_enabled = tvd_enabled
        self.tvd_weight = tvd_weight
        self.tvd_milestones = tvd_milestones or []
        self.tvd_weights = tvd_weights or [tvd_weight]

        # Validation dataset (always required)
        self.val_dataset = val_dataset

        # Detaching behavior
        self.detach_after_streaming = detach_after_streaming

        # Last epoch train loss (for experiment tracker: log both train and val)
        self._last_epoch_train_loss: Optional[float] = None

        # Validation rollout cache (data, epoch) for field errors + image logging
        self._validation_rollout_cache: Optional[tuple] = None
        self._last_epoch_field_errors: Dict[str, float] = {}

    def _handle_obstacle_and_bc(self, Fi, Gi, rho, ux, uy, T, khi, zetax, zetay):
        """
        Handle obstacle and boundary conditions (Cylinder cases only).

        Parameters
        ----------
        Fi : torch.Tensor
            F distribution after streaming
        Gi : torch.Tensor
            G distribution after streaming
        rho : torch.Tensor
            Density
        ux : torch.Tensor
            x-velocity
        uy : torch.Tensor
            y-velocity
        T : torch.Tensor
            Temperature
        khi : numpy.ndarray
            Lagrange multiplier for density
        zetax : numpy.ndarray
            Lagrange multiplier for x-velocity
        zetay : numpy.ndarray
            Lagrange multiplier for y-velocity

        Returns
        -------
        tuple
            (Fi_new, Gi_new) with obstacle and BC applied
        """
        # Check if solver has obstacle/BC methods (Cylinder cases)
        if hasattr(self.solver, "get_obs_distribution") and hasattr(
            self.solver, "enforce_Obs_and_BC"
        ):
            with torch.no_grad():
                khi_detached = detach(torch.zeros_like(ux))
                zetax_detached = detach(torch.zeros_like(ux))
                zetay_detached = detach(torch.zeros_like(ux))

                Fi_obs_cyl, Gi_obs_cyl, Fi_obs_Inlet, Gi_obs_Inlet = (
                    self.solver.get_obs_distribution(
                        rho, ux, uy, T, khi_detached, zetax_detached, zetay_detached
                    )
                )

                Fi_new, Gi_new = self.solver.enforce_Obs_and_BC(
                    Fi, Gi, Fi_obs_cyl, Gi_obs_cyl, Fi_obs_Inlet, Gi_obs_Inlet
                )
            return Fi_new, Gi_new
        else:
            # SOD case - no obstacle/BC handling
            return Fi, Gi

    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number

        Returns
        -------
        float
            Average loss for the epoch
        """
        self.model.train()
        loss_epoch = 0.0
        num_batches = 0

        # Get TVD weight for this epoch
        if self.tvd_enabled:
            current_tvd_weight = tvd_weight_scheduler(
                epoch, self.tvd_milestones, self.tvd_weights
            )
        else:
            current_tvd_weight = 0.0

        for batch_idx, (F_seq, G_seq, Feq_seq, Geq_seq) in enumerate(self.dataloader):
            self.optimizer.zero_grad()
            total_loss = 0.0

            F_seq = F_seq.to(self.device)
            G_seq = G_seq.to(self.device)
            Fi0 = F_seq[0, 0, ...]
            Gi0 = G_seq[0, 0, ...]

            # Initialize TVD tracking variables for this batch
            if self.tvd_enabled:
                ux_old = None
                T_old = None
                rho_old = None

            for rollout in range(self.num_rollout):
                # Get macroscopic variables
                rho, ux, uy, E = self.solver.get_macroscopic(Fi0, Gi0)
                T = self.solver.get_temp_from_energy(ux, uy, E)

                # Get equilibrium F
                Feq = self.solver.get_Feq(rho, ux, uy, T)

                # Prepare model input
                inputs = torch.stack(
                    [
                        rho.unsqueeze(0),
                        ux.unsqueeze(0),
                        uy.unsqueeze(0),
                        T.unsqueeze(0),
                    ],
                    dim=1,
                ).to(self.device)

                # Model prediction
                Geq_pred = self.model(inputs, self.basis)

                Geq_target = Geq_seq[0, rollout].to(self.device)

                # Compute loss
                inner_loss = l2_error(
                    Geq_pred, Geq_target.permute(1, 2, 0).reshape(-1, 9)
                )
                total_loss += inner_loss

                # TVD loss (if enabled and not first rollout)
                if self.tvd_enabled and rollout > 0:
                    loss_TVD = (
                        TVD_norm(T, T_old)
                        + TVD_norm(ux, ux_old)
                        + TVD_norm(rho, rho_old)
                    )
                    total_loss += current_tvd_weight * loss_TVD
                    ux_old = ux.clone()
                    T_old = T.clone()
                    rho_old = rho.clone()
                elif self.tvd_enabled and rollout == 0:
                    # Initialize tracking variables
                    ux_old = ux.clone()
                    T_old = T.clone()
                    rho_old = rho.clone()

                # Collision
                Geq_pred_reshaped = Geq_pred.permute(1, 0).reshape(
                    self.solver.Qn, self.solver.Y, self.solver.X
                )
                Fi0, Gi0 = self.solver.collision(
                    Fi0, Gi0, Feq, Geq_pred_reshaped, rho, ux, uy, T
                )

                # Streaming
                Fi, Gi = self.solver.streaming(Fi0, Gi0)

                # Handle obstacle and BC (Cylinder cases)
                khi = detach(torch.zeros_like(ux))
                zetax = detach(torch.zeros_like(ux))
                zetay = detach(torch.zeros_like(ux))

                Fi, Gi = self._handle_obstacle_and_bc(
                    Fi, Gi, rho, ux, uy, T, khi, zetax, zetay
                )

                # Detach after streaming (SOD case)
                if self.detach_after_streaming:
                    Fi0 = Fi.detach()
                    Gi0 = Gi.detach()
                else:
                    Fi0 = Fi.clone()
                    Gi0 = Gi.clone()

            # Backward pass
            total_loss.backward()
            self.adaptive_clipper.step(self, self.model)

            # Grad and weight norms once per epoch (use last batch's gradients)
            is_last_batch = batch_idx == len(self.dataloader) - 1
            if is_last_batch:
                with torch.no_grad():
                    grad = self.get_norm(self.model, "grad")
                    weight = self.get_norm(self.model, "weights")
                self._last_epoch_grad_norm_avg = float(grad.item())
                self._last_epoch_weight_norm_avg = float(weight.item())

            self.optimizer.step()
            if self.scheduler is not None and type(self.scheduler).__name__ != "ReduceLROnPlateau":
                self.scheduler.step()

            # Track batch loss (exclude non-finite so one bad batch doesn't poison epoch metric)
            batch_loss = total_loss.item() / self.num_rollout
            loss_val = total_loss.item()
            if math.isfinite(loss_val):
                loss_epoch += loss_val
                num_batches += 1
            else:
                self.log.warning(
                    f"Epoch {epoch + 1} Batch {batch_idx + 1}: non-finite loss {loss_val}, "
                    "excluding from epoch average"
                )

            # Global step for experiment tracking
            self.global_step += 1

            # Terminal logging every N batches
            if (
                self.log_to_screen
                and self.terminal_batch_log_interval > 0
                and (batch_idx + 1) % self.terminal_batch_log_interval == 0
            ):
                self.log.info(
                    f"Epoch {epoch + 1} Batch {batch_idx + 1}/{len(self.dataloader)} "
                    f"- loss={batch_loss:.6f}"
                )

            # Experiment tracker logging every M batches (step_metric so WandB uses batch x-axis)
            if (
                self.tracker_batch_log_interval > 0
                and (batch_idx + 1) % self.tracker_batch_log_interval == 0
            ):
                metrics = {"train/batch_loss": float(batch_loss)}
                grad_stats = self.adaptive_clipper.stats_for_logging(
                    warn=False, logger=self.log
                )
                metrics.update(grad_stats)
                self.tracker.log_metrics(
                    metrics, step=self.global_step, step_metric="batch"
                )
                self.adaptive_clipper.reset_log_interval_stats()

        avg_loss = loss_epoch / num_batches if num_batches > 0 else 0.0
        if num_batches == 0:
            self._last_epoch_grad_norm_avg = None
            self._last_epoch_weight_norm_avg = None

        # Validation (always enabled)
        val_loss = self._validate()
        self.log.info(
            f"Epoch: {epoch}, Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}"
        )

        # Run validation rollout once per epoch for field-error metrics (and image cache)
        rollout_data = self._run_validation_rollout_for_plotting()
        if rollout_data is not None:
            self._validation_rollout_cache = (rollout_data, epoch)
            self._last_epoch_field_errors = self._field_errors_from_rollout_data(
                rollout_data
            )
        else:
            self._validation_rollout_cache = None
            self._last_epoch_field_errors = {}

        # Store train loss for build_epoch_log_metrics (val/epoch_loss and train/epoch_loss)
        self._last_epoch_train_loss = float(avg_loss)
        # Use validation loss for best model tracking
        return val_loss

    def _validate(self) -> float:
        """
        Run validation loop.

        Returns
        -------
        float
            Average validation loss
        """
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            # Get initial state from validation dataset
            val_iter = iter(self.val_dataset)
            Fi0 = next(val_iter)[0].to(self.device)
            Gi0 = next(val_iter)[1].to(self.device)

            for F_val, G_val, Feq_val, Geq_val in self.val_dataset:
                # Get macroscopic variables
                rho, ux, uy, E = self.solver.get_macroscopic(Fi0, Gi0)
                T = self.solver.get_temp_from_energy(ux, uy, E)

                # Get equilibrium F
                Feq = self.solver.get_Feq(rho, ux, uy, T)

                # Prepare model input
                inputs = torch.stack(
                    [
                        rho.unsqueeze(0),
                        ux.unsqueeze(0),
                        uy.unsqueeze(0),
                        T.unsqueeze(0),
                    ],
                    dim=1,
                ).to(self.device)

                # Model prediction
                Geq_pred = self.model(inputs, self.basis)
                Geq_target = Geq_val.to(self.device)

                # Compute loss
                inner_loss = l2_error(
                    Geq_pred, Geq_target.permute(1, 2, 0).reshape(-1, 9)
                )
                val_loss += inner_loss.item()

                # Collision
                Geq_pred_reshaped = Geq_pred.permute(1, 0).reshape(
                    self.solver.Qn, self.solver.Y, self.solver.X
                )
                Fi0, Gi0 = self.solver.collision(
                    Fi0, Gi0, Feq, Geq_pred_reshaped, rho, ux, uy, T
                )

                # Streaming
                Fi, Gi = self.solver.streaming(Fi0, Gi0)

                # Detach after streaming (SOD case)
                if self.detach_after_streaming:
                    Fi0 = Fi.detach()
                    Gi0 = Gi.detach()
                else:
                    Fi0 = Fi.clone()
                    Gi0 = Gi.clone()

        return val_loss / len(self.val_dataset)

    def build_epoch_log_metrics(
        self, epoch: int, primary_loss: float
    ) -> Dict[str, Any]:
        """
        Build epoch-level metrics for the experiment tracker.

        Stage 2 logs both train/epoch_loss and val/epoch_loss (primary_loss is val loss),
        plus validation field errors (rho, T, ux, P, and uy for cylinder).
        """
        metrics: Dict[str, Any] = {
            "val/epoch_loss": float(primary_loss),
            "epoch": epoch + 1,
        }
        if self._last_epoch_train_loss is not None:
            metrics["train/epoch_loss"] = self._last_epoch_train_loss
        if self._last_epoch_field_errors:
            metrics.update(self._last_epoch_field_errors)
        return metrics

    def update_best_models(self, current_loss: float, epoch: int) -> bool:
        """
        Update top 3 best models with case-specific naming.

        Parameters
        ----------
        current_loss : float
            Current validation loss value
        epoch : int
            Current epoch

        Returns
        -------
        bool
            True if this epoch updated any of the top-3 best.
        """
        if current_loss < max(self.best_losses):
            max_index = self.best_losses.index(max(self.best_losses))
            self.best_losses[max_index] = current_loss
            self.best_models[max_index] = self.model.state_dict()

            # Always save the first best model, then respect save_frequency for subsequent saves
            is_first_save = self.best_model_paths[max_index] is None
            if self.save_model and (
                is_first_save
                or self.epochs_since_last_save[max_index] >= self.save_frequency
            ):
                # Remove old checkpoint if exists
                if self.best_model_paths[max_index] and os.path.exists(
                    self.best_model_paths[max_index]
                ):
                    os.remove(self.best_model_paths[max_index])

                # Save new checkpoint with case-specific naming
                if self.case_name:
                    save_path = os.path.join(
                        self.model_dir,
                        f"best_model_{self.case_name}_epoch_{epoch+1}_top_{max_index+1}_val_loss_{current_loss:.6f}.pt",
                    )
                else:
                    save_path = os.path.join(
                        self.model_dir,
                        f"best_model_epoch_{epoch+1}_top_{max_index+1}_val_loss_{current_loss:.6f}.pt",
                    )
                torch.save(self.best_models[max_index], save_path)
                self.log.info(f"Top {max_index+1} model saved to: {save_path}")
                self.best_model_paths[max_index] = save_path
                self.epochs_since_last_save[max_index] = 0
            else:
                self.epochs_since_last_save[max_index] += 1
            return True
        else:
            # Increment all counters
            for i in range(3):
                self.epochs_since_last_save[i] += 1
            return False

    def _field_errors_from_rollout_data(self, data: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute mean absolute errors for density, temperature, velocity, pressure
        from NN vs GT rollout data. Used for experiment-tracking metrics.
        """

        def to_np(x):
            if hasattr(x, "detach"):
                x = x.detach().cpu().numpy()
            return np.asarray(x).reshape(-1)

        rho_nn = to_np(data["rho_NN"])
        rho_gt = to_np(data["rho_GT"])
        T_nn = to_np(data["T_NN"])
        T_gt = to_np(data["T_GT"])
        ux_nn = to_np(data["ux_NN"])
        ux_gt = to_np(data["ux_GT"])
        P_nn = to_np(data["P_NN"])
        P_gt = to_np(data["P_GT"])

        errors: Dict[str, float] = {
            "val/error_rho": float(np.mean(np.abs(rho_nn - rho_gt))),
            "val/error_T": float(np.mean(np.abs(T_nn - T_gt))),
            "val/error_ux": float(np.mean(np.abs(ux_nn - ux_gt))),
            "val/error_P": float(np.mean(np.abs(P_nn - P_gt))),
        }
        if "uy_NN" in data and "uy_GT" in data:
            uy_nn = to_np(data["uy_NN"])
            uy_gt = to_np(data["uy_GT"])
            errors["val/error_uy"] = float(np.mean(np.abs(uy_nn - uy_gt)))
        return errors

    def _run_validation_rollout_for_plotting(self) -> Optional[Dict[str, Any]]:
        """
        Run a short validation rollout from a fixed initial condition and return
        NN vs GT fields at the end of the rollout for plotting.

        Returns
        -------
        Optional[Dict[str, Any]]
            Plot data (Mach for Cylinder; rho, ux, T, P for SOD), or None if
            validation set is too small to run at least one step.
        """
        n_val = len(self.val_dataset)
        plot_steps = min(self.num_rollout, max(0, n_val - 1))
        if plot_steps <= 0:
            return None

        self.model.eval()
        with torch.no_grad():
            F0, G0, _, _ = self.val_dataset[0]
            Fi0 = F0.unsqueeze(0).to(self.device)
            Gi0 = G0.unsqueeze(0).to(self.device)

            for _ in range(plot_steps):
                rho, ux, uy, E = self.solver.get_macroscopic(Fi0[0], Gi0[0])
                T = self.solver.get_temp_from_energy(ux, uy, E)
                Feq = self.solver.get_Feq(rho, ux, uy, T)
                inputs = torch.stack(
                    [
                        rho.unsqueeze(0),
                        ux.unsqueeze(0),
                        uy.unsqueeze(0),
                        T.unsqueeze(0),
                    ],
                    dim=1,
                ).to(self.device)
                Geq_pred = self.model(inputs, self.basis)
                Geq_pred_reshaped = Geq_pred.permute(1, 0).reshape(
                    self.solver.Qn, self.solver.Y, self.solver.X
                )
                Fi0, Gi0 = self.solver.collision(
                    Fi0[0], Gi0[0], Feq, Geq_pred_reshaped, rho, ux, uy, T
                )
                Fi, Gi = self.solver.streaming(Fi0, Gi0)
                khi = detach(torch.zeros_like(ux))
                zetax = detach(torch.zeros_like(ux))
                zetay = detach(torch.zeros_like(ux))
                Fi, Gi = self._handle_obstacle_and_bc(
                    Fi, Gi, rho, ux, uy, T, khi, zetax, zetay
                )
                if self.detach_after_streaming:
                    Fi0 = Fi.detach().unsqueeze(0)
                    Gi0 = Gi.detach().unsqueeze(0)
                else:
                    Fi0 = Fi.unsqueeze(0).clone()
                    Gi0 = Gi.unsqueeze(0).clone()

            rho_nn, ux_nn, uy_nn, E_nn = self.solver.get_macroscopic(Fi0[0], Gi0[0])
            T_nn = self.solver.get_temp_from_energy(ux_nn, uy_nn, E_nn)

            F_gt, G_gt, _, _ = self.val_dataset[plot_steps]
            F_gt = F_gt.to(self.device)
            G_gt = G_gt.to(self.device)
            rho_gt, ux_gt, uy_gt, E_gt = self.solver.get_macroscopic(F_gt, G_gt)
            T_gt = self.solver.get_temp_from_energy(ux_gt, uy_gt, E_gt)

            is_cylinder = hasattr(self.solver, "get_local_Mach")
            if is_cylinder:
                Ma_NN = self.solver.get_local_Mach(ux_nn, uy_nn, T_nn)
                Ma_GT = self.solver.get_local_Mach(ux_gt, uy_gt, T_gt)
                P_nn = self.solver.get_pressure(T_nn, rho_nn)
                P_gt = self.solver.get_pressure(T_gt, rho_gt)
                return {
                    "Ma_NN": Ma_NN.cpu().numpy(),
                    "Ma_GT": Ma_GT.cpu().numpy(),
                    "rho_NN": rho_nn.cpu().numpy(),
                    "ux_NN": ux_nn.cpu().numpy(),
                    "uy_NN": uy_nn.cpu().numpy(),
                    "T_NN": T_nn.cpu().numpy(),
                    "P_NN": P_nn.cpu().numpy(),
                    "rho_GT": rho_gt.cpu().numpy(),
                    "ux_GT": ux_gt.cpu().numpy(),
                    "uy_GT": uy_gt.cpu().numpy(),
                    "T_GT": T_gt.cpu().numpy(),
                    "P_GT": P_gt.cpu().numpy(),
                    "time_value": plot_steps,
                    "case_type": "cylinder",
                }
            else:
                P_nn = self.solver.get_pressure(T_nn, rho_nn)
                P_gt = self.solver.get_pressure(T_gt, rho_gt)
                case_number = int(self.case_name) if self.case_name else 1
                return {
                    "rho_NN": rho_nn,
                    "ux_NN": ux_nn,
                    "T_NN": T_nn,
                    "P_NN": P_nn,
                    "rho_GT": rho_gt.cpu().numpy(),
                    "ux_GT": ux_gt.cpu().numpy(),
                    "T_GT": T_gt.cpu().numpy(),
                    "P_GT": P_gt.cpu().numpy(),
                    "time_step": plot_steps,
                    "case_number": case_number,
                }

    def log_validation_image(self, epoch: int) -> None:
        """
        Run a validation rollout for plotting, build the comparison figure,
        convert to PIL, log to the experiment tracker, and close the figure.
        Uses cached rollout data when available (same epoch) to avoid double rollout.
        """
        if (
            self._validation_rollout_cache is not None
            and self._validation_rollout_cache[1] == epoch
        ):
            data = self._validation_rollout_cache[0]
        else:
            data = self._run_validation_rollout_for_plotting()
        if data is None:
            return
        if "Ma_NN" in data:
            fig = plot_cylinder_results(
                data["Ma_NN"],
                data["Ma_GT"],
                data["time_value"],
                case_type=data.get("case_type", "cylinder"),
                save=False,
            )
        else:
            fig = plot_sod_results(
                data["rho_NN"],
                data["ux_NN"],
                data["T_NN"],
                data["P_NN"],
                data["rho_GT"],
                data["ux_GT"],
                data["T_GT"],
                data["P_GT"],
                data["time_step"],
                data["case_number"],
                save=False,
            )
        img = fig_to_image(fig)
        self.tracker.log_image(
            img,
            key="validation_plot",
            step=epoch + 1,
            step_metric="epoch",
        )
        plt.close(fig)
