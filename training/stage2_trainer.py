"""
Stage 2 training class.

This module provides the Stage2Trainer class for training the model on
rollout sequences with solver integration.
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Any
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from .base import BaseTrainer
from utils.datasets import Stage2Dataset
from utils.loss import l2_error, TVD_norm
from utils.case_specific import tvd_weight_scheduler
from utils.core import detach


class Stage2Trainer(BaseTrainer):
    """
    Trainer for Stage 2 (rollout training with solver integration).

    Trains the model on rollout sequences, integrating with the solver
    for collision and streaming operations.

    Usage by case:
    -------------
    Cylinder/Cylinder_faster:
        - TVD: False (default)
        - Validation: False (default)
        - Detach after streaming: False (default)
        - EMA: True (default, always enabled)

    SOD_shock_tube:
        - TVD: Optional (configurable, used in case 2)
        - Validation: Optional (configurable)
        - Detach after streaming: True (default for SOD)
        - EMA: True (default, always enabled)
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
        save_model: bool = True,
        save_frequency: int = 1,
        case_name: Optional[str] = None,
        # Optional features
        tvd_enabled: bool = False,
        tvd_weight: float = 15.0,
        tvd_milestones: Optional[list] = None,
        tvd_weights: Optional[list] = None,
        validation_enabled: bool = False,
        val_dataset: Optional[Stage2Dataset] = None,
        detach_after_streaming: bool = False,
        ema_alpha: float = 0.1,
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
        save_model : bool, optional
            Whether to save model checkpoints (default: True)
        save_frequency : int, optional
            Minimum epochs between saves (default: 1)
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
        validation_enabled : bool, optional
            Whether to run validation loop (default: False)
        val_dataset : Optional[Stage2Dataset], optional
            Validation dataset (required if validation_enabled=True)
        detach_after_streaming : bool, optional
            Whether to detach gradients after streaming (default: False)
        ema_alpha : float, optional
            EMA smoothing factor for batch loss (default: 0.1)
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            model_dir=model_dir,
            save_model=save_model,
            save_frequency=save_frequency,
        )
        self.dataloader = dataloader
        self.solver = solver
        self.basis = basis
        self.num_rollout = num_rollout
        self.case_name = case_name

        # TVD settings
        self.tvd_enabled = tvd_enabled
        self.tvd_weight = tvd_weight
        self.tvd_milestones = tvd_milestones or []
        self.tvd_weights = tvd_weights or [tvd_weight]

        # Validation settings
        self.validation_enabled = validation_enabled
        self.val_dataset = val_dataset
        if validation_enabled and val_dataset is None:
            raise ValueError("val_dataset must be provided if validation_enabled=True")

        # Detaching behavior
        self.detach_after_streaming = detach_after_streaming

        # EMA tracking (always enabled by default)
        self.ema_alpha = ema_alpha
        self.ema_batch_loss = None

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
        batch_losses = []

        # Get TVD weight for this epoch
        if self.tvd_enabled:
            current_tvd_weight = tvd_weight_scheduler(
                epoch, self.tvd_milestones, self.tvd_weights
            )
        else:
            current_tvd_weight = 0.0

        # Create inner progress bar for batches (nested under epoch bar)
        batch_pbar = tqdm(
            self.dataloader,
            desc=f"Epoch {epoch}",
            position=1,
            leave=False,
            disable=False,
        )
        for batch_idx, (F_seq, G_seq, Feq_seq, Geq_seq) in enumerate(batch_pbar):
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track batch loss
            batch_loss = total_loss.item() / self.num_rollout
            loss_epoch += total_loss.item()
            batch_losses.append(batch_loss)
            num_batches += 1

            # Update EMA
            if self.ema_batch_loss is None:
                self.ema_batch_loss = batch_loss
            else:
                self.ema_batch_loss = (
                    self.ema_alpha * batch_loss
                    + (1 - self.ema_alpha) * self.ema_batch_loss
                )

            # Calculate std of batch losses
            if len(batch_losses) > 1:
                batch_std = float(np.std(batch_losses))
            else:
                batch_std = 0.0

            # Update inner progress bar
            batch_pbar.set_postfix(
                {
                    "batch": batch_idx,
                    "train_loss_ema": f"{self.ema_batch_loss:.6f}",
                    "train_loss_std": f"{batch_std:.6f}",
                }
            )

        avg_loss = loss_epoch / num_batches if num_batches > 0 else 0.0

        # Validation (if enabled)
        val_loss = None
        if self.validation_enabled:
            val_loss = self._validate()
            print(
                f"Epoch: {epoch}, Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}"
            )
        else:
            if epoch % 50 == 0 and epoch > 0:
                print(f"Epoch: {epoch}, Loss: {avg_loss:.6f}")

        # Use validation loss for best model tracking if available
        return val_loss if val_loss is not None else avg_loss

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

    def update_best_models(self, current_loss: float, epoch: int):
        """
        Update top 3 best models with case-specific naming.

        Parameters
        ----------
        current_loss : float
            Current loss value (validation loss if validation enabled, else train loss)
        epoch : int
            Current epoch
        """
        if current_loss < max(self.best_losses):
            max_index = self.best_losses.index(max(self.best_losses))
            self.best_losses[max_index] = current_loss
            self.best_models[max_index] = self.model.state_dict()

            if (
                self.save_model
                and self.epochs_since_last_save[max_index] >= self.save_frequency
            ):
                # Remove old checkpoint if exists
                if self.best_model_paths[max_index] and os.path.exists(
                    self.best_model_paths[max_index]
                ):
                    os.remove(self.best_model_paths[max_index])

                # Save new checkpoint with case-specific naming
                if self.case_name:
                    if self.validation_enabled:
                        save_path = os.path.join(
                            self.model_dir,
                            f"best_model_{self.case_name}_epoch_{epoch+1}_top_{max_index+1}_val_loss_{current_loss:.6f}.pt",
                        )
                    else:
                        save_path = os.path.join(
                            self.model_dir,
                            f"best_model_{self.case_name}_epoch_{epoch+1}_top_{max_index+1}_{current_loss:.12f}.pt",
                        )
                else:
                    if self.validation_enabled:
                        save_path = os.path.join(
                            self.model_dir,
                            f"best_model_epoch_{epoch+1}_top_{max_index+1}_val_loss_{current_loss:.6f}.pt",
                        )
                    else:
                        save_path = os.path.join(
                            self.model_dir,
                            f"best_model_epoch_{epoch+1}_top_{max_index+1}_{current_loss:.12f}.pt",
                        )
                torch.save(self.best_models[max_index], save_path)
                print(f"Top {max_index+1} model saved to: {save_path}")
                self.best_model_paths[max_index] = save_path
                self.epochs_since_last_save[max_index] = 0
            else:
                self.epochs_since_last_save[max_index] += 1
        else:
            # Increment all counters
            for i in range(3):
                self.epochs_since_last_save[i] += 1
