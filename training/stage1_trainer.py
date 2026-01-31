"""
Stage 1 training class.

This module provides the Stage1Trainer class for training the model on
equilibrium state data (predicting Geq from macroscopic variables).
"""

import os
from omegaconf import DictConfig
import torch
import torch.nn as nn
from typing import Optional
from torch.utils.data import DataLoader

from .base import BaseTrainer
from utils.loss import l2_error


class Stage1Trainer(BaseTrainer):
    """
    Trainer for Stage 1 (equilibrium state prediction).

    Trains the model to predict equilibrium G distribution function (Geq)
    from macroscopic variables (rho, ux, uy, T).

    Usage by case:
    -------------
    All cases (Cylinder, Cylinder_faster, SOD_shock_tube) use the same
    Stage 1 training procedure. Differences are handled via:
    - Velocity shift calculation (Uax, Uay) - configurable
    - Model save naming (case number for SOD) - configurable
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        dataloader: DataLoader,
        device: torch.device,
        model_dir: str,
        basis: torch.Tensor,
        save_model: bool = True,
        save_frequency: int = 1,
        checkpoint_frequency: int = 0,
        keep_checkpoints: int = 5,
        resume_from: Optional[str] = None,
        case_name: Optional[str] = None,
        cfg: DictConfig = None,
    ):
        """
        Initialize Stage 1 trainer.

        Parameters
        ----------
        model : nn.Module
            Neural network model
        optimizer : torch.optim.Optimizer
            Optimizer for training
        scheduler : Optional[torch.optim.lr_scheduler._LRScheduler]
            Learning rate scheduler
        dataloader : DataLoader
            DataLoader for training data
        device : torch.device
            Device for computation
        model_dir : str
            Directory to save models
        basis : torch.Tensor
            Basis vectors for lattice Boltzmann method (from create_basis)
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
            Case name for model save naming (e.g., "1", "2" for SOD cases)
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
        self.basis = basis
        self.case_name = case_name

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

        for batch_idx, (rho_batch, ux_batch, uy_batch, T_batch, Geq_batch) in enumerate(
            self.dataloader
        ):
            # Prepare input data: stack (rho, ux, uy, T) into (batch, 4, Y, X)
            input_data = torch.stack(
                [rho_batch, ux_batch, uy_batch, T_batch], dim=1
            ).to(self.device)

            # Prepare targets: reshape Geq from (batch, Q, Y, X) to (batch*Y*X, Q)
            targets = Geq_batch.permute(0, 2, 3, 1).reshape(-1, 9).to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            Geq_pred = self.model(input_data, self.basis)

            # Compute loss
            loss = l2_error(Geq_pred, targets)

            # Backward pass
            loss.backward()
            self.adaptive_clipper.step(self, self.model)

            # Grad and weight norms once per epoch (use last batch's gradients)
            is_last_batch = batch_idx == len(self.dataloader) - 1
            if is_last_batch:
                with torch.no_grad():
                    grad_sq = self.get_norm(self.model, "grad")
                    weight_sq = self.get_norm(self.model, "weights")
                self._last_epoch_grad_norm_avg = float(grad_sq.sqrt().item())
                self._last_epoch_weight_norm_avg = float(weight_sq.sqrt().item())

            self.optimizer.step()

            loss_epoch += loss.item()
            num_batches += 1

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
                    f"- loss={loss.item():.6f}"
                )

            # Experiment tracker logging every M batches (step_metric so WandB uses batch x-axis)
            if (
                self.tracker_batch_log_interval > 0
                and (batch_idx + 1) % self.tracker_batch_log_interval == 0
            ):
                metrics = {"train/batch_loss": float(loss.item())}
                grad_stats = self.adaptive_clipper.stats_for_logging(
                    warn=False, logger=self.log
                )
                metrics.update(grad_stats)
                self.tracker.log_metrics(
                    metrics, step=self.global_step, step_metric="batch"
                )

        avg_loss = loss_epoch / num_batches if num_batches > 0 else 0.0
        if num_batches == 0:
            self._last_epoch_grad_norm_avg = None
            self._last_epoch_weight_norm_avg = None

        return avg_loss

    def update_best_models(self, current_loss: float, epoch: int) -> bool:
        """
        Update top 3 best models with case-specific naming.

        Parameters
        ----------
        current_loss : float
            Current loss value
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
                        f"best_model_{self.case_name}_epoch_{epoch+1}_top_{max_index+1}_loss_{current_loss:.6f}.pt",
                    )
                else:
                    save_path = os.path.join(
                        self.model_dir,
                        f"best_model_epoch_{epoch+1}_top_{max_index+1}_loss_{current_loss:.6f}.pt",
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
