"""
Stage 1 training class.

This module provides the Stage1Trainer class for training the model on
equilibrium state data (predicting Geq from macroscopic variables).
"""

import os
import torch
import torch.nn as nn
from typing import Optional
from torch.utils.data import DataLoader
from tqdm import tqdm

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

        # Create inner progress bar for batches (nested under epoch bar)
        batch_pbar = tqdm(
            self.dataloader,
            desc=f"Epoch {epoch}",
            position=1,
            leave=False,
            disable=False,
        )
        for rho_batch, ux_batch, uy_batch, T_batch, Geq_batch in batch_pbar:
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            loss_epoch += loss.item()
            num_batches += 1

            # Update inner progress bar
            batch_pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_loss = loss_epoch / num_batches if num_batches > 0 else 0.0

        return avg_loss

    def update_best_models(self, current_loss: float, epoch: int):
        """
        Update top 3 best models with case-specific naming.

        Parameters
        ----------
        current_loss : float
            Current loss value
        epoch : int
            Current epoch
        """
        if current_loss < max(self.best_losses):
            max_index = self.best_losses.index(max(self.best_losses))
            self.best_losses[max_index] = current_loss
            self.best_models[max_index] = self.model.state_dict()

            # Always save the first best model, then respect save_frequency for subsequent saves
            is_first_save = self.best_model_paths[max_index] is None
            if (
                self.save_model
                and (is_first_save or self.epochs_since_last_save[max_index] >= self.save_frequency)
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
                print(f"Top {max_index+1} model saved to: {save_path}")
                self.best_model_paths[max_index] = save_path
                self.epochs_since_last_save[max_index] = 0
            else:
                self.epochs_since_last_save[max_index] += 1
        else:
            # Increment all counters
            for i in range(3):
                self.epochs_since_last_save[i] += 1
