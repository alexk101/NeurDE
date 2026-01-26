"""
Base training utilities and abstract trainer class.

This module provides common functionality used across all training stages
and cases.
"""

import torch
import torch.nn as nn
import os
from abc import ABC, abstractmethod
from typing import Optional
from tqdm import tqdm


def create_basis(Uax, Uay, device):
    """
    Create basis vectors for the lattice Boltzmann method.

    This function is used identically across all cases (Cylinder, Cylinder_faster, SOD_shock_tube).

    Parameters
    ----------
    Uax : float or torch.Tensor
        x-component of velocity shift
    Uay : float or torch.Tensor
        y-component of velocity shift
    device : torch.device
        Device to place tensors on

    Returns
    -------
    torch.Tensor
        Basis tensor of shape (9, 2) containing (ex, ey) pairs for each velocity direction
    """
    ex_values = [1, 0, -1, 0, 1, -1, -1, 1, 0]
    ey_values = [0, 1, 0, -1, 1, 1, -1, -1, 0]
    ex = torch.tensor(ex_values, dtype=torch.float32) + Uax
    ey = torch.tensor(ey_values, dtype=torch.float32) + Uay
    basis = torch.stack([ex, ey], dim=-1).to(device)
    return basis


class BaseTrainer(ABC):
    """
    Abstract base class for training.

    Provides common functionality for model initialization, checkpoint management,
    and best model tracking (top 3 models).

    Attributes
    ----------
    model : nn.Module
        Neural network model
    device : torch.device
        Device for computation
    optimizer : torch.optim.Optimizer
        Optimizer for training
    scheduler : Optional[torch.optim.lr_scheduler._LRScheduler]
        Learning rate scheduler
    model_dir : str
        Directory to save models
    save_model : bool
        Whether to save model checkpoints
    save_frequency : int
        Minimum epochs between saves
    best_losses : list[float]
        Top 3 best losses
    best_models : list[Optional[Dict]]
        Top 3 best model state dicts
    best_model_paths : list[Optional[str]]
        Paths to saved top 3 models
    epochs_since_last_save : list[int]
        Epochs since last save for each top model
    epoch_pbar : Optional[tqdm]
        Outer progress bar for epochs (used for nested progress bars)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        device: torch.device,
        model_dir: str,
        save_model: bool = True,
        save_frequency: int = 1,
    ):
        """
        Initialize base trainer.

        Parameters
        ----------
        model : nn.Module
            Neural network model
        optimizer : torch.optim.Optimizer
            Optimizer for training
        scheduler : Optional[torch.optim.lr_scheduler._LRScheduler]
            Learning rate scheduler
        device : torch.device
            Device for computation
        model_dir : str
            Directory to save models
        save_model : bool, optional
            Whether to save model checkpoints (default: True)
        save_frequency : int, optional
            Minimum epochs between saves (default: 1)
        """
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_dir = model_dir
        self.save_model = save_model
        self.save_frequency = save_frequency

        # Best model tracking (top 3)
        self.best_losses = [float("inf")] * 3
        self.best_models = [None] * 3
        self.best_model_paths = [None] * 3
        self.epochs_since_last_save = [0] * 3

        # Progress bar for nested display (outer = epochs)
        self.epoch_pbar = None

        # Create model directory
        os.makedirs(self.model_dir, exist_ok=True)

    def load_checkpoint(self, checkpoint_path: str, handle_compile: bool = False):
        """
        Load model checkpoint.

        Handles both compiled and non-compiled model checkpoints.

        Parameters
        ----------
        checkpoint_path : str
            Path to checkpoint file
        handle_compile : bool, optional
            Whether to handle compiled model format (default: False)
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if handle_compile:
            # Handle compiled model format
            new_state_dict = {}
            for k, v in checkpoint.items():
                if k.startswith("_orig_mod."):
                    new_k = k.replace("_orig_mod.", "")
                    new_state_dict[new_k] = v
                else:
                    new_state_dict[k] = v
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(checkpoint)

        print(f"Checkpoint loaded from {checkpoint_path}")

    def save_checkpoint(self, epoch: int, loss: float, prefix: str = "model"):
        """
        Save model checkpoint.

        Parameters
        ----------
        epoch : int
            Current epoch
        loss : float
            Current loss value
        prefix : str, optional
            Filename prefix (default: "model")
        """
        save_path = os.path.join(
            self.model_dir,
            f"{prefix}_epoch_{epoch}_loss_{loss:.6f}.pt",
        )
        torch.save(self.model.state_dict(), save_path)
        return save_path

    def update_best_models(self, current_loss: float, epoch: int):
        """
        Update top 3 best models based on current loss.

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

            if (
                self.save_model
                and self.epochs_since_last_save[max_index] >= self.save_frequency
            ):
                # Remove old checkpoint if exists
                if self.best_model_paths[max_index] and os.path.exists(
                    self.best_model_paths[max_index]
                ):
                    os.remove(self.best_model_paths[max_index])

                # Save new checkpoint
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

    @abstractmethod
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

    def train(self, epochs: int):
        """
        Main training loop.

        Parameters
        ----------
        epochs : int
            Number of epochs to train
        """
        # Create outer progress bar for epochs
        self.epoch_pbar = tqdm(range(epochs), desc="Training", position=0, leave=True)
        
        for epoch in self.epoch_pbar:
            avg_loss = self.train_epoch(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            self.update_best_models(avg_loss, epoch)

            # Update outer progress bar
            self.epoch_pbar.set_postfix({"avg_loss": f"{avg_loss:.6f}"})

            # Save last model at end
            if epoch == epochs - 1 and self.save_model:
                last_path = os.path.join(
                    self.model_dir,
                    f"last_model_epoch_{epochs}_loss_{avg_loss:.6f}.pt",
                )
                torch.save(self.model.state_dict(), last_path)
                print(f"Last model saved to: {last_path}")
        
        # Close outer progress bar
        if self.epoch_pbar is not None:
            self.epoch_pbar.close()
            self.epoch_pbar = None
