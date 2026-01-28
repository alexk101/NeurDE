"""
Base training utilities and abstract trainer class.

This module provides common functionality used across all training stages
and cases.
"""

import torch
import torch.nn as nn
import os
import glob
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from tqdm import tqdm

from utils.core import adapt_checkpoint_keys


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
        checkpoint_frequency: int = 0,
        keep_checkpoints: int = 5,
        resume_from: Optional[str] = None,
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
            Minimum epochs between saves for best models (default: 1)
        checkpoint_frequency : int, optional
            Save full checkpoint every N epochs (0 = disabled, default: 0)
        keep_checkpoints : int, optional
            Number of periodic checkpoints to keep (default: 5)
        resume_from : Optional[str], optional
            Path to checkpoint to resume from (default: None)
        """
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_dir = model_dir
        self.save_model = save_model
        self.save_frequency = save_frequency
        self.checkpoint_frequency = checkpoint_frequency
        self.keep_checkpoints = keep_checkpoints

        # Best model tracking (top 3)
        self.best_losses = [float("inf")] * 3
        self.best_models = [None] * 3
        self.best_model_paths = [None] * 3
        self.epochs_since_last_save = [0] * 3

        # Loss history for tracking
        self.loss_history = []

        # Progress bar for nested display (outer = epochs)
        self.epoch_pbar = None

        # Create model directory
        os.makedirs(self.model_dir, exist_ok=True)

        # Resume from checkpoint if provided
        self.start_epoch = 0
        if resume_from is not None:
            self.start_epoch = self.resume_from_checkpoint(resume_from)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint (model state_dict only).

        Handles both compiled and non-compiled model checkpoints.
        Automatically detects and handles format mismatches.
        For full checkpoint loading with optimizer/scheduler, use resume_from_checkpoint.

        Parameters
        ----------
        checkpoint_path : str
            Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Adapt checkpoint keys to match model format (handles torch.compile prefix)
        state_dict = adapt_checkpoint_keys(checkpoint, self.model)
        self.model.load_state_dict(state_dict)
        print(f"Checkpoint loaded from {checkpoint_path}")

    def resume_from_checkpoint(self, checkpoint_path: str) -> int:
        """
        Resume training from a full checkpoint.

        Loads model, optimizer, scheduler, epoch, and loss history.

        Parameters
        ----------
        checkpoint_path : str
            Path to full checkpoint file

        Returns
        -------
        int
            Epoch to resume from (next epoch to train)

        Raises
        ------
        ValueError
            If checkpoint is missing required keys (model_state_dict or optimizer_state_dict)
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Check if it's a full checkpoint or just a state_dict
        if "model_state_dict" not in checkpoint and "optimizer_state_dict" not in checkpoint:
            # This looks like just a state_dict, not a full checkpoint
            raise ValueError(
                f"Checkpoint {checkpoint_path} appears to be a model state_dict only, "
                "not a full checkpoint. Use 'pretrained_path' for loading model weights only, "
                "or use 'resume_from' with a full checkpoint created by save_full_checkpoint()."
            )

        # Load model state using adapt_checkpoint_keys to handle compiled model format
        state_dict = adapt_checkpoint_keys(checkpoint, self.model)
        self.model.load_state_dict(state_dict)

        # Load optimizer state
        if "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except ValueError as e:
                print(f"Warning: Could not load optimizer state: {e}")
                print("Continuing with fresh optimizer state.")

        # Load scheduler state
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except ValueError as e:
                print(f"Warning: Could not load scheduler state: {e}")
                print("Continuing with fresh scheduler state.")

        # Load training state
        start_epoch = checkpoint.get("epoch", 0)
        if "loss_history" in checkpoint:
            self.loss_history = checkpoint["loss_history"]
        if "best_losses" in checkpoint:
            self.best_losses = checkpoint["best_losses"]
        if "best_model_paths" in checkpoint:
            self.best_model_paths = checkpoint["best_model_paths"]

        print(f"Resumed from checkpoint: {checkpoint_path}")
        print(f"Resuming from epoch {start_epoch + 1}")
        if self.loss_history:
            print(f"Previous loss history: {len(self.loss_history)} epochs")
        return start_epoch + 1

    def save_checkpoint(self, epoch: int, loss: float, prefix: str = "model"):
        """
        Save model checkpoint (model state_dict only).

        For full checkpoint with optimizer/scheduler, use save_full_checkpoint.

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

    def save_full_checkpoint(
        self, epoch: int, loss: float, prefix: str = "checkpoint", metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save full training checkpoint (model, optimizer, scheduler, training state).

        Parameters
        ----------
        epoch : int
            Current epoch
        loss : float
            Current loss value
        prefix : str, optional
            Filename prefix (default: "checkpoint")
        metadata : Optional[Dict[str, Any]], optional
            Additional metadata to save (default: None)

        Returns
        -------
        str
            Path to saved checkpoint
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "loss_history": self.loss_history,
            "best_losses": self.best_losses,
            "best_model_paths": self.best_model_paths,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if metadata is not None:
            checkpoint["metadata"] = metadata

        save_path = os.path.join(
            self.model_dir,
            f"{prefix}_epoch_{epoch}_loss_{loss:.6f}.pt",
        )
        torch.save(checkpoint, save_path)
        return save_path

    def _cleanup_old_checkpoints(self, prefix: str = "checkpoint"):
        """
        Clean up old periodic checkpoints, keeping only the most recent N.

        Parameters
        ----------
        prefix : str, optional
            Checkpoint prefix to match (default: "checkpoint")
        """
        if self.keep_checkpoints <= 0:
            return

        # Find all checkpoint files with the given prefix
        pattern = os.path.join(self.model_dir, f"{prefix}_epoch_*_loss_*.pt")
        checkpoint_files = glob.glob(pattern)

        if len(checkpoint_files) <= self.keep_checkpoints:
            return

        # Sort by modification time (newest first)
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)

        # Remove oldest checkpoints
        for old_checkpoint in checkpoint_files[self.keep_checkpoints:]:
            try:
                os.remove(old_checkpoint)
            except OSError:
                pass  # File might have been removed already

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
        # Create outer progress bar for epochs (starting from start_epoch)
        self.epoch_pbar = tqdm(
            range(self.start_epoch, epochs),
            desc="Training",
            position=0,
            leave=True,
            initial=self.start_epoch,
            total=epochs,
        )

        for epoch in self.epoch_pbar:
            avg_loss = self.train_epoch(epoch)

            # Track loss history
            self.loss_history.append(avg_loss)

            if self.scheduler is not None:
                self.scheduler.step()

            self.update_best_models(avg_loss, epoch)

            # Save periodic full checkpoint
            if self.save_model and self.checkpoint_frequency > 0:
                if (epoch + 1) % self.checkpoint_frequency == 0:
                    self.save_full_checkpoint(epoch, avg_loss)
                    self._cleanup_old_checkpoints()
            elif self.save_model and epoch == 0:
                # If checkpoint_frequency is 0, at least save the first epoch
                self.save_full_checkpoint(epoch, avg_loss)
                print(f"First epoch checkpoint saved (checkpoint_frequency=0, saving first epoch only)")

            # Update outer progress bar
            self.epoch_pbar.set_postfix({"avg_loss": f"{avg_loss:.6f}"})

            # Save last model at end
            if epoch == epochs - 1 and self.save_model:
                # Save as both state_dict and full checkpoint
                last_path = os.path.join(
                    self.model_dir,
                    f"last_model_epoch_{epochs}_loss_{avg_loss:.6f}.pt",
                )
                torch.save(self.model.state_dict(), last_path)
                print(f"Last model saved to: {last_path}")

                # Also save as full checkpoint
                last_full_path = self.save_full_checkpoint(
                    epoch, avg_loss, prefix="last_checkpoint"
                )
                print(f"Last full checkpoint saved to: {last_full_path}")

        # Close outer progress bar
        if self.epoch_pbar is not None:
            self.epoch_pbar.close()
            self.epoch_pbar = None
