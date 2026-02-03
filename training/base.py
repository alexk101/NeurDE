"""
Base training utilities and abstract trainer class.

This module provides common functionality used across all training stages
and cases, including checkpointing, best-model tracking, and integration
with the experiment-tracking backends.
"""

from logging import getLogger
import torch
import torch.nn as nn
import os
import glob
import json
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from omegaconf import DictConfig
from dotenv import load_dotenv

from utils.adaptive_gradient_clipper import AdaptiveGradientClipper
from utils.core import adapt_checkpoint_keys
from utils.experiment_tracking import create_experiment_tracker, ExperimentTracker
import urllib3

# SILENCE SSL WARNINGS
# This stops the "InsecureRequestWarning" spam when using self-signed certs
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


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
        cfg: DictConfig = None,
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
        cfg : DictConfig
            Hydra configuration object
        """
        load_dotenv(override=False)

        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_dir = model_dir
        self.save_model = save_model
        self.save_frequency = save_frequency
        self.checkpoint_frequency = checkpoint_frequency
        self.keep_checkpoints = keep_checkpoints
        self.log = getLogger(__name__)
        self.cfg = cfg

        # Best model tracking (top 3)
        self.best_losses = [float("inf")] * 3
        self.best_models = [None] * 3
        self.best_model_paths = [None] * 3
        self.epochs_since_last_save = [0] * 3
        self.setup_logging()

        # Loss history for tracking
        self.loss_history = []

        # Global step counter (increments per training batch)
        self.global_step: int = 0

        # Logging / experiment tracking configuration
        logging_cfg = getattr(cfg, "logging", None) if cfg is not None else None
        if logging_cfg is not None:
            # How often to log to the terminal (in batches)
            self.terminal_batch_log_interval: int = int(
                getattr(logging_cfg, "terminal_batch_log_interval", 0) or 0
            )
            # Whether to actually log to screen
            self.log_to_screen: bool = bool(getattr(logging_cfg, "log_to_screen", True))

            tracker_cfg = getattr(logging_cfg, "tracker", None)
            if tracker_cfg is not None:
                self.tracker_batch_log_interval: int = int(
                    getattr(tracker_cfg, "batch_log_interval", 0) or 0
                )
                self.log_images: bool = bool(getattr(tracker_cfg, "log_images", False))
                self.image_log_interval: int = int(
                    getattr(tracker_cfg, "image_log_interval", 0) or 0
                )
                self.image_log_max_count: int = int(
                    getattr(tracker_cfg, "image_log_max_count", 0) or 0
                )
            else:
                self.tracker_batch_log_interval = 0
                self.log_images = False
                self.image_log_interval = 0
                self.image_log_max_count = 0
        else:
            self.terminal_batch_log_interval = 0
            self.tracker_batch_log_interval = 0
            self.log_to_screen = True
            self.log_images = False
            self.image_log_interval = 0
            self.image_log_max_count = 0

        self._image_log_count: int = 0

        # When resuming, load checkpoint once to get MLflow run_id for resuming the same run
        mlflow_run_id: Optional[str] = None
        preloaded_checkpoint: Optional[Dict[str, Any]] = None
        if resume_from is not None and os.path.exists(resume_from):
            ckpt = torch.load(resume_from, map_location=self.device)
            preloaded_checkpoint = ckpt
            if isinstance(ckpt, dict) and "metadata" in ckpt:
                mlflow_run_id = ckpt["metadata"].get("mlflow_run_id")

        # Experiment tracker backend (may be a no-op tracker); pass run_id when resuming
        self.tracker: ExperimentTracker = create_experiment_tracker(
            cfg, self.log, run_id=mlflow_run_id
        )

        # Create model directory
        os.makedirs(self.model_dir, exist_ok=True)

        # Cursor file: (epoch, last_logged_batch_index). When resuming, we replay that epoch
        # from batch 0 up to last_logged_batch_index without logging, then log from the next batch.
        self._cursor_path = os.path.join(self.model_dir, "training_cursor.json")
        self._log_from_batch_index: int = 0  # first batch index from which we log (0 = log from start)

        # Resume from checkpoint if provided
        self.start_epoch = 0
        if resume_from is not None:
            if preloaded_checkpoint is not None:
                self.start_epoch = self.resume_from_checkpoint(
                    checkpoint=preloaded_checkpoint
                )
            else:
                self.start_epoch = self.resume_from_checkpoint(checkpoint_path=resume_from)
            # If cursor exists for the epoch we're about to run, don't log until we pass that batch
            self._log_from_batch_index = self._read_cursor_log_from(self.start_epoch)
            if self._log_from_batch_index > 0:
                self.log.info(
                    f"Resuming mid-epoch: will run epoch {self.start_epoch + 1} from batch 0, "
                    f"logging from batch {self._log_from_batch_index} onward (cursor file)"
                )

        # Initialize adaptive gradient clipper
        adaptive_clip_config = getattr(cfg.training, "adaptive_clip", {})
        self.adaptive_clipper = AdaptiveGradientClipper(adaptive_clip_config)
        self.log.info(
            "Initialized adaptive gradient clipper: "
            f"alpha={self.adaptive_clipper.alpha}, "
            f"z_thresh={self.adaptive_clipper.z_thresh}, "
            f"mode={self.adaptive_clipper.mode}, "
            f"warmup_steps={self.adaptive_clipper.warmup_steps}"
        )

    def setup_logging(self):
        """
        Setup logging for the trainer.
        """
        logging_cfg = (
            getattr(self.cfg, "logging", None) if self.cfg is not None else None
        )

        if logging_cfg is not None:
            tracker_cfg = getattr(logging_cfg, "tracker", {})
            backend = getattr(logging_cfg, "backend", "none")
            if backend == "mlflow":
                # A. Handle TLS verification for self-signed certs
                if tracker_cfg.get("insecure_tls", False):
                    os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"

                # B. Handle Username (Config) + Password (Env/Dotenv)
                username = tracker_cfg.get("username", None)
                if username:
                    os.environ["MLFLOW_TRACKING_USERNAME"] = username
                    if "MLFLOW_TRACKING_PASSWORD" not in os.environ:
                        self.log.warning(
                            "MLflow username set, but MLFLOW_TRACKING_PASSWORD not found in environment!"
                        )

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
        self.log.info(f"Checkpoint loaded from {checkpoint_path}")

    def _read_cursor_log_from(self, for_epoch: int) -> int:
        """
        Read cursor file. Returns the first batch index from which we should log.
        Cursor stores (epoch, last_logged_batch_index). If it matches for_epoch,
        we replay that epoch up to last_logged_batch_index without logging, then log from the next batch.
        """
        if not os.path.exists(self._cursor_path):
            return 0
        try:
            with open(self._cursor_path) as f:
                data = json.load(f)
            if int(data.get("epoch", -1)) != for_epoch:
                return 0
            last_batch = int(data.get("batch_index", -1))
            return last_batch + 1
        except (json.JSONDecodeError, ValueError, OSError):
            return 0

    def _write_cursor(self, epoch: int, batch_index: int) -> None:
        """Write cursor file (epoch, last batch at which we logged). On resume we replay that epoch and log only after this batch."""
        try:
            with open(self._cursor_path, "w") as f:
                json.dump({"epoch": epoch, "batch_index": batch_index}, f)
        except OSError as e:
            self.log.warning(f"Could not write cursor file: {e}")

    def resume_from_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        checkpoint: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Resume training from a full checkpoint.

        Loads model, optimizer, scheduler, epoch, and loss history.
        Provide either checkpoint_path (load from file) or checkpoint (preloaded dict).

        Parameters
        ----------
        checkpoint_path : Optional[str]
            Path to full checkpoint file (used if checkpoint is None)
        checkpoint : Optional[Dict]
            Preloaded checkpoint dict (avoids loading twice when resuming MLflow run)

        Returns
        -------
        int
            Epoch to resume from (next epoch to train)

        Raises
        ------
        ValueError
            If checkpoint is missing required keys (model_state_dict or optimizer_state_dict)
        """
        if checkpoint is not None:
            pass  # use provided dict
        elif checkpoint_path is not None and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        else:
            if checkpoint_path:
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            raise ValueError("Provide either checkpoint_path or checkpoint")

        # Check if it's a full checkpoint or just a state_dict
        if (
            "model_state_dict" not in checkpoint
            and "optimizer_state_dict" not in checkpoint
        ):
            path_desc = checkpoint_path or "(in-memory checkpoint)"
            raise ValueError(
                f"Checkpoint {path_desc} appears to be a model state_dict only, "
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
                self.log.info(f"Warning: Could not load optimizer state: {e}")
                self.log.info("Continuing with fresh optimizer state.")

        # Load scheduler state
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except ValueError as e:
                self.log.info(f"Warning: Could not load scheduler state: {e}")
                self.log.info("Continuing with fresh scheduler state.")

        # Load training state
        start_epoch = checkpoint.get("epoch", 0)
        if "loss_history" in checkpoint:
            self.loss_history = checkpoint["loss_history"]
        if "best_losses" in checkpoint:
            self.best_losses = checkpoint["best_losses"]
        if "best_model_paths" in checkpoint:
            self.best_model_paths = checkpoint["best_model_paths"]
        if "global_step" in checkpoint:
            self.global_step = int(checkpoint["global_step"])

        # Load adaptive gradient clipper state (EMA, buffer, etc.)
        if "adaptive_clipper_state_dict" in checkpoint:
            try:
                self.adaptive_clipper.load_state_dict(
                    checkpoint["adaptive_clipper_state_dict"], strict=False
                )
                self.log.info("Restored adaptive gradient clipper state from checkpoint.")
            except (KeyError, ValueError) as e:
                self.log.info(
                    f"Warning: Could not load adaptive clipper state: {e}. "
                    "Clipper will reinitialize (warmup)."
                )

        self.log.info(
            f"Resumed from checkpoint: {checkpoint_path or '(preloaded)'}"
        )
        self.log.info(f"Resuming from epoch {start_epoch + 1}, global_step={self.global_step}")
        if self.loss_history:
            self.log.info(f"Previous loss history: {len(self.loss_history)} epochs")
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
        self,
        epoch: int,
        loss: float,
        prefix: str = "checkpoint",
        metadata: Optional[Dict[str, Any]] = None,
        path: Optional[str] = None,
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
        path : Optional[str], optional
            If given, save to this path (single file, e.g. last_checkpoint.pt).

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
            "global_step": self.global_step,
            "adaptive_clipper_state_dict": self.adaptive_clipper.state_dict(),
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        meta = dict(metadata) if metadata else {}
        if hasattr(self.tracker, "get_run_id") and self.tracker.get_run_id():
            meta["mlflow_run_id"] = self.tracker.get_run_id()
        if meta:
            checkpoint["metadata"] = meta

        if path is None:
            path = os.path.join(
                self.model_dir,
                f"{prefix}_epoch_{epoch}_loss_{loss:.6f}.pt",
            )
        torch.save(checkpoint, path)
        return path

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
        for old_checkpoint in checkpoint_files[self.keep_checkpoints :]:
            try:
                os.remove(old_checkpoint)
            except OSError:
                pass  # File might have been removed already

    def update_best_models(self, current_loss: float, epoch: int) -> bool:
        """
        Update top 3 best models based on current loss (validation loss).

        Parameters
        ----------
        current_loss : float
            Current loss value (primary metric, e.g. validation loss)
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

                # Save new checkpoint
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

    def build_epoch_log_metrics(
        self, epoch: int, primary_loss: float
    ) -> Dict[str, Any]:
        """
        Build the dict of epoch-level metrics to log to the experiment tracker.

        Stage 1 returns train loss as primary_loss; Stage 2 returns validation loss
        and overrides this method to log both train/epoch_loss and val/epoch_loss.

        Parameters
        ----------
        epoch : int
            Current epoch index (0-based)
        primary_loss : float
            The loss value returned by train_epoch (train loss for Stage 1, val loss for Stage 2)

        Returns
        -------
        Dict[str, Any]
            Metric names -> values for logging
        """
        return {
            "train/epoch_loss": float(primary_loss),
            "epoch": epoch + 1,
        }

    @abstractmethod
    def train_epoch(self, epoch: int, log_from_batch_index: int = 0) -> float:
        """
        Train for one epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number
        log_from_batch_index : int, optional
            When resuming mid-epoch, do not log (terminal/tracker) until this batch index.
            All batches are run (forward/backward/step); only logging is skipped until then.

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
        log_from_batch = self._log_from_batch_index
        for epoch in range(self.start_epoch, epochs):
            # Only on first epoch after resume: don't log until we pass the batch in the cursor
            first_log_batch = log_from_batch if epoch == self.start_epoch else 0
            if epoch != self.start_epoch:
                log_from_batch = 0
            avg_loss = self.train_epoch(epoch, log_from_batch_index=first_log_batch)

            # Track loss history
            self.loss_history.append(avg_loss)

            # Scheduler is stepped per batch inside train_epoch (stage trainers).
            # ReduceLROnPlateau is stepped once per epoch here with the epoch loss.
            if self.scheduler is not None:
                if type(self.scheduler).__name__ == "ReduceLROnPlateau":
                    self.scheduler.step(avg_loss)

            # Update best-model tracking based on the returned loss (validation)
            updated_best = self.update_best_models(avg_loss, epoch)

            # Epoch-level logging to terminal
            if self.log_to_screen:
                self.log.info(f"Epoch {epoch + 1}/{epochs} - avg_loss={avg_loss:.6f}")

            # Epoch-level logging to experiment tracker (descriptive metric names per stage)
            if self.tracker is not None:
                metrics = self.build_epoch_log_metrics(epoch, avg_loss)
                # Learning rate (first param group)
                if self.optimizer.param_groups:
                    metrics["train/learning_rate"] = self.optimizer.param_groups[0][
                        "lr"
                    ]
                # Include gradient clipping statistics (clip count always; mean/std when adaptive)
                grad_stats = self.adaptive_clipper.stats_for_logging(
                    warn=False, logger=self.log
                )
                metrics.update(grad_stats)
                # Epoch-averaged grad norm and weight norm (set by stage trainers)
                if getattr(self, "_last_epoch_grad_norm_avg", None) is not None:
                    metrics["train/grad_norm"] = self._last_epoch_grad_norm_avg
                if getattr(self, "_last_epoch_weight_norm_avg", None) is not None:
                    metrics["train/weight_norm"] = self._last_epoch_weight_norm_avg
                # step_metric="epoch" so WandB uses epoch as x-axis; step=global_step
                # so MLflow keeps a single timeline with batch-level metrics
                self.tracker.log_metrics(
                    metrics, step=self.global_step, step_metric="epoch"
                )
                # Best epoch: log two curves when this epoch is a new best
                if updated_best:
                    train_at_best = getattr(self, "_last_epoch_train_loss", None)
                    best_metrics = {
                        "best/train_loss": float(
                            train_at_best if train_at_best is not None else avg_loss
                        ),
                        "best/val_loss": float(avg_loss),
                        "epoch": epoch + 1,
                    }
                    self.tracker.log_metrics(
                        best_metrics, step=self.global_step, step_metric="epoch"
                    )
                # Reset per-interval grad-clip counters
                self.adaptive_clipper.reset_log_interval_stats()

            # Image logging (Stage 2 only; log every image_log_interval epochs, cap at max)
            if (
                self.tracker is not None
                and self.log_images
                and self.image_log_interval > 0
                and (epoch + 1) % self.image_log_interval == 0
                and (
                    self.image_log_max_count <= 0
                    or self._image_log_count < self.image_log_max_count
                )
            ):
                log_fn = getattr(self, "log_validation_image", None)
                if callable(log_fn):
                    try:
                        log_fn(epoch)
                        self._image_log_count += 1
                    except Exception as e:
                        self.log.warning(f"Image logging failed: {e}")

            # Save periodic full checkpoint (when checkpoint_frequency > 0)
            if self.save_model and self.checkpoint_frequency > 0:
                if (epoch + 1) % self.checkpoint_frequency == 0:
                    self.save_full_checkpoint(epoch, avg_loss)
                    self._cleanup_old_checkpoints()

            # Save last model and full checkpoint (single file each, overwritten every epoch)
            if self.save_model:
                last_model_path = os.path.join(self.model_dir, "last_model.pt")
                torch.save(self.model.state_dict(), last_model_path)
                last_checkpoint_path = os.path.join(
                    self.model_dir, "last_checkpoint.pt"
                )
                self.save_full_checkpoint(epoch, avg_loss, path=last_checkpoint_path)
                if epoch == 0 or epoch == epochs - 1:
                    self.log.info(
                        f"Last model: {os.path.abspath(last_model_path)}, "
                        f"last checkpoint: {os.path.abspath(last_checkpoint_path)}"
                    )

        # Ensure tracker is flushed/closed cleanly
        if self.tracker is not None:
            self.tracker.finish()

    @staticmethod
    def get_norm(model: nn.Module, vector_type: str = "grad", p_norm: float = 2.0):
        """
        Get the norm of the model parameters or gradients.

        Parameters
        ----------
        model : nn.Module
            Model to get the norm of
        vector_type : str
            Type of vector to get the norm of ("weights" or "grad")
        p_norm : float
            The order of the norm (default: 2.0 for L2 norm).
            Use float('inf') for max norm.

        Returns
        -------
        torch.Tensor
            The calculated norm (scalar tensor).
        """
        device = next(model.parameters()).device
        total_norm = torch.zeros(1, dtype=torch.float32, device=device)

        for p in model.parameters():
            if vector_type == "weights":
                param_data = p.detach().float()
            elif vector_type == "grad":
                if p.grad is None:
                    continue
                param_data = p.grad.detach().float()
            else:
                raise ValueError(f"Unknown vector_type: {vector_type}")

            # Accumulate the p-th power of the norm
            if p_norm == float("inf"):
                total_norm = torch.max(total_norm, param_data.abs().max())
            else:
                total_norm += param_data.norm(p_norm).pow(p_norm)

        # Return the p-th root to get the actual norm
        if p_norm != float("inf"):
            total_norm = total_norm.pow(1.0 / p_norm)

        return total_norm
