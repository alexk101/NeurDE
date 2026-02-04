"""
Experiment tracking interface for NeurDE.

This module is the public API: it delegates to tracking backends (W&B, MLflow,
etc.) so callers can use a single import while the implementation is swappable.

Usage:
------
    from utils.experiment_tracking import create_tracker, ExperimentTracker

    tracker = create_tracker(cfg, logger, run_id=None)
    tracker.log_metrics({"loss": 0.5}, step=100, step_metric="batch")
    tracker.log_figure(fig, "validation_plot", step=1, step_metric="epoch")
    tracker.finish()

Backends:
---------
- "wandb": Weights & Biases (requires `pip install wandb`)
- "mlflow": MLflow (requires `pip install mlflow`)
- "none": NullTracker (no-op, used when tracking is disabled)

Configuration:
--------------
The backend is selected via `cfg.logging.backend`. Interactive plots can be
enabled via `cfg.logging.tracker.interactive_plots` (default: False).
"""

from __future__ import annotations

# Re-export the public API from the trackers module
from .trackers import (
    ExperimentTracker,
    NullTracker,
    FigureFormat,
    get_tracker,
    create_tracker,
)

__all__ = [
    "ExperimentTracker",
    "NullTracker",
    "FigureFormat",
    "get_tracker",
    "create_tracker",
]
