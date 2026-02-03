"""
Experiment tracking abstraction for NeurDE.

Provides a small, generic interface that can be backed by different
experiment-tracking systems (e.g., Weights & Biases, MLflow), while
allowing the training code to remain agnostic to the specific backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional
import os
import logging

from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from PIL import Image


log = logging.getLogger(__name__)


class ExperimentTracker(ABC):
    """Abstract experiment-tracking interface."""

    @abstractmethod
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        step_metric: Optional[str] = None,
    ) -> None:
        """Log scalar metrics.

        step_metric: When provided ("batch" or "epoch"), backends can use it so
        metrics are plotted on the correct time scale. WandB uses define_metric
        so per-batch and per-epoch metrics get separate x-axes; MLflow uses step
        as a single timeline (caller should pass global_step for epoch-level so
        epoch and batch metrics align).
        """

    @abstractmethod
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters / configuration."""

    def log_image(
        self,
        image: "Image.Image",
        key: str,
        step: Optional[int] = None,
        step_metric: Optional[str] = None,
    ) -> None:
        """Log a PIL image (e.g. validation plot). Default is no-op; backends override."""

    @abstractmethod
    def finish(self) -> None:
        """Clean up any resources before exiting."""

    def get_run_id(self) -> Optional[str]:
        """Return the current run ID if the backend supports resuming (e.g. MLflow)."""
        return None


class NullTracker(ExperimentTracker):
    """No-op tracker used when experiment tracking is disabled."""

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        step_metric: Optional[str] = None,
    ) -> None:  # type: ignore[override]
        return

    def log_hyperparams(self, params: Dict[str, Any]) -> None:  # type: ignore[override]
        return

    def log_image(
        self,
        image: "Image.Image",
        key: str,
        step: Optional[int] = None,
        step_metric: Optional[str] = None,
    ) -> None:  # type: ignore[override]
        return

    def finish(self) -> None:  # type: ignore[override]
        return


class WandbTracker(ExperimentTracker):
    """Weights & Biases implementation of the experiment tracker."""

    def __init__(self, cfg: DictConfig, logging_cfg: DictConfig):
        try:
            import wandb  # type: ignore
        except Exception as e:  # pragma: no cover - import-time failure path
            log.warning(f"Could not import wandb ({e}); falling back to NullTracker.")
            raise

        tracker_cfg = getattr(logging_cfg, "tracker", {})

        project = tracker_cfg.get("project", None)
        entity = tracker_cfg.get("entity", None)
        run_name = tracker_cfg.get("run_name", None)
        run_tag = tracker_cfg.get("run_tag", None)
        extra_kwargs = tracker_cfg.get("extra_init_kwargs", {}) or {}

        # Flatten the Hydra config into a plain dict for logging
        config_dict = OmegaConf.to_container(cfg, resolve=True)

        tags = []
        if run_tag is not None:
            tags.append(str(run_tag))

        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=config_dict,
            tags=tags if tags else None,
            **extra_kwargs,
        )

        # Register custom step metrics so per-batch and per-epoch metrics can
        # use different x-axes (see docs.wandb.ai/guides/track/log/customize-logging-axes)
        self._run.define_metric("batch")
        self._run.define_metric("epoch")

        self._defined_for_step: set = set()  # (key, step_metric) for lazy define_metric

        log.info(
            f"Initialized Weights & Biases run: project={project}, "
            f"entity={entity}, name={run_name}, tags={tags}"
        )

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        step_metric: Optional[str] = None,
    ) -> None:  # type: ignore[override]
        if step_metric in ("batch", "epoch"):
            # Log with custom x-axis so batch-level and epoch-level metrics
            # don't share the same step scale (WandB would otherwise mix them).
            to_log = dict(metrics)
            if step_metric == "batch" and step is not None:
                to_log["batch"] = step
            # "epoch" is already in metrics from build_epoch_log_metrics
            for key in to_log:
                if key in ("batch", "epoch"):
                    continue
                tag = (key, step_metric)
                if tag not in self._defined_for_step:
                    self._run.define_metric(key, step_metric=step_metric)
                    self._defined_for_step.add(tag)
            self._wandb.log(to_log)
        else:
            if step is not None:
                self._wandb.log(metrics, step=step)
            else:
                self._wandb.log(metrics)

    def log_hyperparams(self, params: Dict[str, Any]) -> None:  # type: ignore[override]
        # For wandb, hyperparams are typically passed via config at init time,
        # but we allow additional updates here.
        if self._run is not None:
            for k, v in params.items():
                self._run.config[k] = v

    def log_image(
        self,
        image: "Image.Image",
        key: str,
        step: Optional[int] = None,
        step_metric: Optional[str] = None,
    ) -> None:  # type: ignore[override]
        to_log: Dict[str, Any] = {key: self._wandb.Image(image)}
        if step_metric in ("batch", "epoch") and step is not None:
            to_log[step_metric] = step
            if (key, step_metric) not in self._defined_for_step:
                self._run.define_metric(key, step_metric=step_metric)
                self._defined_for_step.add((key, step_metric))
        self._wandb.log(to_log, step=step if step_metric is None else None)

    def finish(self) -> None:  # type: ignore[override]
        if self._run is not None:
            self._run.finish()


class MlflowTracker(ExperimentTracker):
    """MLflow implementation of the experiment tracker.

    Supports local (sqlite/file) and remote tracking via tracking_uri.
    See https://mlflow.org/docs/latest/self-hosting/ for server setup.
    When run_id is provided (e.g. from a checkpoint when resuming), continues
    logging to that existing run instead of creating a new one.
    """

    def __init__(
        self, cfg: DictConfig, logging_cfg: DictConfig, run_id: Optional[str] = None
    ):
        try:
            import mlflow
        except Exception as e:
            log.warning(f"Could not import mlflow ({e}); falling back to NullTracker.")
            raise

        mlflow.enable_system_metrics_logging()
        tracker_cfg = getattr(logging_cfg, "tracker", {})

        # --- HYBRID CONFIGURATION HANDLING ---

        # 1. TLS Verification (Config controls Env)
        # If using self-signed certs via Caddy, we must tell MLflow to ignore verification
        if tracker_cfg.get("insecure_tls", False):
            os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"

        # 2. Authentication (Config + Env)
        # We take the username from Config, but rely on the environment for the password
        username = tracker_cfg.get("username", None)
        if username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = username

        # Optional: Warn if password is missing when username is present
        if username and "MLFLOW_TRACKING_PASSWORD" not in os.environ:
            log.warning(
                "MLflow username set in config, but MLFLOW_TRACKING_PASSWORD not found in environment!"
            )

        # 3. Tracking URI
        tracking_uri = tracker_cfg.get("tracking_uri", None)
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
            log.info(f"MLflow tracking URI set to: {tracking_uri}")

        project = tracker_cfg.get("project", None)
        run_name = tracker_cfg.get("run_name", None)
        run_tag = tracker_cfg.get("run_tag", None)
        # Allow run_id from config (e.g. logging.tracker.run_id) for manual resume
        resume_run_id = run_id or tracker_cfg.get("run_id", None)

        experiment_name = project or "default"
        mlflow.set_experiment(experiment_name)

        self._mlflow = mlflow
        if resume_run_id is not None:
            self._run = mlflow.start_run(run_id=resume_run_id)
            log.info(f"Resumed MLflow run: run_id={resume_run_id}")
        else:
            self._run = mlflow.start_run(run_name=run_name)

        if run_tag is not None:
            mlflow.set_tag("run_tag", str(run_tag))

        # Log full config as parameters (flattened)
        config_dict = OmegaConf.to_container(cfg, resolve=True)

        def _flatten(prefix: str, d: Dict[str, Any], out: Dict[str, Any]) -> None:
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                if isinstance(v, dict):
                    _flatten(key, v, out)
                else:
                    out[key] = v

        flat_params: Dict[str, Any] = {}
        if isinstance(config_dict, dict):
            _flatten("", config_dict, flat_params)

        for k, v in flat_params.items():
            try:
                mlflow.log_param(k, v)
            except Exception:
                # Skip values that MLflow cannot serialize as params
                continue

        log.info(
            f"Initialized MLflow run: experiment={experiment_name}, name={run_name}, "
            f"tracking_uri={mlflow.get_tracking_uri()}"
        )

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        step_metric: Optional[str] = None,
    ) -> None:  # type: ignore[override]
        # MLflow uses a single step timeline; step_metric is ignored. Caller should
        # pass step=global_step for both batch- and epoch-level so they align.
        for k, v in metrics.items():
            try:
                if step is not None:
                    self._mlflow.log_metric(k, v, step=step)
                else:
                    self._mlflow.log_metric(k, v)
            except Exception:
                # Skip metrics that cannot be logged
                continue

    def log_hyperparams(self, params: Dict[str, Any]) -> None:  # type: ignore[override]
        for k, v in params.items():
            try:
                self._mlflow.log_param(k, v)
            except Exception:
                continue

    def log_image(
        self,
        image: "Image.Image",
        key: str,
        step: Optional[int] = None,
        step_metric: Optional[str] = None,
    ) -> None:  # type: ignore[override]
        self._mlflow.log_image(image, key=key, step=step)

    def finish(self) -> None:  # type: ignore[override]
        self._mlflow.end_run()

    def get_run_id(self) -> Optional[str]:  # type: ignore[override]
        if self._run is not None:
            return self._run.info.run_id
        return None


def create_experiment_tracker(
    cfg: DictConfig, logger: logging.Logger, run_id: Optional[str] = None
) -> ExperimentTracker:
    """
    Factory that creates an appropriate ExperimentTracker instance based on config.

    If tracking is disabled or configuration is missing, returns a NullTracker.
    When resuming from a checkpoint, pass run_id (e.g. from checkpoint metadata) so
    MLflow continues logging to the same run instead of creating a new one.
    """
    logging_cfg = getattr(cfg, "logging", None)
    if logging_cfg is None:
        return NullTracker()

    backend = logging_cfg.get("backend", "none")
    tracker_cfg = getattr(logging_cfg, "tracker", None)
    enabled = (
        getattr(tracker_cfg, "enabled", False) if tracker_cfg is not None else False
    )

    if not enabled or backend in (None, "none"):
        logger.info(
            "Experiment tracking disabled (backend=none or tracker.enabled=false)."
        )
        return NullTracker()

    try:
        if backend == "wandb":
            return WandbTracker(cfg, logging_cfg)
        elif backend == "mlflow":
            return MlflowTracker(cfg, logging_cfg, run_id=run_id)
        else:
            logger.warning(
                f"Unknown experiment tracking backend '{backend}', disabling tracking."
            )
            return NullTracker()
    except Exception:
        # Any failure during backend initialization should not break training;
        # we log a warning and fall back to a no-op tracker.
        logger.warning(
            f"Failed to initialize experiment tracker backend '{backend}'. Using NullTracker instead."
        )
        return NullTracker()
