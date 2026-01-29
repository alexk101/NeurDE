"""
Adaptive Gradient Clipping Module

This module provides adaptive gradient clipping functionality that extends the existing
custom gradient clipping with ZClip-style adaptive thresholds based on gradient norm statistics.
It's designed to work with custom distributed training setups that have complex parallelism
patterns (TP, SP, DP) and custom gradient synchronization.
Reference: https://raw.githubusercontent.com/bluorion-com/ZClip/refs/heads/main/zclip/zclip.py
"""

import math
import torch


def _to_float(value):
    """Convert tensor to float, or return float value as-is."""
    if torch.is_tensor(value):
        return float(value.item())
    return float(value)


class AdaptiveGradientClipper:
    """
    Adaptive gradient clipping that extends the existing custom gradient clipping
    with ZClip-style adaptive thresholds based on gradient norm statistics.
    
    This implementation is designed to work with custom distributed training setups
    that have complex parallelism patterns and custom gradient synchronization.
    """
    
    def __init__(self, cfg):
        """
        Initialize adaptive gradient clipper.
        
        Args:
            alpha: EMA decay factor for gradient norm statistics
            z_thresh: Z-score threshold for triggering adaptive clipping
            max_grad_norm: Optional maximum gradient norm (fallback threshold)
            eps: Small constant to avoid division by zero
            warmup_steps: Number of steps to collect gradient norms before EMA initialization
            mode: Clipping mode ("zscore" or "percentile")
            clip_option: Only used when mode is "zscore" ("adaptive_scaling" or "mean")
            clip_factor: Multiplier for the adaptive scaling threshold
            skip_update_on_spike: If True, skip updating EMA statistics when a spike is detected
            enabled: Whether adaptive clipping is enabled
        """
        self.alpha = cfg.get('alpha', 0.97)
        self.z_thresh = cfg.get('z_thresh', 2.5)
        self.max_grad_norm = cfg.get('max_grad_norm', None)
        self.eps = cfg.get('eps', 1e-8)
        self.warmup_steps = cfg.get('warmup_steps', 100)
        self.mode = cfg.get('mode', "zscore").lower()
        self.clip_option = cfg.get('clip_option', "adaptive_scaling").lower() if cfg.get('clip_option', None) else None
        self.clip_factor = cfg.get('clip_factor', 1.0)
        self.skip_update_on_spike = cfg.get('skip_update_on_spike', False)
        self.enabled = cfg.get('enabled', True)
        
        # Validate parameters
        if self.mode == "zscore":
            assert self.clip_option in ["mean", "adaptive_scaling"], (
                "For zscore mode, clip_option must be either 'mean' or 'adaptive_scaling'."
            )
        elif self.mode == "percentile":
            self.clip_option = None
        else:
            raise ValueError("mode must be either 'zscore' or 'percentile'.")
        
        # Initialize state
        self.buffer = []
        self.initialized = False
        self.mean = None
        self.var = None
        self.clips_per_log_interval = 0
        self._last_stats_log_iter = None
        
    def reset_log_interval_stats(self):
        """Reset per-log-interval statistics (e.g., clip count).
        
        This should be called when starting a new logging interval (e.g., every log_every iterations).
        """
        self.clips_per_log_interval = 0
        
    def _initialize_ema(self):
        """Initialize EMA statistics from the buffer."""
        # Filter out any non-finite values that may have slipped through
        valid_buffer = [x for x in self.buffer if math.isfinite(x)]
        if len(valid_buffer) == 0:
            # Fallback: use a reasonable default if all values were invalid
            self.mean = 1.0
            self.var = 0.01
            self.initialized = True
            self.buffer = []
            return
            
        self.mean = sum(valid_buffer) / len(valid_buffer)
        self.var = max(sum((x - self.mean) ** 2 for x in valid_buffer) / len(valid_buffer), self.eps)
        self.initialized = True
        self.buffer = []
        
    def _update_ema(self, grad_norm):
        """Update EMA for mean and variance using the new effective gradient norm."""
        # Convert grad_norm to float if it's a tensor to ensure mean/var stay as Python floats
        if torch.is_tensor(grad_norm):
            grad_norm = float(grad_norm.item())
        
        # Store old mean before updating (needed for correct variance update)
        old_mean = self.mean
        # Update mean
        self.mean = self.alpha * self.mean + (1 - self.alpha) * grad_norm
        # Update variance using the OLD mean (this is the correct formula for EMA variance)
        # Using the new mean would be incorrect and cause numerical issues
        self.var = self.alpha * self.var + (1 - self.alpha) * (grad_norm - old_mean) ** 2
        # Ensure variance never becomes negative or too small
        self.var = max(self.var, self.eps)
        # Ensure mean and var remain finite
        if not math.isfinite(self.mean):
            self.mean = old_mean  # Revert if update produced non-finite
        if not math.isfinite(self.var):
            self.var = max(self.var if math.isfinite(self.var) else self.eps, self.eps)
        
    def _compute_positive_zscore(self, grad_norm):
        """Compute the positive z-score for the current gradient norm."""
        # Convert grad_norm to float if it's a tensor
        if torch.is_tensor(grad_norm):
            grad_norm = float(grad_norm.item())
        
        # Ensure stats are finite before computing
        if not (math.isfinite(self.mean) and math.isfinite(self.var) and self.var >= 0):
            # Fallback to safe defaults
            return 0.0, math.sqrt(self.eps)
        std = math.sqrt(self.var)
        z = (grad_norm - self.mean) / (std + self.eps)
        # Ensure z-score is finite
        if not math.isfinite(z):
            z = 0.0
        return z, std
        
    def _compute_clip_val(self, grad_norm):
        """Compute the clipping threshold based on the selected mode and clip_option."""
        # Convert grad_norm to float if it's a tensor
        if torch.is_tensor(grad_norm):
            grad_norm = float(grad_norm.item())
        
        # Validate that stats are finite before using them
        if not (math.isfinite(self.mean) and math.isfinite(self.var) and self.var >= 0):
            # If stats are corrupted, fall back to max_grad_norm or return None
            return self.max_grad_norm if self.max_grad_norm is not None else None
            
        std = math.sqrt(self.var)
        
        if self.mode == "percentile":
            # Always clip to a threshold computed as: EMA mean + (z_thresh × std)
            threshold = self.mean + self.z_thresh * std
            if grad_norm > threshold:
                return threshold
        elif self.mode == "zscore":
            # Compute the z-score for the current gradient norm
            z, std = self._compute_positive_zscore(grad_norm)
            if z > self.z_thresh:
                if self.clip_option == "adaptive_scaling":
                    eta = z / self.z_thresh  # This rescaling ratio imposes a greater penalty on large outliers
                    threshold = self.mean + (self.z_thresh * std) / eta
                    threshold = threshold * self.clip_factor
                elif self.clip_option == "mean":
                    threshold = self.mean
                return threshold
                
        return None  # No clipping needed
        
    def _apply_clipping(self, model, clip_val, total_norm):
        """Apply clipping to the gradients by merging the computed clip value with the optional max_grad_norm."""
        # Convert total_norm to float for math operations
        total_norm_float = _to_float(total_norm)
        
        # Use the computed clip_val if available; otherwise, use the total norm
        adaptive_clip = clip_val if clip_val is not None else total_norm_float
        
        # Ensure adaptive_clip is finite
        if not math.isfinite(adaptive_clip) or adaptive_clip <= 0:
            # Fall back to max_grad_norm or total_norm if adaptive_clip is invalid
            if self.max_grad_norm is not None and math.isfinite(self.max_grad_norm):
                adaptive_clip = self.max_grad_norm
            else:
                adaptive_clip = total_norm_float if math.isfinite(total_norm_float) and total_norm_float > 0 else 1.0
        
        if self.max_grad_norm is not None:
            effective_clip = min(adaptive_clip, self.max_grad_norm)
        else:
            effective_clip = adaptive_clip
        
        # Ensure effective_clip is finite and positive
        if not (math.isfinite(effective_clip) and effective_clip > 0):
            effective_clip = total_norm_float if math.isfinite(total_norm_float) and total_norm_float > 0 else 1.0
            
        # Apply clipping in-place (use tensor version for comparison)
        if total_norm_float > effective_clip and math.isfinite(total_norm_float):
            clip_coef = effective_clip / (total_norm_float + self.eps)
            # Ensure clip_coef is finite and convert to scalar
            if math.isfinite(clip_coef):
                # Convert to scalar if it's a tensor
                if torch.is_tensor(clip_coef):
                    clip_coef = clip_coef.item()
                clip_coef = float(clip_coef)
                # Collect all gradients that need updating
                grads = []
                for param in model.parameters():
                    if param.grad is not None:
                        grads.append(param.grad)
                
                # Use vectorized operation if available (PyTorch 1.10+)
                if grads:
                    torch._foreach_mul_(grads, clip_coef)
                    
        return effective_clip
        
    def step(self, trainer, model):
        """
        Apply adaptive gradient clipping using the trainer's custom get_norm method.
        
        Args:
            trainer: Trainer instance with get_norm method
            model: Model to clip gradients for
            
        Returns:
            float: The total gradient norm (before clipping) for monitoring
        """
        if not self.enabled:
            # Fallback to static clipping if adaptive clipping is disabled
            if self.max_grad_norm is not None:
                total_norm = trainer.get_norm(model, vector_type="grad")
                total_norm_float = _to_float(total_norm)
                if total_norm_float > self.max_grad_norm:
                    clip_coef = self.max_grad_norm / (total_norm_float + self.eps)
                    # Ensure clip_coef is a scalar
                    clip_coef = float(clip_coef)
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad.data.mul_(clip_coef)
                    self.clips_per_log_interval += 1
                    trainer.log.info(f"Static clipping: norm={float(total_norm):.4f}, threshold={self.max_grad_norm}")
                return total_norm
            else:
                # No clipping - just compute and return norm for consistency
                total_norm = trainer.get_norm(model, vector_type="grad")
                return total_norm
        
        # Use trainer's existing get_norm method which handles distributed correctly
        total_norm = trainer.get_norm(model, vector_type="grad")
        
        # Convert to float for math operations and checks (keep original for tensor operations)
        total_norm_float = _to_float(total_norm)

        # If norm is non-finite on this rank, zero gradients and skip stats/clip
        if not math.isfinite(total_norm_float):
            trainer.log.warning(
                f"Non-finite gradient norm detected on rank {trainer.world_rank}: {total_norm}. "
                f"Zeroing grads and skipping clip/EMA update."
            )
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach().zero_()
            return total_norm
        
        # During warmup, collect gradient norms without applying clipping
        # BUT: Filter out non-finite values to prevent corrupting initialization
        if not self.initialized:
            # Only add finite norms to buffer
            if math.isfinite(total_norm_float) and total_norm_float > 0:
                self.buffer.append(total_norm_float)
            else:
                trainer.log.warning(
                    f"Skipping non-finite norm {total_norm} during warmup on rank {trainer.world_rank}"
                )
            
            if len(self.buffer) >= self.warmup_steps:
                self._initialize_ema()
                trainer.log.info(f"Adaptive clipper initialized after {self.warmup_steps} steps. "
                                 f"Initial mean={float(self.mean):.4f}, var={float(self.var):.4f}")
            # Even if we filtered the norm, return it for consistency
            return total_norm
            
        # If we're still in warmup, don't apply clipping
        if not self.initialized:
            return total_norm
            
        # Apply max_grad_norm clipping if specified (fallback behavior)
        if self.max_grad_norm is not None:
            if total_norm_float > self.max_grad_norm:
                clip_coef = self.max_grad_norm / (total_norm_float + self.eps)
                # Ensure clip_coef is a scalar
                clip_coef = float(clip_coef)
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data.mul_(clip_coef)
                self.clips_per_log_interval += 1
                trainer.log.info(f"Max norm clipping: norm={float(total_norm):.4f}, threshold={self.max_grad_norm}")
                return total_norm
                
        # Compute the clip value based on the selected mode and clip_option
        clip_val = self._compute_clip_val(total_norm)
        
        # Apply adaptive clipping
        effective_clip = self._apply_clipping(model, clip_val, total_norm)
        
        # Enhanced logging with gradient statistics
        z_score, std = self._compute_positive_zscore(total_norm)
        
        if clip_val is not None:
            # Adaptive clipping was applied
            self.clips_per_log_interval += 1
            trainer.log.info(f"Adaptive clipping: norm={float(total_norm):.4f}, z_score={z_score:.2f}, "
                            f"threshold={float(effective_clip):.4f}, ema_mean={float(self.mean):.4f}, ema_std={std:.4f}")
        else:
            # No clipping needed, but still log statistics (at most once per log interval)
            current_iter = getattr(trainer, "iters", None)
            should_log = False
            
            if current_iter is not None:
                # Check if we should log based on iteration
                log_every = getattr(trainer, "log_every", None)
                if log_every is not None:
                    # Log at the start of each log interval
                    should_log = (current_iter % log_every == 1) or (current_iter == 1)
                else:
                    # Log once per iteration if no log_every is set
                    should_log = current_iter != self._last_stats_log_iter
                    if should_log:
                        self._last_stats_log_iter = current_iter
            
            if should_log:
                trainer.log.info(
                    f"Gradient stats: norm={float(total_norm):.4f}, z_score={z_score:.2f}, "
                    f"ema_mean={float(self.mean):.4f}, ema_std={std:.4f}"
                )
                           
        # Update EMA with the effective norm (either the computed clip or the original norm)
        # Only update if the value to add is finite
        if not (clip_val is not None and self.skip_update_on_spike):
            ema_update_value = clip_val if clip_val is not None else total_norm
            # Only update EMA if the value is finite
            if math.isfinite(ema_update_value):
                self._update_ema(ema_update_value)
            else:
                trainer.log.warning(f"Skipping EMA update with non-finite value: {ema_update_value}")
            
        return total_norm
        
    def get_stats(self):
        """Get current EMA statistics for monitoring."""
        return {
            'ema_mean': self.mean,
            'ema_var': self.var,
            'step_count': len(self.buffer) + (self.warmup_steps if self.initialized else 0),
            'initialized': self.initialized,
            'mode': self.mode,
            'clip_option': self.clip_option
        }

    def stats_for_logging(self, warn: bool = False, logger=None):
        """
        Return a dict of logging-ready statistics.

        - Clip count is always logged (adaptive and fixed max_grad_norm).
        - EMA mean/std are only logged when adaptive clipping is enabled and initialized.

        Args:
            warn: If True, emit warnings when stats are invalid
            logger: Optional logger to emit warnings to
        """
        out = {"grad_clip_count_per_log_interval": self.clips_per_log_interval}

        if not self.enabled:
            # Fixed clipping only: log clip count, no adaptive stats
            return out

        if (
            self.initialized
            and self.mean is not None
            and self.var is not None
            and math.isfinite(self.mean)
            and math.isfinite(self.var)
            and self.var >= 0
        ):
            ema_mean = float(self.mean)
            ema_std = float(max(self.var, self.eps) ** 0.5)
            if math.isfinite(ema_mean) and math.isfinite(ema_std):
                out["grad_clip_ema_mean"] = ema_mean
                out["grad_clip_ema_std"] = ema_std
            elif warn and logger is not None:
                logger.warning(
                    "EMA stats became non-finite during computation; logging clip count only"
                )
        elif warn and logger is not None and self.enabled:
            logger.warning(
                "Adaptive clipper not initialized or stats invalid; logging clip count only"
            )

        return out
    
    def state_dict(self):
        """
        Return a dictionary containing the full state of the adaptive gradient clipper.
        This is used for checkpointing to enable proper resumption of training.
        
        Returns:
            dict: State dictionary containing all necessary state for resumption
        """
        return {
            'buffer': self.buffer.copy(),  # Copy to avoid mutation issues
            'initialized': self.initialized,
            'mean': self.mean,
            'var': self.var,
            'clips_per_log_interval': self.clips_per_log_interval,
            # Configuration parameters (useful for validation on load)
            'alpha': self.alpha,
            'z_thresh': self.z_thresh,
            'max_grad_norm': self.max_grad_norm,
            'eps': self.eps,
            'warmup_steps': self.warmup_steps,
            'mode': self.mode,
            'clip_option': self.clip_option,
            'clip_factor': self.clip_factor,
            'skip_update_on_spike': self.skip_update_on_spike,
            'enabled': self.enabled,
        }
    
    def load_state_dict(self, state_dict, strict=True):
        """
        Load state from a state dictionary, typically from a checkpoint.
        
        Args:
            state_dict: Dictionary containing the clipper state
            strict: If True, raise an error if configuration parameters don't match.
                   If False, only load runtime state and ignore config mismatches.
        
        Returns:
            tuple: (missing_keys, unexpected_keys) lists for compatibility reporting
        """
        missing_keys = []
        unexpected_keys = []
        
        # Define expected keys
        runtime_keys = {'buffer', 'initialized', 'mean', 'var', 'clips_per_log_interval'}
        config_keys = {'alpha', 'z_thresh', 'max_grad_norm', 'eps', 'warmup_steps',
                       'mode', 'clip_option', 'clip_factor', 'skip_update_on_spike', 'enabled'}
        expected_keys = runtime_keys | config_keys
        
        # Check for unexpected keys
        for key in state_dict:
            if key not in expected_keys:
                unexpected_keys.append(key)
        
        # Check for missing runtime keys (config keys are optional)
        required_runtime_keys = {'buffer', 'initialized', 'mean', 'var', 'clips_per_log_interval'}
        for key in required_runtime_keys:
            if key not in state_dict:
                missing_keys.append(key)
        
        if missing_keys:
            raise KeyError(f"Missing required keys in state_dict: {missing_keys}")
        
        # Validate configuration parameters if strict mode
        if strict:
            config_mismatches = []
            for key in config_keys:
                if key in state_dict:
                    current_val = getattr(self, key)
                    saved_val = state_dict[key]
                    if current_val != saved_val:
                        config_mismatches.append(f"{key}: current={current_val}, saved={saved_val}")
            
            if config_mismatches:
                raise ValueError(
                    f"Adaptive gradient clipper configuration mismatch when loading checkpoint. "
                    f"Mismatches: {config_mismatches}. "
                    f"Use strict=False to ignore config differences and only load runtime state."
                )
        
        # Load runtime state
        self.buffer = state_dict['buffer'].copy() if isinstance(state_dict['buffer'], list) else list(state_dict['buffer'])
        self.initialized = state_dict['initialized']
        self.mean = state_dict['mean']
        self.var = state_dict['var']
        self.clips_per_log_interval = state_dict['clips_per_log_interval']
        
        return missing_keys, unexpected_keys