"""
Core utility functions used across all cases.

This module provides basic utilities for device management, random seeding,
tensor operations, and model checkpoint loading.
"""

import torch
import numpy as np
import random
from typing import Dict, Any


def get_device():
    """
    Get the device to use for training.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_seed(seed=0):
    """
    Set random seeds for reproducibility.

    Sets seeds for PyTorch, NumPy, and Python's random module.
    Also sets CUDA seeds if CUDA is available.

    Parameters
    ----------
    seed : int, optional
        Random seed value (default: 0)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def detach(x):
    """
    Convert PyTorch tensor to NumPy array.

    Detaches tensor from computation graph, moves to CPU, and converts to NumPy.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor

    Returns
    -------
    numpy.ndarray
        NumPy array on CPU
    """
    return x.detach().cpu().numpy()


def adapt_checkpoint_keys(
    checkpoint: Dict[str, Any], model: torch.nn.Module
) -> Dict[str, Any]:
    """
    Adapt checkpoint keys to match model format (handles torch.compile prefix mismatch).

    When using torch.compile(), models get wrapped and their state_dict keys
    have an "_orig_mod." prefix. This function automatically handles the mismatch
    between checkpoint format and model format.

    Parameters
    ----------
    checkpoint : Dict[str, Any]
        Checkpoint dictionary (can be full checkpoint with "model_state_dict" key
        or just a state_dict)
    model : torch.nn.Module
        Model to load checkpoint into

    Returns
    -------
    Dict[str, Any]
        Transformed state_dict with keys matching the model's expected format

    Examples
    --------
    >>> # Load checkpoint and adapt keys
    >>> checkpoint = torch.load("model.pt")
    >>> state_dict = adapt_checkpoint_keys(checkpoint, model)
    >>> model.load_state_dict(state_dict)
    """
    # Extract state_dict from checkpoint (handle both full checkpoint and state_dict)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Determine if checkpoint has _orig_mod prefix
    checkpoint_has_prefix = any(k.startswith("_orig_mod.") for k in state_dict.keys())

    # Determine if model is compiled (check if model expects _orig_mod prefix)
    if len(model.state_dict()) == 0:
        # Empty model - can't determine, assume no prefix needed
        model_is_compiled = False
    else:
        model_sample_key = next(iter(model.state_dict().keys()))
        model_is_compiled = model_sample_key.startswith("_orig_mod.")

    # Transform checkpoint keys to match model format
    new_state_dict = {}
    for k, v in state_dict.items():
        if checkpoint_has_prefix and not model_is_compiled:
            # Checkpoint has prefix, model doesn't - remove prefix
            if k.startswith("_orig_mod."):
                new_k = k.replace("_orig_mod.", "")
                new_state_dict[new_k] = v
            else:
                new_state_dict[k] = v
        elif not checkpoint_has_prefix and model_is_compiled:
            # Checkpoint doesn't have prefix, model does - add prefix
            new_k = f"_orig_mod.{k}"
            new_state_dict[new_k] = v
        else:
            # Both have same format - use as-is
            new_state_dict[k] = v

    return new_state_dict
