"""
Core utility functions used across all cases.

This module provides basic utilities for device management, random seeding,
and tensor operations.
"""

import torch
import numpy as np
import random


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
