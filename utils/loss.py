"""
Loss functions and related utilities for training.

This module provides loss functions used across all cases, as well as
case-specific loss functions (e.g., TVD norm for SOD_shock_tube).

Usage by case:
-------------
Cylinder/Cylinder_faster:
    - relative_l2_error: Common loss function

SOD_shock_tube:
    - relative_l2_error: Common loss function
    - TVD_norm: Total Variation Diminishing norm (SOD-specific)
    - tvd_weight_scheduler: See utils.case_specific (SOD-specific)
"""

import torch
import torch.nn.functional as F

# ============================================================================
# Constants
# ============================================================================

# Epsilon for numerical stability in relative error calculation
RELATIVE_ERROR_EPS = 1e-7

# Epsilon for TVD norm threshold
TVD_THRESHOLD = 1e-7


# ============================================================================
# Common Loss Functions
# ============================================================================


def l2_error(pred, target):
    """
    Calculate relative L2 error between prediction and target.

    Used by all cases (Cylinder, Cylinder_faster, SOD_shock_tube).

    Formula: ||pred - target||_2 / (||target||_2 + eps)

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values
    target : torch.Tensor
        Target/ground truth values

    Returns
    -------
    torch.Tensor
        Relative L2 error (scalar)
    """
    eps = RELATIVE_ERROR_EPS
    return torch.norm(pred - target) / (torch.norm(target) + eps)


# ============================================================================
# SOD-specific Loss Functions
# ============================================================================


def TVD_norm(U_new, U_old):
    """
    Compute the Total Variation Diminishing (TVD) norm of the difference between two fields.

    This function is specific to SOD_shock_tube case and is used to enforce
    TVD constraints during training.

    Parameters
    ----------
    U_new : torch.Tensor
        New field values (must have at least 2 dimensions)
    U_old : torch.Tensor
        Old field values (must have same shape as U_new)

    Returns
    -------
    torch.Tensor
        TVD norm value

    Raises
    ------
    ValueError
        If tensors have different shapes or less than 2 dimensions
    """
    if U_new.shape != U_old.shape:
        raise ValueError("Input tensors U_new and U_old must have the same shape.")

    if U_new.ndim < 2:
        raise ValueError("Input tensors must have at least 2 dimensions.")

    # Compute differences along the second dimension (spatial dimension)
    diff_new = U_new[2, 1:] - U_new[2, :-1]
    diff_old = U_old[2, 1:] - U_old[2, :-1]

    # Compute total variation
    TV_new = torch.abs(diff_new).sum()
    TV_old = torch.abs(diff_old).sum()

    # Compute TVD (Total Variation Diminishing) constraint
    TVD = F.relu(TV_new - TV_old) ** 2

    # Threshold small values to zero for numerical stability
    TVD = torch.where(TVD <= TVD_THRESHOLD, torch.tensor(0.0, device=TVD.device), TVD)

    return TVD
