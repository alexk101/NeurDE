"""
Plotting utilities for visualization of simulation results.

This module provides case-specific plotting functions for visualizing
simulation results. Each case has different visualization needs:
- Cylinder: 2D field plots (Mach number)
- SOD_shock_tube: 1D line plots (density, temperature, velocity, pressure)

Usage by case:
-------------
Cylinder/Cylinder_faster:
    - plot_cylinder_results: 2D field visualization

SOD_shock_tube:
    - plot_sod_results: 1D line plot visualization
"""

import os
import matplotlib.pyplot as plt
from .core import detach

# ============================================================================
# Constants
# ============================================================================

# Default figure sizes
CYLINDER_FIGSIZE = (8, 4.5)
SOD_FIGSIZE = (16, 6)

# Default line width for plots
SOD_LINEWIDTH = 5

# Default font sizes
CYLINDER_TITLE_FONTSIZE = 16
SOD_TITLE_FONTSIZE = 25
SOD_SUBPLOT_TITLE_FONTSIZE = 18


# ============================================================================
# Cylinder Plotting Functions
# ============================================================================


def plot_cylinder_results(
    Ma_NN, Ma_GT, time_value, case_type="cylinder", output_dir=None
):
    """
    Plot and save Cylinder simulation results (2D field visualization).

    Creates a side-by-side 2D visualization comparing neural network prediction
    and ground truth local Mach number fields for the supersonic flow around
    a circular cylinder.

    Parameters
    ----------
    Ma_NN : array-like
        Neural network predicted local Mach number field
    Ma_GT : array-like
        Ground truth local Mach number field
    time_value : float or str
        Time value for the simulation snapshot
    case_type : str, optional
        Case type name (default: "cylinder")
    output_dir : str, optional
        Directory to save the plot. If None, saves to images/Cylinder/
        relative to the project root.

    Returns
    -------
    str
        Path to the saved image file
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot NN prediction
    im1 = ax1.imshow(Ma_NN, cmap="jet")
    ax1.set_title("Mach number - NN", fontsize=CYLINDER_TITLE_FONTSIZE)
    plt.colorbar(im1, ax=ax1)

    # Plot Ground Truth
    im2 = ax2.imshow(Ma_GT, cmap="jet")
    ax2.set_title("Mach number - GT", fontsize=CYLINDER_TITLE_FONTSIZE)
    plt.colorbar(im2, ax=ax2)

    fig.suptitle(
        f"{case_type.capitalize()} - Sample {time_value}",
        fontsize=CYLINDER_TITLE_FONTSIZE,
    )

    # Reduced whitespace
    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=0.35, w_pad=0.35)

    # Save to Images directory
    if output_dir is None:
        # Get project root (assuming this file is in utils/)
        current_file = os.path.abspath(__file__)
        main_dir = os.path.dirname(os.path.dirname(current_file))
        image_dir = os.path.join(main_dir, "images", case_type.capitalize())
    else:
        image_dir = output_dir

    os.makedirs(image_dir, exist_ok=True)
    output_path = os.path.join(image_dir, f"{case_type.capitalize()}_{time_value}.png")
    plt.savefig(output_path)
    plt.close(fig)

    return output_path


# ============================================================================
# SOD Shock Tube Plotting Functions
# ============================================================================


def plot_sod_results(
    rho_NN,
    ux_NN,
    T_NN,
    P_NN,
    rho_GT,
    ux_GT,
    T_GT,
    P_GT,
    time_step,
    case_number,
    output_dir=None,
):
    """
    Plot and save SOD shock tube simulation results (1D line plots).

    Creates a 4-panel plot comparing neural network predictions and ground truth
    for density, temperature, velocity in x, and pressure along the shock tube.

    Parameters
    ----------
    rho_NN : torch.Tensor or array-like
        Neural network predicted density values (2D array, uses middle row: rho_NN[2, :])
    ux_NN : torch.Tensor or array-like
        Neural network predicted x-velocity values (2D array, uses middle row: ux_NN[2, :])
    T_NN : torch.Tensor or array-like
        Neural network predicted temperature values (2D array, uses middle row: T_NN[2, :])
    P_NN : torch.Tensor or array-like
        Neural network predicted pressure values (2D array, uses middle row: P_NN[2, :])
    rho_GT : array-like
        Ground truth density values (2D array, uses middle row: rho_GT[2, :])
    ux_GT : array-like
        Ground truth x-velocity values (2D array, uses middle row: ux_GT[2, :])
    T_GT : array-like
        Ground truth temperature values (2D array, uses middle row: T_GT[2, :])
    P_GT : array-like
        Ground truth pressure values (2D array, uses middle row: P_GT[2, :])
    time_step : int or str
        Time step number for the simulation snapshot
    case_number : int
        SOD shock tube case number
    output_dir : str, optional
        Directory to save the plot. If None, saves to images/SOD_case{case_number}/
        relative to the project root.

    Returns
    -------
    str
        Path to the saved image file
    """
    plt.figure(figsize=SOD_FIGSIZE)

    # Larger title and reduced whitespace
    plt.suptitle(
        f"SOD shock case {case_number} time {time_step}",
        fontweight="bold",
        fontsize=SOD_TITLE_FONTSIZE,
        y=0.95,
    )

    # Plot density (NN: thick line, GT: thin line)
    plt.subplot(221)
    plt.plot(detach(rho_NN[2, :]), linewidth=SOD_LINEWIDTH, label="NN")
    plt.plot(rho_GT[2, :], linewidth=2, label="GT")
    plt.title("Density", fontsize=SOD_SUBPLOT_TITLE_FONTSIZE)

    # Plot temperature
    plt.subplot(222)
    plt.plot(detach(T_NN[2, :]), linewidth=SOD_LINEWIDTH)
    plt.plot(T_GT[2, :], linewidth=2)
    plt.title("Temperature", fontsize=SOD_SUBPLOT_TITLE_FONTSIZE)

    # Plot velocity in x
    plt.subplot(223)
    plt.plot(detach(ux_NN[2, :]), linewidth=SOD_LINEWIDTH)
    plt.plot(ux_GT[2, :], linewidth=2)
    plt.title("Velocity in x", fontsize=SOD_SUBPLOT_TITLE_FONTSIZE)

    # Plot pressure
    plt.subplot(224)
    plt.plot(detach(P_NN[2, :]), linewidth=SOD_LINEWIDTH)
    plt.plot(P_GT[2, :], linewidth=2)
    plt.title("Pressure", fontsize=SOD_SUBPLOT_TITLE_FONTSIZE)

    # Reduced whitespace
    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=0.35, w_pad=0.35)

    # Save to Images directory
    if output_dir is None:
        # Get project root (assuming this file is in utils/)
        current_file = os.path.abspath(__file__)
        main_dir = os.path.dirname(os.path.dirname(current_file))
        image_dir = os.path.join(main_dir, "images", f"SOD_case{case_number}")
    else:
        image_dir = output_dir

    os.makedirs(image_dir, exist_ok=True)
    output_path = os.path.join(image_dir, f"SOD_case{case_number}_{time_step}.png")
    plt.savefig(output_path)
    plt.close()

    return output_path
