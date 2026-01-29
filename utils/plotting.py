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

For experiment tracking (WandB, MLflow), use fig_to_image() to convert a
matplotlib figure to a PIL Image with configurable DPI.
"""

import io
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from .core import detach

# ============================================================================
# Constants
# ============================================================================

# Default DPI when saving figures (controls resolution for saved files and logged images)
FIG_DPI = 200

# Default figure sizes (cylinder: NN, GT, error; SOD: row1 fields, row2 errors)
CYLINDER_FIGSIZE = (14, 5)
SOD_FIGSIZE = (16, 10)

# Default line width for plots
SOD_LINEWIDTH = 5

# Default font sizes
CYLINDER_TITLE_FONTSIZE = 16
SOD_TITLE_FONTSIZE = 25
SOD_SUBPLOT_TITLE_FONTSIZE = 18


# ============================================================================
# Figure-to-image conversion (for logging with controlled DPI)
# ============================================================================


def fig_to_image(fig, dpi: int = FIG_DPI) -> Image.Image:
    """
    Convert a matplotlib figure to a PIL Image with configurable DPI.

    Use this when logging plots to experiment trackers (WandB, MLflow) so you
    control output resolution.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to convert
    dpi : int, optional
        Dots per inch for the output image (default: FIG_DPI)

    Returns
    -------
    PIL.Image.Image
        PIL Image (e.g. for wandb.Image(img), mlflow.log_image, etc.)
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    img.load()
    buf.close()
    return img


# ============================================================================
# Cylinder Plotting Functions
# ============================================================================


def plot_cylinder_results(
    Ma_NN, Ma_GT, time_value, case_type="cylinder", output_dir=None, save=True
):
    """
    Plot Cylinder simulation results (2D field visualization).

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
        Directory to save the plot. If None (and save=True), saves to
        images/Cylinder/ relative to the project root.
    save : bool, optional
        If True, save to disk and close the figure (default). If False,
        return the figure without saving (e.g. for fig_to_image then close).

    Returns
    -------
    str or matplotlib.figure.Figure
        If save=True, path to the saved image file. If save=False, the figure.
    """
    Ma_NN = np.asarray(Ma_NN)
    Ma_GT = np.asarray(Ma_GT)
    Ma_err = np.abs(Ma_NN - Ma_GT)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=CYLINDER_FIGSIZE)

    # Plot NN prediction
    im1 = ax1.imshow(Ma_NN, cmap="jet")
    ax1.set_title("Mach number - NN", fontsize=CYLINDER_TITLE_FONTSIZE)
    plt.colorbar(im1, ax=ax1)

    # Plot Ground Truth
    im2 = ax2.imshow(Ma_GT, cmap="jet")
    ax2.set_title("Mach number - GT", fontsize=CYLINDER_TITLE_FONTSIZE)
    plt.colorbar(im2, ax=ax2)

    # Plot absolute error
    im3 = ax3.imshow(Ma_err, cmap="hot")
    ax3.set_title("|NN − GT|", fontsize=CYLINDER_TITLE_FONTSIZE)
    plt.colorbar(im3, ax=ax3)

    fig.suptitle(
        f"{case_type.capitalize()} - Sample {time_value}",
        fontsize=CYLINDER_TITLE_FONTSIZE,
    )

    # Reduced whitespace
    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=0.35, w_pad=0.35)

    if not save:
        return fig

    # Save to Images directory
    if output_dir is None:
        current_file = os.path.abspath(__file__)
        main_dir = os.path.dirname(os.path.dirname(current_file))
        image_dir = os.path.join(main_dir, "images", case_type.capitalize())
    else:
        image_dir = output_dir

    os.makedirs(image_dir, exist_ok=True)
    output_path = os.path.join(image_dir, f"{case_type.capitalize()}_{time_value}.png")
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
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
    save=True,
):
    """
    Plot SOD shock tube simulation results (1D line plots).

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
    save : bool, optional
        If True, save to disk and close the figure (default). If False,
        return the figure without saving (e.g. for fig_to_image then close).

    Returns
    -------
    str or matplotlib.figure.Figure
        If save=True, path to the saved image file. If save=False, the figure.
    """
    rho_nn = np.asarray(detach(rho_NN[2, :]))
    ux_nn = np.asarray(detach(ux_NN[2, :]))
    T_nn = np.asarray(detach(T_NN[2, :]))
    P_nn = np.asarray(detach(P_NN[2, :]))
    rho_gt = np.asarray(rho_GT[2, :])
    ux_gt = np.asarray(ux_GT[2, :])
    T_gt = np.asarray(T_GT[2, :])
    P_gt = np.asarray(P_GT[2, :])
    err_rho = np.abs(rho_nn - rho_gt)
    err_ux = np.abs(ux_nn - ux_gt)
    err_T = np.abs(T_nn - T_gt)
    err_P = np.abs(P_nn - P_gt)

    fig = plt.figure(figsize=SOD_FIGSIZE)

    # Larger title and reduced whitespace
    plt.suptitle(
        f"SOD shock case {case_number} time {time_step}",
        fontweight="bold",
        fontsize=SOD_TITLE_FONTSIZE,
        y=0.98,
    )

    # Row 1: NN vs GT
    plt.subplot(2, 4, 1)
    plt.plot(rho_nn, linewidth=SOD_LINEWIDTH, label="NN")
    plt.plot(rho_gt, linewidth=2, label="GT")
    plt.title("Density", fontsize=SOD_SUBPLOT_TITLE_FONTSIZE)

    plt.subplot(2, 4, 2)
    plt.plot(T_nn, linewidth=SOD_LINEWIDTH)
    plt.plot(T_gt, linewidth=2)
    plt.title("Temperature", fontsize=SOD_SUBPLOT_TITLE_FONTSIZE)

    plt.subplot(2, 4, 3)
    plt.plot(ux_nn, linewidth=SOD_LINEWIDTH)
    plt.plot(ux_gt, linewidth=2)
    plt.title("Velocity in x", fontsize=SOD_SUBPLOT_TITLE_FONTSIZE)

    plt.subplot(2, 4, 4)
    plt.plot(P_nn, linewidth=SOD_LINEWIDTH)
    plt.plot(P_gt, linewidth=2)
    plt.title("Pressure", fontsize=SOD_SUBPLOT_TITLE_FONTSIZE)

    # Row 2: Absolute error
    plt.subplot(2, 4, 5)
    plt.plot(err_rho, linewidth=2, color="C2")
    plt.title("|NN − GT| Density", fontsize=SOD_SUBPLOT_TITLE_FONTSIZE)

    plt.subplot(2, 4, 6)
    plt.plot(err_T, linewidth=2, color="C2")
    plt.title("|NN − GT| Temperature", fontsize=SOD_SUBPLOT_TITLE_FONTSIZE)

    plt.subplot(2, 4, 7)
    plt.plot(err_ux, linewidth=2, color="C2")
    plt.title("|NN − GT| Velocity", fontsize=SOD_SUBPLOT_TITLE_FONTSIZE)

    plt.subplot(2, 4, 8)
    plt.plot(err_P, linewidth=2, color="C2")
    plt.title("|NN − GT| Pressure", fontsize=SOD_SUBPLOT_TITLE_FONTSIZE)

    # Reduced whitespace
    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=0.35, w_pad=0.35)

    if not save:
        return fig

    # Save to Images directory
    if output_dir is None:
        current_file = os.path.abspath(__file__)
        main_dir = os.path.dirname(os.path.dirname(current_file))
        image_dir = os.path.join(main_dir, "images", f"SOD_case{case_number}")
    else:
        image_dir = output_dir

    os.makedirs(image_dir, exist_ok=True)
    output_path = os.path.join(image_dir, f"SOD_case{case_number}_{time_step}.png")
    fig.savefig(output_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    return output_path
