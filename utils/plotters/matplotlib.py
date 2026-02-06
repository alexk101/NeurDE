"""
Matplotlib plotting backend.

Figures are created with a fixed DPI (FIG_DPI) at instantiation so that
saved files and logged figures have consistent resolution. Experiment
trackers handle matplotlib figures natively via their log_figure() method.
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..core import detach

# -----------------------------------------------------------------------------
# Constants (shared with interface; backend may override)
# -----------------------------------------------------------------------------

FIG_DPI = 200
CYLINDER_FIGSIZE = (14, 5)
CYLINDER_SINGLE_FIGSIZE = (5, 4)
SOD_FIGSIZE = (16, 10)
SOD_SINGLE_FIGSIZE = (16, 4)
SOD_LINEWIDTH = 5
SOD_ROW_INDEX = 2  # row index for 1D slice in SOD plots
CYLINDER_TITLE_FONTSIZE = 16
SOD_TITLE_FONTSIZE = 25
SOD_SUBPLOT_TITLE_FONTSIZE = 18
STAGE1_FIGSIZE = (14, 5)


def _image_dir(default_subdir: str, output_dir: str | None, base_dir: str) -> str:
    if output_dir is not None:
        return output_dir
    return os.path.join(base_dir, "images", default_subdir)


def plot_cylinder_field(
    Ma,
    title="Mach number",
    ax=None,
    vmin=None,
    vmax=None,
    cmap="jet",
):
    """
    Draw a single 2D Mach number field. Used for dataset-only views and as a
    building block for plot_cylinder_results (NN / GT / error panels).

    Parameters
    ----------
    Ma : array-like
        2D Mach number (Y, X).
    title : str
        Subplot title.
    ax : matplotlib axes, optional
        If given, draw on this axis and return its figure. Otherwise create a new figure.
    vmin, vmax : float, optional
        Color scale. If None, use min/max of Ma.
    cmap : str
        Colormap name (default "jet").

    Returns
    -------
    matplotlib.figure.Figure
    """
    Ma = np.asarray(Ma)
    if vmin is None:
        vmin = float(np.nanmin(Ma))
    if vmax is None:
        vmax = float(np.nanmax(Ma))
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=CYLINDER_SINGLE_FIGSIZE, dpi=FIG_DPI)
    else:
        fig = ax.figure
    im = ax.imshow(Ma, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=CYLINDER_TITLE_FONTSIZE)
    plt.colorbar(im, ax=ax)
    ax.axis("off")
    return fig


def plot_cylinder_results(
    Ma_NN,
    Ma_GT,
    time_value,
    case_type="cylinder",
    output_dir=None,
    save=True,
):
    Ma_NN = np.asarray(Ma_NN)
    Ma_GT = np.asarray(Ma_GT)
    Ma_err = np.abs(Ma_NN - Ma_GT)

    vmin = float(np.nanmin([Ma_NN.min(), Ma_GT.min()]))
    vmax = float(np.nanmax([Ma_NN.max(), Ma_GT.max()]))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=CYLINDER_FIGSIZE, dpi=FIG_DPI)

    plot_cylinder_field(Ma_NN, title="Mach number - NN", ax=ax1, vmin=vmin, vmax=vmax)
    plot_cylinder_field(Ma_GT, title="Mach number - GT", ax=ax2, vmin=vmin, vmax=vmax)
    plot_cylinder_field(Ma_err, title="|NN - GT|", ax=ax3, vmin=None, vmax=None, cmap="hot")

    fig.suptitle(
        f"{case_type.capitalize()} - Sample {time_value}",
        fontsize=CYLINDER_TITLE_FONTSIZE,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=0.35, w_pad=0.35)

    if not save:
        return fig

    current_file = os.path.abspath(__file__)
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    image_dir = _image_dir(case_type.capitalize(), output_dir, main_dir)
    os.makedirs(image_dir, exist_ok=True)
    output_path = os.path.join(image_dir, f"{case_type.capitalize()}_{time_value}.png")
    fig.savefig(output_path, dpi=fig.get_dpi(), bbox_inches="tight")
    plt.close(fig)
    return output_path


def _sod_2d_to_1d(rho_2d, ux_2d, T_2d, P_2d):
    """Extract 1D profile at SOD_ROW_INDEX from 2D fields (Y, X). Handles tensors and arrays."""
    r = SOD_ROW_INDEX

    def to_1d(arr):
        s = arr[r, :]
        return np.asarray(detach(s) if hasattr(s, "detach") else s)

    return (to_1d(rho_2d), to_1d(ux_2d), to_1d(T_2d), to_1d(P_2d))


def _draw_sod_profile_row(axes, rho, ux, T, P, linewidth=2, label=None):
    """Draw one set of SOD 1D profiles (rho, ux, T, P) on four axes."""
    for ax, data, title in zip(
        axes,
        [rho, T, ux, P],
        ["Density", "Temperature", "Velocity in x", "Pressure"],
    ):
        ax.plot(data, linewidth=linewidth, label=label)
        ax.set_title(title, fontsize=SOD_SUBPLOT_TITLE_FONTSIZE)
        ax.axis("off")


def _draw_sod_errors_row(axes, err_rho, err_ux, err_T, err_P):
    """Draw SOD error profiles on four axes."""
    for ax, data, title in zip(
        axes,
        [err_rho, err_T, err_ux, err_P],
        ["|NN - GT| Density", "|NN - GT| Temperature", "|NN - GT| Velocity", "|NN - GT| Pressure"],
    ):
        ax.plot(data, linewidth=2, color="C2")
        ax.set_title(title, fontsize=SOD_SUBPLOT_TITLE_FONTSIZE)
        ax.axis("off")


def plot_sod_profiles(
    rho_2d,
    ux_2d,
    T_2d,
    P_2d,
    time_step,
    case_number,
    output_dir=None,
    save=True,
):
    """
    Draw a single view of SOD 1D profiles (rho, ux, T, P) at one time step.
    Used for dataset-only views (e.g. animation). Inputs are 2D (Y, X);
    the row at SOD_ROW_INDEX is plotted.

    Returns
    -------
    Figure or path depending on save.
    """
    rho, ux, T, P = _sod_2d_to_1d(rho_2d, ux_2d, T_2d, P_2d)
    fig, axes = plt.subplots(1, 4, figsize=SOD_SINGLE_FIGSIZE, dpi=FIG_DPI)
    fig.suptitle(
        f"SOD shock case {case_number} time {time_step}",
        fontweight="bold",
        fontsize=SOD_TITLE_FONTSIZE,
        y=0.98,
    )
    _draw_sod_profile_row(axes, rho, ux, T, P, linewidth=SOD_LINEWIDTH)
    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=0.35, w_pad=0.35)

    if not save:
        return fig

    current_file = os.path.abspath(__file__)
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    image_dir = _image_dir(f"SOD_case{case_number}", output_dir, main_dir)
    os.makedirs(image_dir, exist_ok=True)
    output_path = os.path.join(image_dir, f"SOD_case{case_number}_{time_step}.png")
    fig.savefig(output_path, dpi=fig.get_dpi(), bbox_inches="tight")
    plt.close(fig)
    return output_path


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
    rho_nn, ux_nn, T_nn, P_nn = _sod_2d_to_1d(rho_NN, ux_NN, T_NN, P_NN)
    rho_gt = np.asarray(rho_GT[SOD_ROW_INDEX, :])
    ux_gt = np.asarray(ux_GT[SOD_ROW_INDEX, :])
    T_gt = np.asarray(T_GT[SOD_ROW_INDEX, :])
    P_gt = np.asarray(P_GT[SOD_ROW_INDEX, :])
    err_rho = np.abs(rho_nn - rho_gt)
    err_ux = np.abs(ux_nn - ux_gt)
    err_T = np.abs(T_nn - T_gt)
    err_P = np.abs(P_nn - P_gt)

    fig, axes_2d = plt.subplots(2, 4, figsize=SOD_FIGSIZE, dpi=FIG_DPI)
    fig.suptitle(
        f"SOD shock case {case_number} time {time_step}",
        fontweight="bold",
        fontsize=SOD_TITLE_FONTSIZE,
        y=0.98,
    )
    row0, row1 = axes_2d[0], axes_2d[1]
    _draw_sod_profile_row(row0, rho_nn, ux_nn, T_nn, P_nn, linewidth=SOD_LINEWIDTH, label="NN")
    _draw_sod_profile_row(row0, rho_gt, ux_gt, T_gt, P_gt, linewidth=2, label="GT")
    _draw_sod_errors_row(row1, err_rho, err_ux, err_T, err_P)
    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=0.35, w_pad=0.35)

    if not save:
        return fig

    current_file = os.path.abspath(__file__)
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    image_dir = _image_dir(f"SOD_case{case_number}", output_dir, main_dir)
    os.makedirs(image_dir, exist_ok=True)
    output_path = os.path.join(image_dir, f"SOD_case{case_number}_{time_step}.png")
    fig.savefig(output_path, dpi=fig.get_dpi(), bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_stage1_validation(
    Geq_GT,
    Geq_NN,
    case_type: str,
    epoch_or_sample=0,
    channel: int = 0,
    output_dir=None,
    save=True,
):
    Geq_GT = np.asarray(Geq_GT)
    Geq_NN = np.asarray(Geq_NN)
    err = np.sqrt(np.sum((Geq_GT - Geq_NN) ** 2, axis=0))
    gt_ch = Geq_GT[channel]
    nn_ch = Geq_NN[channel]

    is_1d = case_type.lower() == "sod_shock_tube"
    fig, axes = plt.subplots(1, 3, figsize=STAGE1_FIGSIZE, dpi=FIG_DPI)

    if is_1d:
        mid = gt_ch.shape[0] // 2
        gt_line = gt_ch[mid, :]
        nn_line = nn_ch[mid, :]
        ymin = float(np.nanmin([gt_line.min(), nn_line.min()]))
        ymax = float(np.nanmax([gt_line.max(), nn_line.max()]))
        axes[0].plot(gt_line, linewidth=2)
        axes[0].set_ylim(ymin, ymax)
        axes[0].set_title(f"Geq ch{channel} GT", fontsize=CYLINDER_TITLE_FONTSIZE)
        axes[0].axis("off")
        axes[1].plot(nn_line, linewidth=2)
        axes[1].set_ylim(ymin, ymax)
        axes[1].set_title(f"Geq ch{channel} NN", fontsize=CYLINDER_TITLE_FONTSIZE)
        axes[1].axis("off")
        axes[2].plot(err[mid, :], linewidth=2, color="C2")
        axes[2].set_title("|GT - NN| L2", fontsize=CYLINDER_TITLE_FONTSIZE)
        axes[2].axis("off")
    else:
        vmin = float(np.nanmin([gt_ch.min(), nn_ch.min()]))
        vmax = float(np.nanmax([gt_ch.max(), nn_ch.max()]))
        im0 = axes[0].imshow(gt_ch, cmap="jet", vmin=vmin, vmax=vmax)
        axes[0].set_title(f"Geq ch{channel} GT", fontsize=CYLINDER_TITLE_FONTSIZE)
        plt.colorbar(im0, ax=axes[0])
        axes[0].axis("off")
        im1 = axes[1].imshow(nn_ch, cmap="jet", vmin=vmin, vmax=vmax)
        axes[1].set_title(f"Geq ch{channel} NN", fontsize=CYLINDER_TITLE_FONTSIZE)
        plt.colorbar(im1, ax=axes[1])
        axes[1].axis("off")
        im2 = axes[2].imshow(err, cmap="hot")
        axes[2].set_title("|GT - NN| L2", fontsize=CYLINDER_TITLE_FONTSIZE)
        plt.colorbar(im2, ax=axes[2])
        axes[2].axis("off")

    fig.suptitle(
        f"Stage 1 validation ({case_type}) — sample/epoch {epoch_or_sample}",
        fontsize=CYLINDER_TITLE_FONTSIZE,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=0.35, w_pad=0.35)

    if not save:
        return fig

    current_file = os.path.abspath(__file__)
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    image_dir = _image_dir("stage1_validation", output_dir, main_dir)
    os.makedirs(image_dir, exist_ok=True)
    output_path = os.path.join(
        image_dir, f"stage1_{case_type}_epoch_{epoch_or_sample}.png"
    )
    fig.savefig(output_path, dpi=fig.get_dpi(), bbox_inches="tight")
    plt.close(fig)
    return output_path
