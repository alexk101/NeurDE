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
SOD_FIGSIZE = (16, 10)
SOD_LINEWIDTH = 5
CYLINDER_TITLE_FONTSIZE = 16
SOD_TITLE_FONTSIZE = 25
SOD_SUBPLOT_TITLE_FONTSIZE = 18
STAGE1_FIGSIZE = (14, 5)


def _image_dir(default_subdir: str, output_dir: str | None, base_dir: str) -> str:
    if output_dir is not None:
        return output_dir
    return os.path.join(base_dir, "images", default_subdir)


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

    im1 = ax1.imshow(Ma_NN, cmap="jet", vmin=vmin, vmax=vmax)
    ax1.set_title("Mach number - NN", fontsize=CYLINDER_TITLE_FONTSIZE)
    plt.colorbar(im1, ax=ax1)
    ax1.axis("off")

    im2 = ax2.imshow(Ma_GT, cmap="jet", vmin=vmin, vmax=vmax)
    ax2.set_title("Mach number - GT", fontsize=CYLINDER_TITLE_FONTSIZE)
    plt.colorbar(im2, ax=ax2)
    ax2.axis("off")

    im3 = ax3.imshow(Ma_err, cmap="hot")
    ax3.set_title("|NN - GT|", fontsize=CYLINDER_TITLE_FONTSIZE)
    plt.colorbar(im3, ax=ax3)
    ax3.axis("off")

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

    fig = plt.figure(figsize=SOD_FIGSIZE, dpi=FIG_DPI)

    plt.suptitle(
        f"SOD shock case {case_number} time {time_step}",
        fontweight="bold",
        fontsize=SOD_TITLE_FONTSIZE,
        y=0.98,
    )

    plt.subplot(2, 4, 1)
    plt.plot(rho_nn, linewidth=SOD_LINEWIDTH, label="NN")
    plt.plot(rho_gt, linewidth=2, label="GT")
    plt.title("Density", fontsize=SOD_SUBPLOT_TITLE_FONTSIZE)
    plt.gca().axis("off")

    plt.subplot(2, 4, 2)
    plt.plot(T_nn, linewidth=SOD_LINEWIDTH)
    plt.plot(T_gt, linewidth=2)
    plt.title("Temperature", fontsize=SOD_SUBPLOT_TITLE_FONTSIZE)
    plt.gca().axis("off")

    plt.subplot(2, 4, 3)
    plt.plot(ux_nn, linewidth=SOD_LINEWIDTH)
    plt.plot(ux_gt, linewidth=2)
    plt.title("Velocity in x", fontsize=SOD_SUBPLOT_TITLE_FONTSIZE)
    plt.gca().axis("off")

    plt.subplot(2, 4, 4)
    plt.plot(P_nn, linewidth=SOD_LINEWIDTH)
    plt.plot(P_gt, linewidth=2)
    plt.title("Pressure", fontsize=SOD_SUBPLOT_TITLE_FONTSIZE)
    plt.gca().axis("off")

    plt.subplot(2, 4, 5)
    plt.plot(err_rho, linewidth=2, color="C2")
    plt.title("|NN - GT| Density", fontsize=SOD_SUBPLOT_TITLE_FONTSIZE)
    plt.gca().axis("off")

    plt.subplot(2, 4, 6)
    plt.plot(err_T, linewidth=2, color="C2")
    plt.title("|NN - GT| Temperature", fontsize=SOD_SUBPLOT_TITLE_FONTSIZE)
    plt.gca().axis("off")

    plt.subplot(2, 4, 7)
    plt.plot(err_ux, linewidth=2, color="C2")
    plt.title("|NN - GT| Velocity", fontsize=SOD_SUBPLOT_TITLE_FONTSIZE)
    plt.gca().axis("off")

    plt.subplot(2, 4, 8)
    plt.plot(err_P, linewidth=2, color="C2")
    plt.title("|NN - GT| Pressure", fontsize=SOD_SUBPLOT_TITLE_FONTSIZE)
    plt.gca().axis("off")

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
