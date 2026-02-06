"""
Plotly plotting backend.

Creates interactive figures using plotly.graph_objects. Figures can be
logged to experiment trackers (W&B, MLflow) as interactive HTML or
static images depending on tracker configuration.
"""

from __future__ import annotations

import logging
import os
from typing import Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Silence kaleido and choreographer (image export) log noise
logging.getLogger("kaleido").setLevel(logging.WARNING)
logging.getLogger("choreographer").setLevel(logging.WARNING)

from ..core import detach, ensure_kaleido_chrome

ensure_kaleido_chrome()

# -----------------------------------------------------------------------------
# Constants (matching matplotlib backend for consistency)
# -----------------------------------------------------------------------------

FIG_DPI = 200  # Used for static image export
CYLINDER_FIGSIZE = (14, 5)  # width, height in inches
SOD_FIGSIZE = (16, 10)
SOD_LINEWIDTH = 5
CYLINDER_TITLE_FONTSIZE = 16
SOD_TITLE_FONTSIZE = 25
SOD_SUBPLOT_TITLE_FONTSIZE = 18
STAGE1_FIGSIZE = (14, 5)

# Convert inches to pixels (plotly uses pixels)
_DPI_SCALE = 100  # Base DPI for size conversion
CYLINDER_SIZE = (
    int(CYLINDER_FIGSIZE[0] * _DPI_SCALE),
    int(CYLINDER_FIGSIZE[1] * _DPI_SCALE),
)
SOD_SIZE = (int(SOD_FIGSIZE[0] * _DPI_SCALE), int(SOD_FIGSIZE[1] * _DPI_SCALE))
STAGE1_SIZE = (int(STAGE1_FIGSIZE[0] * _DPI_SCALE), int(STAGE1_FIGSIZE[1] * _DPI_SCALE))


def _image_dir(default_subdir: str, output_dir: str | None, base_dir: str) -> str:
    """Get the output directory for saving images."""
    if output_dir is not None:
        return output_dir
    return os.path.join(base_dir, "images", default_subdir)


# Built-in plotly colorscales (case-insensitive strings)
# See: https://plotly.com/python/builtin-colorscales/
COLORSCALE_JET = "jet"
COLORSCALE_HOT = "hot"

# Single-panel sizes (for plot_cylinder_field, plot_sod_profiles)
CYLINDER_SINGLE_FIGSIZE = (5, 4)
SOD_SINGLE_FIGSIZE = (16, 4)
SOD_ROW_INDEX = 2

CYLINDER_SINGLE_SIZE = (
    int(CYLINDER_SINGLE_FIGSIZE[0] * _DPI_SCALE),
    int(CYLINDER_SINGLE_FIGSIZE[1] * _DPI_SCALE),
)
SOD_SINGLE_SIZE = (
    int(SOD_SINGLE_FIGSIZE[0] * _DPI_SCALE),
    int(SOD_SINGLE_FIGSIZE[1] * _DPI_SCALE),
)


def plot_cylinder_field(
    Ma,
    title: str = "Mach number",
    ax=None,
    vmin=None,
    vmax=None,
    cmap: str = "jet",
) -> go.Figure:
    """Draw a single 2D Mach field. (ax is ignored in plotly backend.)"""
    Ma = np.asarray(Ma)
    if vmin is None:
        vmin = float(np.nanmin(Ma))
    if vmax is None:
        vmax = float(np.nanmax(Ma))
    fig = go.Figure(
        data=go.Heatmap(
            z=Ma,
            colorscale=cmap,
            zmin=vmin,
            zmax=vmax,
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=CYLINDER_TITLE_FONTSIZE), x=0.5),
        width=CYLINDER_SINGLE_SIZE[0],
        height=CYLINDER_SINGLE_SIZE[1],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, autorange="reversed"),
    )
    return fig


def _sod_2d_to_1d(rho_2d, ux_2d, T_2d, P_2d):
    """Extract 1D profile at SOD_ROW_INDEX. Handles tensors and arrays."""
    def to_1d(arr):
        s = arr[SOD_ROW_INDEX, :]
        return np.asarray(detach(s) if hasattr(s, "detach") else s)
    return (to_1d(rho_2d), to_1d(ux_2d), to_1d(T_2d), to_1d(P_2d))


def plot_sod_profiles(
    rho_2d,
    ux_2d,
    T_2d,
    P_2d,
    time_step,
    case_number: int,
    output_dir: str | None = None,
    save: bool = True,
) -> Union[str, go.Figure]:
    """Draw a single view of SOD 1D profiles (rho, ux, T, P)."""
    rho, ux, T, P = _sod_2d_to_1d(rho_2d, ux_2d, T_2d, P_2d)
    fig = make_subplots(
        rows=1,
        cols=4,
        subplot_titles=["Density", "Temperature", "Velocity in x", "Pressure"],
        horizontal_spacing=0.06,
    )
    for col, data in enumerate([rho, T, ux, P], start=1):
        fig.add_trace(go.Scatter(y=data, line=dict(width=SOD_LINEWIDTH)), row=1, col=col)
    fig.update_layout(
        title=dict(
            text=f"SOD shock case {case_number} time {time_step}",
            font=dict(size=SOD_TITLE_FONTSIZE),
            x=0.5,
        ),
        width=SOD_SINGLE_SIZE[0],
        height=SOD_SINGLE_SIZE[1],
        showlegend=False,
    )
    for col in range(1, 5):
        fig.update_xaxes(visible=False, row=1, col=col)
        fig.update_yaxes(visible=False, row=1, col=col)

    if not save:
        return fig
    current_file = os.path.abspath(__file__)
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    image_dir = _image_dir(f"SOD_case{case_number}", output_dir, main_dir)
    os.makedirs(image_dir, exist_ok=True)
    output_path = os.path.join(image_dir, f"SOD_case{case_number}_{time_step}.html")
    fig.write_html(output_path)
    return output_path


def plot_cylinder_results(
    Ma_NN,
    Ma_GT,
    time_value,
    case_type: str = "cylinder",
    output_dir: str | None = None,
    save: bool = True,
) -> Union[str, go.Figure]:
    """Plot cylinder 2D field comparison (NN, GT, error).

    Parameters
    ----------
    Ma_NN : array-like
        Neural network predicted Mach number field.
    Ma_GT : array-like
        Ground truth Mach number field.
    time_value : int or float
        Time step or sample index for labeling.
    case_type : str
        Case identifier (e.g., "cylinder", "cylinder_faster").
    output_dir : str | None
        Output directory for saving. If None, uses default.
    save : bool
        If True, save to file and return path. If False, return figure.

    Returns
    -------
    str or go.Figure
        File path if save=True, else the plotly Figure object.
    """
    Ma_NN = np.asarray(Ma_NN)
    Ma_GT = np.asarray(Ma_GT)
    Ma_err = np.abs(Ma_NN - Ma_GT)

    vmin = float(np.nanmin([Ma_NN.min(), Ma_GT.min()]))
    vmax = float(np.nanmax([Ma_NN.max(), Ma_GT.max()]))

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Mach number - NN", "Mach number - GT", "|NN - GT|"],
        horizontal_spacing=0.08,
    )

    # NN prediction
    fig.add_trace(
        go.Heatmap(
            z=Ma_NN,
            colorscale=COLORSCALE_JET,
            zmin=vmin,
            zmax=vmax,
            colorbar=dict(x=0.29, len=0.9),
        ),
        row=1,
        col=1,
    )

    # Ground truth
    fig.add_trace(
        go.Heatmap(
            z=Ma_GT,
            colorscale=COLORSCALE_JET,
            zmin=vmin,
            zmax=vmax,
            colorbar=dict(x=0.635, len=0.9),
        ),
        row=1,
        col=2,
    )

    # Error
    fig.add_trace(
        go.Heatmap(
            z=Ma_err,
            colorscale=COLORSCALE_HOT,
            colorbar=dict(x=1.0, len=0.9),
        ),
        row=1,
        col=3,
    )

    fig.update_layout(
        title=dict(
            text=f"{case_type.capitalize()} - Sample {time_value}",
            font=dict(size=CYLINDER_TITLE_FONTSIZE),
            x=0.5,
        ),
        width=CYLINDER_SIZE[0],
        height=CYLINDER_SIZE[1],
        showlegend=False,
    )

    # Hide axes for cleaner look
    for i in range(1, 4):
        fig.update_xaxes(visible=False, row=1, col=i)
        fig.update_yaxes(visible=False, autorange="reversed", row=1, col=i)

    if not save:
        return fig

    current_file = os.path.abspath(__file__)
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    image_dir = _image_dir(case_type.capitalize(), output_dir, main_dir)
    os.makedirs(image_dir, exist_ok=True)
    output_path = os.path.join(image_dir, f"{case_type.capitalize()}_{time_value}.html")
    fig.write_html(output_path)
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
    case_number: int,
    output_dir: str | None = None,
    save: bool = True,
) -> Union[str, go.Figure]:
    """Plot SOD 1D comparison (4 fields + 4 errors).

    Parameters
    ----------
    rho_NN, ux_NN, T_NN, P_NN : array-like
        Neural network predictions for density, velocity, temperature, pressure.
    rho_GT, ux_GT, T_GT, P_GT : array-like
        Ground truth values.
    time_step : int or float
        Time step for labeling.
    case_number : int
        SOD case number.
    output_dir : str | None
        Output directory for saving.
    save : bool
        If True, save to file and return path. If False, return figure.

    Returns
    -------
    str or go.Figure
        File path if save=True, else the plotly Figure object.
    """
    # Extract middle slice and convert to numpy
    rho_nn = np.asarray(detach(rho_NN[2, :]))
    ux_nn = np.asarray(detach(ux_NN[2, :]))
    T_nn = np.asarray(detach(T_NN[2, :]))
    P_nn = np.asarray(detach(P_NN[2, :]))
    rho_gt = np.asarray(rho_GT[2, :])
    ux_gt = np.asarray(ux_GT[2, :])
    T_gt = np.asarray(T_GT[2, :])
    P_gt = np.asarray(P_GT[2, :])

    # Compute errors
    err_rho = np.abs(rho_nn - rho_gt)
    err_ux = np.abs(ux_nn - ux_gt)
    err_T = np.abs(T_nn - T_gt)
    err_P = np.abs(P_nn - P_gt)

    x = np.arange(len(rho_nn))

    fig = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=[
            "Density",
            "Temperature",
            "Velocity in x",
            "Pressure",
            "|NN - GT| Density",
            "|NN - GT| Temperature",
            "|NN - GT| Velocity",
            "|NN - GT| Pressure",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.06,
    )

    # Top row: NN vs GT comparisons
    # Density
    fig.add_trace(
        go.Scatter(
            x=x, y=rho_nn, mode="lines", name="NN", line=dict(width=SOD_LINEWIDTH)
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=x, y=rho_gt, mode="lines", name="GT", line=dict(width=2)),
        row=1,
        col=1,
    )

    # Temperature
    fig.add_trace(
        go.Scatter(
            x=x,
            y=T_nn,
            mode="lines",
            name="NN",
            line=dict(width=SOD_LINEWIDTH),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=T_gt, mode="lines", name="GT", line=dict(width=2), showlegend=False
        ),
        row=1,
        col=2,
    )

    # Velocity
    fig.add_trace(
        go.Scatter(
            x=x,
            y=ux_nn,
            mode="lines",
            name="NN",
            line=dict(width=SOD_LINEWIDTH),
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=ux_gt, mode="lines", name="GT", line=dict(width=2), showlegend=False
        ),
        row=1,
        col=3,
    )

    # Pressure
    fig.add_trace(
        go.Scatter(
            x=x,
            y=P_nn,
            mode="lines",
            name="NN",
            line=dict(width=SOD_LINEWIDTH),
            showlegend=False,
        ),
        row=1,
        col=4,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=P_gt, mode="lines", name="GT", line=dict(width=2), showlegend=False
        ),
        row=1,
        col=4,
    )

    # Bottom row: Errors (green color like C2 in matplotlib)
    error_color = "rgb(44, 160, 44)"  # matplotlib C2 green

    fig.add_trace(
        go.Scatter(
            x=x,
            y=err_rho,
            mode="lines",
            line=dict(width=2, color=error_color),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=err_T,
            mode="lines",
            line=dict(width=2, color=error_color),
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=err_ux,
            mode="lines",
            line=dict(width=2, color=error_color),
            showlegend=False,
        ),
        row=2,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=err_P,
            mode="lines",
            line=dict(width=2, color=error_color),
            showlegend=False,
        ),
        row=2,
        col=4,
    )

    fig.update_layout(
        title=dict(
            text=f"<b>SOD shock case {case_number} time {time_step}</b>",
            font=dict(size=SOD_TITLE_FONTSIZE),
            x=0.5,
            y=0.98,
        ),
        width=SOD_SIZE[0],
        height=SOD_SIZE[1],
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
    )

    # Update subplot title font sizes
    for annotation in fig.layout.annotations:
        annotation.font.size = SOD_SUBPLOT_TITLE_FONTSIZE

    if not save:
        return fig

    current_file = os.path.abspath(__file__)
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    image_dir = _image_dir(f"SOD_case{case_number}", output_dir, main_dir)
    os.makedirs(image_dir, exist_ok=True)
    output_path = os.path.join(image_dir, f"SOD_case{case_number}_{time_step}.html")
    fig.write_html(output_path)
    return output_path


def plot_stage1_validation(
    Geq_GT,
    Geq_NN,
    case_type: str,
    epoch_or_sample: int = 0,
    channel: int = 0,
    output_dir: str | None = None,
    save: bool = True,
) -> Union[str, go.Figure]:
    """Plot Stage 1 Geq GT vs NN comparison.

    Parameters
    ----------
    Geq_GT : array-like
        Ground truth Geq values, shape (channels, H, W) or (channels, H, W).
    Geq_NN : array-like
        Neural network predicted Geq values.
    case_type : str
        Case type identifier (e.g., "cylinder", "sod_shock_tube").
    epoch_or_sample : int
        Epoch or sample number for labeling.
    channel : int
        Which channel to visualize.
    output_dir : str | None
        Output directory for saving.
    save : bool
        If True, save to file and return path. If False, return figure.

    Returns
    -------
    str or go.Figure
        File path if save=True, else the plotly Figure object.
    """
    Geq_GT = np.asarray(Geq_GT)
    Geq_NN = np.asarray(Geq_NN)
    err = np.sqrt(np.sum((Geq_GT - Geq_NN) ** 2, axis=0))
    gt_ch = Geq_GT[channel]
    nn_ch = Geq_NN[channel]

    is_1d = case_type.lower() == "sod_shock_tube"

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            f"Geq ch{channel} GT",
            f"Geq ch{channel} NN",
            "|GT - NN| L2",
        ],
        horizontal_spacing=0.08,
    )

    if is_1d:
        # 1D line plots (take middle slice)
        mid = gt_ch.shape[0] // 2
        gt_line = gt_ch[mid, :]
        nn_line = nn_ch[mid, :]
        err_line = err[mid, :]
        x = np.arange(len(gt_line))

        ymin = float(np.nanmin([gt_line.min(), nn_line.min()]))
        ymax = float(np.nanmax([gt_line.max(), nn_line.max()]))

        fig.add_trace(
            go.Scatter(
                x=x, y=gt_line, mode="lines", line=dict(width=2), showlegend=False
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=nn_line, mode="lines", line=dict(width=2), showlegend=False
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=err_line,
                mode="lines",
                line=dict(width=2, color="rgb(44, 160, 44)"),
                showlegend=False,
            ),
            row=1,
            col=3,
        )

        # Set consistent y-axis range for GT and NN
        fig.update_yaxes(range=[ymin, ymax], row=1, col=1)
        fig.update_yaxes(range=[ymin, ymax], row=1, col=2)

    else:
        # 2D heatmaps
        vmin = float(np.nanmin([gt_ch.min(), nn_ch.min()]))
        vmax = float(np.nanmax([gt_ch.max(), nn_ch.max()]))

        fig.add_trace(
            go.Heatmap(
                z=gt_ch,
                colorscale=COLORSCALE_JET,
                zmin=vmin,
                zmax=vmax,
                colorbar=dict(x=0.29, len=0.9),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Heatmap(
                z=nn_ch,
                colorscale=COLORSCALE_JET,
                zmin=vmin,
                zmax=vmax,
                colorbar=dict(x=0.635, len=0.9),
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Heatmap(
                z=err,
                colorscale=COLORSCALE_HOT,
                colorbar=dict(x=1.0, len=0.9),
            ),
            row=1,
            col=3,
        )

        # Reverse y-axis for image-like display
        for i in range(1, 4):
            fig.update_xaxes(visible=False, row=1, col=i)
            fig.update_yaxes(visible=False, autorange="reversed", row=1, col=i)

    fig.update_layout(
        title=dict(
            text=f"Stage 1 validation ({case_type}) — sample/epoch {epoch_or_sample}",
            font=dict(size=CYLINDER_TITLE_FONTSIZE),
            x=0.5,
        ),
        width=STAGE1_SIZE[0],
        height=STAGE1_SIZE[1],
        showlegend=False,
    )

    if not save:
        return fig

    current_file = os.path.abspath(__file__)
    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    image_dir = _image_dir("stage1_validation", output_dir, main_dir)
    os.makedirs(image_dir, exist_ok=True)
    output_path = os.path.join(
        image_dir, f"stage1_{case_type}_epoch_{epoch_or_sample}.html"
    )
    fig.write_html(output_path)
    return output_path
