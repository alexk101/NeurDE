"""
Plotting interface for visualization of simulation results.

This module is the public API: it delegates to a plotting backend (matplotlib,
plotly, etc.) so callers can use a single import while the implementation is
swappable.

Usage by case:
-------------
Cylinder/Cylinder_faster:
    - plot_cylinder_results: 2D field comparison (NN, GT, error)
    - plot_cylinder_field: single 2D Mach field (for datasets / animation)

SOD_shock_tube:
    - plot_sod_results: 1D comparison (4 fields + 4 errors)
    - plot_sod_profiles: single 1D profiles (rho, ux, T, P) for datasets / animation

Stage 1 validation:
    - plot_stage1_validation: Geq GT vs NN

Backends create figures with appropriate DPI at instantiation. For experiment
tracking, pass the returned figure to the tracker's log_figure() method.
MLflow and W&B both support matplotlib and plotly figures natively.
"""

from __future__ import annotations

from typing import Any

from .animation import dataset_to_gif
from .plotters import get_plotter

# Default backend; can be overridden via set_plotting_backend()
_plotter = get_plotter("matplotlib")

# Re-export constants from the default backend so existing code keeps working
from .plotters.matplotlib import (
    CYLINDER_FIGSIZE,
    CYLINDER_TITLE_FONTSIZE,
    FIG_DPI,
    SOD_FIGSIZE,
    SOD_LINEWIDTH,
    SOD_SUBPLOT_TITLE_FONTSIZE,
    SOD_TITLE_FONTSIZE,
    STAGE1_FIGSIZE,
)

__all__ = [
    "dataset_to_gif",
    "plot_cylinder_field",
    "plot_cylinder_results",
    "plot_sod_profiles",
    "plot_sod_results",
    "plot_stage1_validation",
    "set_plotting_backend",
    "get_plotting_backend",
    "close_figure",
    "FIG_DPI",
    "CYLINDER_FIGSIZE",
    "SOD_FIGSIZE",
    "STAGE1_FIGSIZE",
    "CYLINDER_TITLE_FONTSIZE",
    "SOD_TITLE_FONTSIZE",
    "SOD_SUBPLOT_TITLE_FONTSIZE",
    "SOD_LINEWIDTH",
]


def set_plotting_backend(backend: str) -> None:
    """Set the active plotting backend (e.g. 'matplotlib', 'plotly')."""
    global _plotter
    _plotter = get_plotter(backend)


def get_plotting_backend() -> Any:
    """Return the current plotter module."""
    return _plotter


def close_figure(fig: Any) -> None:
    """Close the figure if it is a matplotlib figure; no-op for other backends (e.g. plotly)."""
    try:
        from matplotlib.figure import Figure
        if isinstance(fig, Figure):
            import matplotlib.pyplot as plt
            plt.close(fig)
    except ImportError:
        pass


def plot_cylinder_field(
    Ma: Any,
    title: str = "Mach number",
    ax: Any = None,
    vmin: Any = None,
    vmax: Any = None,
    cmap: str = "jet",
) -> Any:
    """Draw a single 2D Mach field. Returns figure (or figure containing ax if ax given)."""
    return _plotter.plot_cylinder_field(
        Ma, title=title, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap
    )


def plot_sod_profiles(
    rho_2d: Any,
    ux_2d: Any,
    T_2d: Any,
    P_2d: Any,
    time_step: Any,
    case_number: int,
    output_dir: str | None = None,
    save: bool = True,
) -> Any:
    """Draw a single view of SOD 1D profiles (rho, ux, T, P). Returns path if save else figure."""
    return _plotter.plot_sod_profiles(
        rho_2d,
        ux_2d,
        T_2d,
        P_2d,
        time_step,
        case_number,
        output_dir=output_dir,
        save=save,
    )


def plot_cylinder_results(
    Ma_NN: Any,
    Ma_GT: Any,
    time_value: Any,
    case_type: str = "cylinder",
    output_dir: str | None = None,
    save: bool = True,
) -> Any:
    """Plot cylinder 2D field comparison (NN, GT, error). Returns path if save else figure."""
    return _plotter.plot_cylinder_results(
        Ma_NN,
        Ma_GT,
        time_value,
        case_type=case_type,
        output_dir=output_dir,
        save=save,
    )


def plot_sod_results(
    rho_NN: Any,
    ux_NN: Any,
    T_NN: Any,
    P_NN: Any,
    rho_GT: Any,
    ux_GT: Any,
    T_GT: Any,
    P_GT: Any,
    time_step: Any,
    case_number: int,
    output_dir: str | None = None,
    save: bool = True,
) -> Any:
    """Plot SOD 1D comparison (4 fields + 4 errors). Returns path if save else figure."""
    return _plotter.plot_sod_results(
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
        output_dir=output_dir,
        save=save,
    )


def plot_stage1_validation(
    Geq_GT: Any,
    Geq_NN: Any,
    case_type: str,
    epoch_or_sample: int = 0,
    channel: int = 0,
    output_dir: str | None = None,
    save: bool = True,
) -> Any:
    """Plot Stage 1 Geq GT vs NN. Returns path if save else figure."""
    return _plotter.plot_stage1_validation(
        Geq_GT,
        Geq_NN,
        case_type,
        epoch_or_sample=epoch_or_sample,
        channel=channel,
        output_dir=output_dir,
        save=save,
    )
