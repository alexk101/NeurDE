"""
Animation utilities for dataset visualization.

Produces GIFs showing evolution over time for cylinder (2D Mach field)
and SOD shock tube (1D rho, ux, T, P) datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .plotters import get_plotter


def _figure_to_pil_image(fig: Any) -> Image.Image:
    """Render a matplotlib figure to a PIL Image (RGB)."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    # matplotlib 3.8+ uses tostring_argb(); older versions use tostring_rgb()
    if hasattr(fig.canvas, "tostring_rgb"):
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape((h, w, 3))
    else:
        buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf = buf.reshape((h, w, 4))  # ARGB
        buf = buf[:, :, 1:4].copy()   # drop alpha -> RGB
    return Image.fromarray(buf)


def _close_figure(fig: Any) -> None:
    """Close a matplotlib figure to free memory."""
    try:
        from matplotlib.figure import Figure
        if isinstance(fig, Figure):
            import matplotlib.pyplot as plt
            plt.close(fig)
    except ImportError:
        pass


def _compute_mach(ux: np.ndarray, uy: np.ndarray, T: np.ndarray, vuy: float = 1.4) -> np.ndarray:
    """Compute Mach number from velocity and temperature. Ma = sqrt((ux^2+uy^2)/(vuy*T))."""
    uu = ux.astype(np.float64) ** 2 + uy.astype(np.float64) ** 2
    cs = vuy * T.astype(np.float64)
    cs = np.maximum(cs, 1e-12)
    return np.sqrt(uu / cs).astype(np.float32)


def _compute_pressure(rho: np.ndarray, T: np.ndarray, R: float = 1.0) -> np.ndarray:
    """Compute pressure P = R * rho * T."""
    return (R * rho.astype(np.float64) * T.astype(np.float64)).astype(np.float32)


def dataset_to_gif(
    h5_path: str | Path,
    output_path: str | Path,
    case_type: str,
    *,
    step_skip: int = 1,
    duration_ms: int = 100,
    vuy: float | None = None,
    max_frames: int | None = None,
) -> Path:
    """
    Generate a GIF animation of a dataset (evolution over time).

    Loads rho, ux, uy, T from the HDF5 file and builds one frame per (possibly
    skipped) time step. Cylinder: 2D Mach number. SOD: 1D rho, ux, T, P at mid row.

    Parameters
    ----------
    h5_path : path
        Path to the HDF5 file (e.g. data_base/cylinder_case.h5 or data_base/SOD_case1.h5).
    output_path : path
        Where to save the output GIF.
    case_type : str
        One of "cylinder", "cylinder_faster", "sod_shock_tube".
    step_skip : int
        Use every step_skip-th time step (default 1).
    duration_ms : int
        Display duration per frame in milliseconds (default 100).
    vuy : float, optional
        Physics constant for Mach (cylinder: 1.4, SOD: 2). Inferred from case_type if None.
    max_frames : int, optional
        Cap number of frames (e.g. 100). If None, use all steps (after step_skip).

    Returns
    -------
    pathlib.Path
        Path to the saved GIF file.
    """
    import h5py

    h5_path = Path(h5_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ct = case_type.lower().strip()
    if ct not in ("cylinder", "cylinder_faster", "sod_shock_tube"):
        raise ValueError(f"case_type must be one of cylinder, cylinder_faster, sod_shock_tube; got {case_type!r}")

    if vuy is None:
        vuy = 2.0 if ct == "sod_shock_tube" else 1.4

    R = 1.0  # gas constant from solver (R = Cp - Cv = 1 for this LBM)

    plotter = get_plotter("matplotlib")

    with h5py.File(h5_path, "r") as f:
        rho = np.asarray(f["rho"])
        ux = np.asarray(f["ux"])
        uy = np.asarray(f["uy"])
        T = np.asarray(f["T"])

    n_steps = rho.shape[0]
    indices = list(range(0, n_steps, step_skip))
    if max_frames is not None and len(indices) > max_frames:
        indices = indices[: max_frames]
    if not indices:
        raise ValueError(f"No frames: n_steps={n_steps}, step_skip={step_skip}")

    images: list[Image.Image] = []

    if ct in ("cylinder", "cylinder_faster"):
        for i in indices:
            Ma = _compute_mach(ux[i], uy[i], T[i], vuy=vuy)
            fig = plotter.plot_cylinder_field(
                Ma,
                title=f"Mach number — step {i}",
                ax=None,
                vmin=None,
                vmax=None,
            )
            images.append(_figure_to_pil_image(fig))
            _close_figure(fig)
    else:
        # SOD: need case_number for plot title; try to infer from filename (e.g. SOD_case1.h5)
        case_number = 1
        name = h5_path.name.lower()
        if "case2" in name or "case_2" in name:
            case_number = 2
        for i in indices:
            P_2d = _compute_pressure(rho[i], T[i], R=R)
            fig = plotter.plot_sod_profiles(
                rho[i],
                ux[i],
                T[i],
                P_2d,
                i,
                case_number,
                output_dir=None,
                save=False,
            )
            images.append(_figure_to_pil_image(fig))
            _close_figure(fig)

    if not images:
        raise RuntimeError("No frames were produced")

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        loop=0,
        duration=int(duration_ms),
    )
    return output_path
