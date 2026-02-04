"""
Abstract interface for plotting backends.

Backends (matplotlib, plotly, etc.) implement the same plotting functions
and return backend-specific figure objects. The experiment tracker handles
logging (e.g. MLflow.log_figure for matplotlib, or conversion to image for WandB).
"""

from __future__ import annotations

from typing import Any, Protocol, Union

# Backend-specific figure types (matplotlib.figure.Figure, plotly.graph_objects.Figure, etc.)
FigureT = Any


class PlotterBackend(Protocol):
    """Protocol for a plotting backend. All plot functions return a figure-like object."""

    def plot_cylinder_results(
        self,
        Ma_NN: Any,
        Ma_GT: Any,
        time_value: Any,
        case_type: str = "cylinder",
        output_dir: str | None = None,
        save: bool = True,
    ) -> Union[str, FigureT]:
        """Plot cylinder 2D field comparison (NN, GT, error). Returns path if save else figure."""
        ...

    def plot_sod_results(
        self,
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
    ) -> Union[str, FigureT]:
        """Plot SOD 1D comparison (4 fields + 4 errors). Returns path if save else figure."""
        ...

    def plot_stage1_validation(
        self,
        Geq_GT: Any,
        Geq_NN: Any,
        case_type: str,
        epoch_or_sample: int = 0,
        channel: int = 0,
        output_dir: str | None = None,
        save: bool = True,
    ) -> Union[str, FigureT]:
        """Plot Stage 1 Geq GT vs NN. Returns path if save else figure."""
        ...
