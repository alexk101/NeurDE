"""
Plotting backends implementing the same interface.

Use get_plotter(backend) to obtain the module implementing plot_cylinder_results,
plot_sod_results, and plot_stage1_validation. The default backend is "matplotlib".
Plotly is loaded lazily so Kaleido/Chrome setup only runs when plotly is selected.
"""

from __future__ import annotations

from typing import Any

from . import matplotlib as _matplotlib

_SUPPORTED_BACKENDS = ("matplotlib", "plotly")
_plotly_module: Any = None


def get_plotter(backend: str = "matplotlib"):
    """Return the plotter module for the given backend."""
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown plotting backend: {backend}. Choose from {list(_SUPPORTED_BACKENDS)}."
        )
    if backend == "matplotlib":
        return _matplotlib
    # Lazy-load plotly so ensure_kaleido_chrome() runs only when plotly is used
    global _plotly_module
    if _plotly_module is None:
        from . import plotly as _pl
        _plotly_module = _pl
    return _plotly_module
