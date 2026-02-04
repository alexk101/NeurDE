"""
Plotting backends implementing the same interface.

Use get_plotter(backend) to obtain the module implementing plot_cylinder_results,
plot_sod_results, and plot_stage1_validation. The default backend is "matplotlib".
"""

from __future__ import annotations

from typing import Any

from . import matplotlib as _matplotlib
from . import plotly as _plotly

_BACKENDS: dict[str, Any] = {
    "matplotlib": _matplotlib,
    "plotly": _plotly,
}


def get_plotter(backend: str = "matplotlib"):
    """Return the plotter module for the given backend."""
    if backend not in _BACKENDS:
        raise ValueError(
            f"Unknown plotting backend: {backend}. Choose from {list(_BACKENDS)}."
        )
    return _BACKENDS[backend]
