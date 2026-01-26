"""
Unified solver implementations and factory.

This package provides unified solver classes for all cases:
- BaseLBSolver: Base class with common functionality
- CylinderSolver: Cylinder case solver (handles both standard and faster variants)
- SODSolver: SOD shock tube case solver
- create_solver: Factory function to create appropriate solver

Usage by case:
-------------
Cylinder:
    - solver_type: "cylinder_base"
    - Uses dense matrix inversion for BC/Obs (use_dense_inv=True)
    - Uses CSR sparse format (sparse_format="csr")

Cylinder_faster:
    - solver_type: "cylinder_faster"
    - Uses sparse matrix inversion for BC/Obs (use_dense_inv=False)
    - Uses CSC sparse format (sparse_format="csc")

SOD_shock_tube:
    - solver_type: "sod_solver"
    - No obstacle/BC handling
    - Uses CSR sparse format (sparse_format="csr")
"""

from typing import Any

from .base import BaseLBSolver
from .cylinder import CylinderSolver
from .sod import SODSolver

__all__ = ["BaseLBSolver", "CylinderSolver", "SODSolver", "create_solver"]


def create_solver(solver_type: str, **kwargs) -> Any:
    """
    Create a solver instance based on solver type.

    Parameters
    ----------
    solver_type : str
        Type of solver: "cylinder_base", "cylinder_faster", or "sod_solver"
    **kwargs
        Solver initialization parameters

    Returns
    -------
    Solver instance (CylinderSolver or SODSolver)

    Raises
    ------
    ValueError
        If solver_type is not recognized
    """
    if solver_type == "cylinder_base":
        # Standard Cylinder: dense inversion, CSR format
        return CylinderSolver(sparse_format="csr", use_dense_inv=True, **kwargs)

    elif solver_type == "cylinder_faster":
        # Faster Cylinder: sparse inversion, CSC format
        return CylinderSolver(sparse_format="csc", use_dense_inv=False, **kwargs)

    elif solver_type == "sod_solver":
        # SOD shock tube solver
        return SODSolver(**kwargs)

    else:
        raise ValueError(
            f"Unknown solver_type: {solver_type}. "
            "Must be one of: 'cylinder_base', 'cylinder_faster', 'sod_solver'"
        )
