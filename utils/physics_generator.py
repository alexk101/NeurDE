"""
Physics Generator module.

This module encapsulates the "Ground Truth" generation logic (Newton-Raphson solvers)
separating it from the main differentiable solver.
"""

import torch
import numpy as np
from .phys.getGeq import levermore_Geq, levermore_Geq_Obs, levermore_Geq_BCs
from .core import detach


class PhysicsGenerator:
    """
    Generator for computing equilibrium distributions using Newton-Raphson methods.
    Used for generating ground truth data and calculating boundary conditions.
    """

    def __init__(self, solver):
        """
        Initialize with a solver instance to access lattice constants.

        Parameters
        ----------
        solver : BaseLBSolver
            The solver instance containing lattice velocities (ex, ey), weights, etc.
        """
        self.solver = solver
        self.device = solver.device

    def get_Geq(self, rho, ux, uy, T, khi, zetax, zetay, sparse_format="csr"):
        """
        Compute G equilibrium for the full grid (Standard).
        Wraps levermore_Geq.
        """
        # Convert tensors to numpy arrays
        rho_np = detach(rho) if not isinstance(rho, np.ndarray) else rho
        ux_np = detach(ux) if not isinstance(ux, np.ndarray) else ux
        uy_np = detach(uy) if not isinstance(uy, np.ndarray) else uy
        T_np = detach(T) if not isinstance(T, np.ndarray) else T
        khi = detach(khi) if not isinstance(khi, np.ndarray) else khi
        zetax = detach(zetax) if not isinstance(zetax, np.ndarray) else zetax
        zetay = detach(zetay) if not isinstance(zetay, np.ndarray) else zetay

        # Compute Geq using unified levermore_Geq
        Geq_np, khi, zetax, zetay = levermore_Geq(
            detach(self.solver.ex),
            detach(self.solver.ey),
            ux_np,
            uy_np,
            T_np,
            rho_np,
            self.solver.Cv,
            self.solver.Qn,
            khi,
            zetax,
            zetay,
            sparse_format=sparse_format,
        )

        # Convert back to torch tensors
        Geq = torch.tensor(Geq_np, dtype=torch.float32, device=self.device)
        return Geq, khi, zetax, zetay

    def get_Geq_obs(
        self,
        rho,
        ux,
        uy,
        T,
        khi,
        zetax,
        zetay,
        obs_mask,
        use_dense_inv=True,
        sparse_format="csr",
    ):
        """
        Compute G equilibrium with Obstacle handling.
        Wraps levermore_Geq_Obs.
        """
        rho_np = detach(rho) if not isinstance(rho, np.ndarray) else rho
        ux_np = detach(ux) if not isinstance(ux, np.ndarray) else ux
        uy_np = detach(uy) if not isinstance(uy, np.ndarray) else uy
        T_np = detach(T) if not isinstance(T, np.ndarray) else T
        khi = detach(khi) if not isinstance(khi, np.ndarray) else khi
        zetax = detach(zetax) if not isinstance(zetax, np.ndarray) else zetax
        zetay = detach(zetay) if not isinstance(zetay, np.ndarray) else zetay

        Geq_np, khi, zetax, zetay = levermore_Geq_Obs(
            detach(self.solver.ex),
            detach(self.solver.ey),
            ux_np,
            uy_np,
            T_np,
            rho_np,
            self.solver.Cv,
            self.solver.Qn,
            khi,
            zetax,
            zetay,
            detach(obs_mask),
            use_dense_inv=use_dense_inv,
            sparse_format=sparse_format,
        )

        Geq_obs = torch.tensor(Geq_np, dtype=torch.float32, device=self.device)
        return Geq_obs, khi, zetax, zetay

    def get_Geq_BC(
        self,
        rho,
        ux,
        uy,
        T,
        khi,
        zetax,
        zetay,
        coly,
        val=0,
        use_dense_inv=True,
        sparse_format="csr",
    ):
        """
        Compute G equilibrium for Boundary Conditions (Inlet/Outlet).
        Wraps levermore_Geq_BCs.
        """
        rho_np = detach(rho) if not isinstance(rho, np.ndarray) else rho
        ux_np = detach(ux) if not isinstance(ux, np.ndarray) else ux
        uy_np = detach(uy) if not isinstance(uy, np.ndarray) else uy
        T_np = detach(T) if not isinstance(T, np.ndarray) else T
        khi = detach(khi) if not isinstance(khi, np.ndarray) else khi
        zetax = detach(zetax) if not isinstance(zetax, np.ndarray) else zetax
        zetay = detach(zetay) if not isinstance(zetay, np.ndarray) else zetay

        Geq_np, khi, zetax, zetay = levermore_Geq_BCs(
            detach(self.solver.ex),
            detach(self.solver.ey),
            ux_np,
            uy_np,
            T_np,
            rho_np,
            self.solver.Cv,
            self.solver.Qn,
            khi,
            zetax,
            zetay,
            detach(coly),
            val,
            use_dense_inv=use_dense_inv,
            sparse_format=sparse_format,
        )

        Geq_BC = torch.tensor(Geq_np, dtype=torch.float32, device=self.device)
        return Geq_BC, khi, zetax, zetay
