"""
SOD shock tube case solver.

This module provides the SODSolver class for the SOD shock tube case.
"""

import torch
import numpy as np

from .base import BaseLBSolver
from ..physics_generator import PhysicsGenerator


class SODSolver(BaseLBSolver):
    """
    Solver for SOD shock tube case.

    This solver handles 1D shock tube problems with case-specific
    initial conditions and inline boundary condition application.
    """

    def __init__(
        self,
        X=3001,
        Y=5,
        Qn=9,
        alpha1=1.2,
        alpha01=1.05,
        vuy=2,
        Pr=0.71,
        muy=0.025,
        Uax=0.0,
        Uay=0.0,
        device="cuda",
    ):
        """
        Initialize SOD solver.

        Parameters
        ----------
        X : int, optional
            Grid size in x direction (default: 3001)
        Y : int, optional
            Grid size in y direction (default: 5)
        Qn : int, optional
            Number of discrete velocities (default: 9)
        alpha1 : float, optional
            Relaxation parameter for moderate deviations (default: 1.2)
        alpha01 : float, optional
            Relaxation parameter for small deviations (default: 1.05)
        vuy : float, optional
            Specific heat ratio (default: 2)
        Pr : float, optional
            Prandtl number (default: 0.71)
        muy : float, optional
            Dynamic viscosity (default: 0.025)
        Uax : float, optional
            Velocity shift in x direction (default: 0.0)
        Uay : float, optional
            Velocity shift in y direction (default: 0.0)
        device : str or torch.device, optional
            Device for computation (default: "cuda")
        """
        # Initialize base class with direct muy (not calculated)
        super().__init__(
            X=X,
            Y=Y,
            Qn=Qn,
            vuy=vuy,
            Pr=Pr,
            alpha1=alpha1,
            alpha01=alpha01,
            device=device,
            Uax=Uax,
            Uay=Uay,
            muy=muy,  # SOD uses direct muy parameter
        )

        # SOD-specific: domain center
        self.Lx = self.X // 2

    def streaming(self, F_pos_coll, G_pos_coll):
        """
        Apply streaming operator with inline boundary conditions.

        SOD applies boundary conditions directly within streaming,
        unlike Cylinder which applies BC after streaming.

        Parameters
        ----------
        F_pos_coll : torch.Tensor
            Post-collision F distribution (Q, Y, X)
        G_pos_coll : torch.Tensor
            Post-collision G distribution (Q, Y, X)

        Returns
        -------
        tuple
            (Fi, Gi) streamed distributions with BC applied (Q, Y, X)
        """
        # Apply interpolation and shift (from base class)
        Fo1, Go1 = self.interpolate_domain(F_pos_coll, G_pos_coll)
        Fi, Gi = self.shift_operator(Fo1, Go1)

        # Apply boundary conditions inline
        coly = torch.arange(1, self.Y + 1, device=self.device) - 1
        Gi[:, coly, 0] = Gi[:, coly, 1]
        Gi[:, coly, self.X - 1] = Gi[:, coly, self.X - 2]
        Fi[:, coly, 0] = Fi[:, coly, 1]
        Fi[:, coly, self.X - 1] = Fi[:, coly, self.X - 2]

        del Fo1, Go1
        return Fi, Gi

    def case_1_initial_conditions(self):
        """
        Compute initial conditions for SOD case 1.

        Returns
        -------
        tuple
            (Fi0, Gi0, khi, zetax, zetay) initial distributions and Lagrange multipliers
        """
        gen = PhysicsGenerator(self)

        rho0 = torch.ones((self.Y, self.X), device=self.device)
        ux0 = torch.zeros((self.Y, self.X), device=self.device)
        uy0 = torch.zeros((self.Y, self.X), device=self.device)
        T0 = torch.ones((self.Y, self.X), device=self.device)

        # Case 1: density and temperature jump
        rho0[:, : self.Lx + 1] = 0.5
        rho0[:, self.Lx + 1 :] = 2
        T0[:, : self.Lx + 1] = 0.2
        T0[:, self.Lx + 1 :] = 0.025

        khi0 = np.zeros((self.Y, self.X))
        zetax0 = np.zeros((self.Y, self.X))
        zetay0 = np.zeros((self.Y, self.X))

        # Analytic Feq (Solver)
        Fi0 = self.get_Feq(rho0, ux0, uy0, T0, Q=self.Qn)
        
        # Newton-Raphson Geq (Generator)
        Gi0, khi, zetax, zetay = gen.get_Geq(
            rho0, ux0, uy0, T0, khi0, zetax0, zetay0, sparse_format="csr"
        )

        Fi0 = Fi0.to(self.device)
        Gi0 = Gi0.to(self.device)
        del T0

        return Fi0, Gi0, khi, zetax, zetay

    def case_2_initial_conditions(self):
        """
        Compute initial conditions for SOD case 2.

        Returns
        -------
        tuple
            (Fi0, Gi0, khi, zetax, zetay) initial distributions and Lagrange multipliers
        """
        # Instantiate Generator
        gen = PhysicsGenerator(self)

        rho_max = 1.0
        p_max = 0.2

        ux0 = torch.zeros((self.Y, self.X), device=self.device)
        uy0 = torch.zeros((self.Y, self.X), device=self.device)
        rho0 = torch.ones((self.Y, self.X), device=self.device)

        # Case 2: density and pressure jump
        rho0[:, : self.Lx + 1] = 1 * rho_max
        rho0[:, self.Lx + 1 :] = 0.125 * rho_max

        P0 = torch.zeros((self.Y, self.X), device=self.device)
        P0[:, : self.Lx + 1] = 1.0 * p_max
        P0[:, self.Lx + 1 :] = 0.1 * p_max

        # Compute temperature from pressure
        T0 = P0 / (rho0 * self.R)

        khi0 = np.zeros((self.Y, self.X))
        zetax0 = np.zeros((self.Y, self.X))
        zetay0 = np.zeros((self.Y, self.X))

        # Analytic Feq (Solver)
        Fi0 = self.get_Feq(rho0, ux0, uy0, T0, Q=self.Qn)
        
        # Newton-Raphson Geq (Generator)
        Gi0, khi, zetax, zetay = gen.get_Geq(
            rho0, ux0, uy0, T0, khi0, zetax0, zetay0, sparse_format="csr"
        )

        Fi0 = Fi0.to(self.device)
        Gi0 = Gi0.to(self.device)
        del P0

        return Fi0, Gi0, khi, zetax, zetay
