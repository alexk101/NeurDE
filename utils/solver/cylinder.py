"""
Cylinder case solver.

This module provides the CylinderSolver class for the supersonic flow around
a circular cylinder case. Handles both standard (dense inversion) and faster
(sparse inversion) variants via configuration.
"""

import torch
import numpy as np

from .base import BaseLBSolver
from ..phys.getFeq import F_pop_torch
from ..physics_generator import PhysicsGenerator
from ..core import detach


class CylinderSolver(BaseLBSolver):
    """
    Solver for supersonic flow around a circular cylinder.

    Supports both standard (dense inversion for BC/Obs) and faster
    (sparse inversion throughout) variants via configuration.

    Usage:
    ------
    Standard Cylinder:
        solver = CylinderSolver(..., sparse_format="csr", use_dense_inv=True)

    Cylinder_faster:
        solver = CylinderSolver(..., sparse_format="csc", use_dense_inv=False)
    """

    def __init__(
        self,
        X=500,
        Y=300,
        Qn=9,
        radius=20,
        Ma0=1.7,
        Re=300,
        rho0=1,
        T0=0.2,
        alpha1=1.35,
        alpha01=1.05,
        vuy=1.4,
        Pr=0.71,
        Ns=0.6,
        device="cuda",
        sparse_format="csr",
        use_dense_inv=True,
    ):
        """
        Initialize Cylinder solver.

        Parameters
        ----------
        X : int, optional
            Grid size in x direction (default: 500)
        Y : int, optional
            Grid size in y direction (default: 300)
        Qn : int, optional
            Number of discrete velocities (default: 9)
        radius : float, optional
            Cylinder radius (default: 20)
        Ma0 : float, optional
            Mach number (default: 1.7)
        Re : float, optional
            Reynolds number (default: 300)
        rho0 : float, optional
            Reference density (default: 1)
        T0 : float, optional
            Reference temperature (default: 0.2)
        alpha1 : float, optional
            Relaxation parameter for moderate deviations (default: 1.35)
        alpha01 : float, optional
            Relaxation parameter for small deviations (default: 1.05)
        vuy : float, optional
            Specific heat ratio (default: 1.4)
        Pr : float, optional
            Prandtl number (default: 0.71)
        Ns : float, optional
            Shift parameter (default: 0.6)
        device : str or torch.device, optional
            Device for computation (default: "cuda")
        sparse_format : str, optional
            Sparse matrix format: "csr" for standard, "csc" for faster (default: "csr")
        use_dense_inv : bool, optional
            Use dense inversion for BC/Obs: True for standard, False for faster (default: True)
        """
        # Calculate velocity shift from physics parameters
        cs0 = np.sqrt(vuy * T0)
        U0 = Ma0 * cs0
        Uax = U0 * Ns
        Uay = 0.0

        # Calculate dynamic viscosity from Re and radius
        muy = U0 * 2 * radius / Re

        # Initialize base class
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
            muy=muy,
        )

        # Cylinder-specific parameters
        self.radius = radius
        self.Ma0 = Ma0
        self.Re = Re
        self.rho0 = rho0
        self.T0 = T0
        self.Ns = Ns
        self.U0 = U0

        # Solver configuration
        self.sparse_format = sparse_format
        self.use_dense_inv = use_dense_inv

        # Cylinder center position
        self.CXR = round(self.X / 3) - 1
        self.CYR = round(self.Y / 2) - 1

        # Create obstacle
        self.create_obstacle()

        # Pre-calculate Static Boundary Conditions
        self._cache_static_boundaries()

    def _cache_static_boundaries(self):
        """
        Pre-calculate and cache the equilibrium distributions for Obstacles and BCs.
        These are static (fixed T, u, rho at boundaries), so we don't need Newton
        solver during training.
        """
        # Instantiate the generator
        gen = PhysicsGenerator(self)
        
        # 1. Create dummy/initial fields for the BC calculation
        # We only care about the values at the masked positions
        rho = torch.ones((self.Y, self.X), device=self.device)
        ux = torch.full((self.Y, self.X), self.U0, device=self.device)
        uy = torch.zeros((self.Y, self.X), device=self.device)
        T = torch.full((self.Y, self.X), self.T0, device=self.device)
        
        khi = np.zeros((self.Y, self.X))
        zetax = np.zeros((self.Y, self.X))
        zetay = np.zeros((self.Y, self.X))

        # 2. Obstacle Setup
        ux_obs = torch.where(self.Obs, torch.tensor(0.0, device=self.device), ux)
        uy_obs = torch.where(self.Obs, torch.tensor(0.0, device=self.device), uy)
        T_obs = torch.where(self.Obs, torch.tensor(self.T0, device=self.device), T)
        rho_obs = torch.where(self.Obs, torch.tensor(1.0, device=self.device), rho)

        # 3. Calculate Obstacle Distributions (Newton)
        self.Fi_obs_cyl = self.get_Feq_obs(rho_obs, ux_obs, uy_obs, T_obs)
        self.Gi_obs_cyl, khi_obs, zetax_obs, zetay_obs = gen.get_Geq_obs(
            rho_obs, ux_obs, uy_obs, T_obs, khi, zetax, zetay, self.Obs,
            use_dense_inv=self.use_dense_inv, sparse_format=self.sparse_format
        )

        # 4. Inlet Setup
        # Note: We reuse the Lagrange multipliers (khi_obs...) from the previous step 
        # as a warm start, though for static caching it matters less.
        ux_obs[self.coly, 0] = self.U0
        uy_obs[self.coly, 0] = 0
        T_obs[self.coly, 0] = self.T0
        rho_obs[self.coly, 0] = self.rho0

        # 5. Calculate Inlet Distributions (Newton)
        self.Fi_obs_Inlet = self.get_Feq_BC(rho_obs, ux_obs, uy_obs, T_obs)
        self.Gi_obs_Inlet, _, _, _ = gen.get_Geq_BC(
            rho_obs, ux_obs, uy_obs, T_obs, khi_obs, zetax_obs, zetay_obs, self.coly, 0,
            use_dense_inv=self.use_dense_inv, sparse_format=self.sparse_format
        )
        
        # Ensure they stay on device
        self.Gi_obs_Inlet = self.Gi_obs_Inlet.to(self.device)
        self.Gi_obs_cyl = self.Gi_obs_cyl.to(self.device)

    def create_obstacle(self):
        """
        Create obstacle mask for circular cylinder.

        Creates a boolean mask indicating obstacle cells and sets up
        boundary condition column vectors.
        """
        # Create meshgrid
        y, x = torch.meshgrid(
            torch.arange(self.Y - 1, -1, -1, device=self.device),
            torch.arange(0, self.X, device=self.device),
            indexing="ij",
        )

        # Calculate obstacle mask (circular cylinder)
        Obs = ((x - self.CXR) ** 2 + (y - self.CYR) ** 2) < self.radius**2

        # Filter columns and rows with less than 2 True values
        Obs[:, torch.sum(Obs, dim=0) < 2] = 0
        Obs[torch.sum(Obs, dim=1) < 2, :] = 0

        # Ensure boolean tensor
        self.Obs = Obs.bool()

        # Create column vectors for boundary conditions
        self.colp = torch.arange(1, self.Y - 1, device=self.device)
        self.colx = torch.arange(self.X, device=self.device)
        self.coly = torch.arange(self.Y, device=self.device)

    def get_local_Mach(self, ux, uy, T):
        """
        Compute local Mach number for visualization.

        Parameters
        ----------
        ux : torch.Tensor
            x-component of velocity (Y, X)
        uy : torch.Tensor
            y-component of velocity (Y, X)
        T : torch.Tensor
            Temperature (Y, X)

        Returns
        -------
        torch.Tensor
            Local Mach number (Y, X)
        """
        uu = self.dot_prod(ux, uy)
        cs = self.vuy * T
        return torch.sqrt(uu / cs)

    def get_Feq_obs(self, rho, ux, uy, T):
        """
        Compute equilibrium F distribution for obstacle.

        Parameters
        ----------
        rho : torch.Tensor
            Density (Y, X)
        ux : torch.Tensor
            x-component of velocity (Y, X)
        uy : torch.Tensor
            y-component of velocity (Y, X)
        T : torch.Tensor
            Temperature (Y, X)

        Returns
        -------
        torch.Tensor
            Equilibrium F distribution for obstacle (Q, Y, X)
        """
        Feq_Obs = F_pop_torch.compute_Feq_obstacle(
            rho, ux, self.Uax, uy, self.Uay, T, obstacle=self.Obs
        )
        return Feq_Obs

    def get_Feq_BC(self, rho, ux, uy, T):
        """
        Compute equilibrium F distribution for boundary conditions.

        Parameters
        ----------
        rho : torch.Tensor
            Density (Y, X)
        ux : torch.Tensor
            x-component of velocity (Y, X)
        uy : torch.Tensor
            y-component of velocity (Y, X)
        T : torch.Tensor
            Temperature (Y, X)

        Returns
        -------
        torch.Tensor
            Equilibrium F distribution for BC (Q, Y, X)
        """
        Feq_BC = F_pop_torch.compute_Feq_BC(
            rho, ux, self.Uax, uy, self.Uay, T, self.coly, 0
        )
        return Feq_BC

    def get_obs_distribution(self, rho, ux, uy, T, khi, zetax, zetay):
        """
        Return cached obstacle and boundary condition distributions.
        Args are ignored as BCs are static.
        """
        return self.Fi_obs_cyl, self.Gi_obs_cyl, self.Fi_obs_Inlet, self.Gi_obs_Inlet

    def enforce_Obs_and_BC(
        self, Fi, Gi, Fi_obs_cyl, Gi_obs_cyl, Fi_obs_Inlet, Gi_obs_Inlet
    ):
        """
        Enforce obstacle and boundary conditions.

        Parameters
        ----------
        Fi : torch.Tensor
            F distribution after streaming (Q, Y, X) or batched (B, Q, Y, X)
        Gi : torch.Tensor
            G distribution after streaming (Q, Y, X) or batched (B, Q, Y, X)
        Fi_obs_cyl : torch.Tensor
            F distribution for obstacle (Q, num_obs)
        Gi_obs_cyl : torch.Tensor
            G distribution for obstacle (Q, num_obs)
        Fi_obs_Inlet : torch.Tensor
            F distribution for inlet (Q, Y)
        Gi_obs_Inlet : torch.Tensor
            G distribution for inlet (Q, Y)

        Returns
        -------
        tuple
            (Fi_obs, Gi_obs) with obstacle and BC applied (Q, Y, X) or (B, Q, Y, X)
        """
        # Check if input is batched
        is_batched = Fi.dim() == 4
        
        Fi_obs = Fi.clone()
        Gi_obs = Gi.clone()

        if is_batched:
            # Batched case: Fi is (B, Q, Y, X)
            B = Fi.size(0)
            
            # Obstacle: broadcast the cached obstacle values across batch
            # Fi_obs_cyl is (Q, num_obs), need to apply to all batches
            Fi_obs[:, :, self.Obs] = Fi_obs_cyl.unsqueeze(0).expand(B, -1, -1)
            Gi_obs[:, :, self.Obs] = Gi_obs_cyl.unsqueeze(0).expand(B, -1, -1)

            # Inlet: Fi_obs_Inlet is (Q, Y), need to apply to all batches
            Fi_obs[:, :, self.coly, 0] = Fi_obs_Inlet.unsqueeze(0).expand(B, -1, -1)
            Gi_obs[:, :, self.coly, 0] = Gi_obs_Inlet.unsqueeze(0).expand(B, -1, -1)

            # Outlet (extrapolation)
            Fi_obs[:, :, self.coly, self.X - 1] = Fi_obs[:, :, self.coly, self.X - 2]
            Gi_obs[:, :, self.coly, self.X - 1] = Gi_obs[:, :, self.coly, self.X - 2]

            # Upper wall (extrapolation)
            Fi_obs[:, :, 0, self.colx] = Fi_obs[:, :, 1, self.colx]
            Gi_obs[:, :, 0, self.colx] = Gi_obs[:, :, 1, self.colx]

            # Lower wall (extrapolation)
            Fi_obs[:, :, self.Y - 1, self.colx] = Fi_obs[:, :, self.Y - 2, self.colx]
            Gi_obs[:, :, self.Y - 1, self.colx] = Gi_obs[:, :, self.Y - 2, self.colx]
        else:
            # Single case: original implementation
            # Obstacle
            Fi_obs[:, self.Obs] = Fi_obs_cyl
            Gi_obs[:, self.Obs] = Gi_obs_cyl

            # Inlet
            Fi_obs[:, self.coly, 0] = Fi_obs_Inlet
            Gi_obs[:, self.coly, 0] = Gi_obs_Inlet

            # Outlet (extrapolation)
            Fi_obs[:, self.coly, self.X - 1] = Fi_obs[:, self.coly, self.X - 2]
            Gi_obs[:, self.coly, self.X - 1] = Gi_obs[:, self.coly, self.X - 2]

            # Upper wall (extrapolation)
            Fi_obs[:, 0, self.colx] = Fi_obs[:, 1, self.colx]
            Gi_obs[:, 0, self.colx] = Gi_obs[:, 1, self.colx]

            # Lower wall (extrapolation)
            Fi_obs[:, self.Y - 1, self.colx] = Fi_obs[:, self.Y - 2, self.colx]
            Gi_obs[:, self.Y - 1, self.colx] = Gi_obs[:, self.Y - 2, self.colx]

        return Fi_obs, Gi_obs

    def initial_conditions(self):
        """
        Compute initial conditions for Cylinder case.

        Returns
        -------
        tuple
            (Fi0, Gi0, khi, zetax, zetay) initial distributions and Lagrange multipliers
        """
        gen = PhysicsGenerator(self) # Create temp generator
        
        rho = torch.ones((self.Y, self.X))
        ux = torch.full((self.Y, self.X), self.U0)
        uy = torch.zeros((self.Y, self.X))
        T = torch.full((self.Y, self.X), self.T0)

        khi0 = np.zeros((self.Y, self.X))
        zetax0 = np.zeros((self.Y, self.X))
        zetay0 = np.zeros((self.Y, self.X))

        Fi0 = self.get_Feq(rho, ux, uy, T)

        # Use generator for the heavy lifting
        Gi0, khi, zetax, zetay = gen.get_Geq(
            rho, ux, uy, T, khi0, zetax0, zetay0, sparse_format=self.sparse_format
        )

        Fi0 = Fi0.to(self.device)
        Gi0 = Gi0.to(self.device)

        return Fi0, Gi0, khi, zetax, zetay
