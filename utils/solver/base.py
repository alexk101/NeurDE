"""
Base Lattice Boltzmann solver class.

This module provides the base solver class with all common functionality
shared across Cylinder and SOD cases.
"""

import torch
import torch.nn as nn
import numpy as np

from ..phys.getFeq import F_pop_torch
from ..phys.getGeq import levermore_Geq
from ..core import detach


class BaseLBSolver(nn.Module):
    """
    Base class for Lattice Boltzmann solvers.

    Contains all common functionality shared between Cylinder and SOD cases.
    Case-specific functionality should be implemented in subclasses.

    This class should not be instantiated directly. Use CylinderSolver or SODSolver instead.
    """

    def __init__(
        self,
        X,
        Y,
        Qn=9,
        vuy=1.4,
        Pr=0.71,
        alpha1=1.35,
        alpha01=1.05,
        device="cuda",
        Uax=0.0,
        Uay=0.0,
        muy=None,  # If None, will be calculated (Cylinder), else used directly (SOD)
    ):
        """
        Initialize base solver.

        Parameters
        ----------
        X : int
            Grid size in x direction
        Y : int
            Grid size in y direction
        Qn : int, optional
            Number of discrete velocities (default: 9)
        vuy : float, optional
            Specific heat ratio (default: 1.4)
        Pr : float, optional
            Prandtl number (default: 0.71)
        alpha1 : float, optional
            Relaxation parameter for moderate deviations (default: 1.35)
        alpha01 : float, optional
            Relaxation parameter for small deviations (default: 1.05)
        device : str or torch.device, optional
            Device for computation (default: "cuda")
        Uax : float, optional
            Velocity shift in x direction (default: 0.0)
        Uay : float, optional
            Velocity shift in y direction (default: 0.0)
        muy : float, optional
            Dynamic viscosity. If None, will be calculated from Re/radius (Cylinder).
            If provided, used directly (SOD). (default: None)
        """
        super(BaseLBSolver, self).__init__()
        self.X = X
        self.Y = Y
        self.Qn = Qn
        self.alpha1 = alpha1
        self.alpha01 = alpha01
        self.vuy = vuy
        self.Pr = Pr
        self.device = device
        self.muy = muy  # May be set later by subclasses

        # Velocity shift (may be overridden by subclasses)
        self.Uax = Uax
        self.Uay = Uay

        # Initialize velocity vectors
        ex_values = [1, 0, -1, 0, 1, -1, -1, 1, 0]
        ey_values = [0, 1, 0, -1, 1, 1, -1, -1, 0]

        self.ex = (
            torch.tensor(ex_values, dtype=torch.float32, device=self.device) + self.Uax
        )
        self.ey = (
            torch.tensor(ey_values, dtype=torch.float32, device=self.device) + self.Uay
        )
        self.ex1 = torch.tensor(ex_values, dtype=torch.float32, device=self.device)
        self.ey1 = torch.tensor(ey_values, dtype=torch.float32, device=self.device)
        del ex_values, ey_values

        # Compute derived quantities
        self.get_derived_quantities()

    def get_derived_quantities(self):
        """
        Compute derived physical quantities and pre-compute indices.

        This method is called during initialization and sets up:
        - Thermodynamic constants (Cv, Cp, R, iCv)
        - Pre-computed velocity products (ex2, ey2, exey)
        - Streaming indices (Y_indices, X_indices, q_indices)
        """
        self.iCv = self.vuy - 1
        self.Cp = self.vuy / self.iCv
        self.Cv = 1 / self.iCv
        self.R = self.Cp - self.Cv  # gas constant

        # Pre-calculate velocity products for efficiency
        self.ex2 = self.ex**2
        self.ey2 = self.ey**2
        self.exey = self.ex * self.ey

        # Compute shifts for streaming
        self.shifts_y = -self.ey1.int()
        self.shifts_x = self.ex1.int()

        # Pre-compute streaming indices
        self.q_indices = torch.arange(self.Qn, device=self.device)[:, None, None]
        Y_indices = (
            torch.arange(self.Y, device=self.device)[None, :, None]
            - self.shifts_y[:, None, None]
        ) % self.Y
        X_indices = (
            torch.arange(self.X, device=self.device)[None, None, :]
            - self.shifts_x[:, None, None]
        ) % self.X

        self.Y_indices = Y_indices.expand(self.Qn, self.Y, self.X)
        self.X_indices = X_indices.expand(self.Qn, self.Y, self.X)

    def dot_prod(self, ux, uy):
        """
        Compute dot product of velocity vector with itself.

        Parameters
        ----------
        ux : torch.Tensor
            x-component of velocity
        uy : torch.Tensor
            y-component of velocity

        Returns
        -------
        torch.Tensor
            ux^2 + uy^2
        """
        return ux**2 + uy**2

    def get_energy_from_temp(self, ux, uy, T):
        """
        Compute energy density from temperature and velocity.

        Parameters
        ----------
        ux : torch.Tensor
            x-component of velocity
        uy : torch.Tensor
            y-component of velocity
        T : torch.Tensor
            Temperature

        Returns
        -------
        torch.Tensor
            Energy density E = T * Cv + (ux^2 + uy^2) / 2
        """
        uu = self.dot_prod(ux, uy)
        return T * self.Cv + uu / 2

    def get_temp_from_energy(self, ux, uy, E):
        """
        Compute temperature from energy density and velocity.

        Parameters
        ----------
        ux : torch.Tensor
            x-component of velocity
        uy : torch.Tensor
            y-component of velocity
        E : torch.Tensor
            Energy density

        Returns
        -------
        torch.Tensor
            Temperature T = iCv * (E - (ux^2 + uy^2) / 2)
        """
        uu = self.dot_prod(ux, uy)
        T = self.iCv * (E - uu / 2)
        # Clamp temperature to avoid negative or invalid values
        T = torch.clamp(T, min=1e-6)
        return T

    def get_heat_flux_Maxwellian(self, rho, ux, uy, E, T):
        """
        Compute Maxwellian heat flux.

        Parameters
        ----------
        rho : torch.Tensor
            Density
        ux : torch.Tensor
            x-component of velocity
        uy : torch.Tensor
            y-component of velocity
        E : torch.Tensor
            Energy density
        T : torch.Tensor
            Temperature

        Returns
        -------
        tuple
            (qx, qy) heat flux components
        """
        H = E + T
        rhoH2 = 2 * rho * H
        qx = rhoH2 * ux
        qy = rhoH2 * uy
        del H, rhoH2
        return qx, qy

    def get_density(self, F):
        """
        Compute density from F distribution function.

        Parameters
        ----------
        F : torch.Tensor
            F distribution function (Q, Y, X)

        Returns
        -------
        torch.Tensor
            Density (Y, X)
        """
        rho = torch.sum(F, dim=0).to(self.device)
        # Avoid division by zero - ensure minimum density
        rho = torch.clamp(rho, min=1e-6)
        return rho

    def get_momentum(self, F):
        """
        Compute momentum from F distribution function.

        Parameters
        ----------
        F : torch.Tensor
            F distribution function (Q, Y, X)

        Returns
        -------
        tuple
            (rho_ux, rho_uy) momentum components (Y, X)
        """
        rho_ux = torch.tensordot(self.ex, F, dims=([0], [0])).to(self.device)
        rho_uy = torch.tensordot(self.ey, F, dims=([0], [0])).to(self.device)
        return rho_ux, rho_uy

    def get_energy_density(self, G):
        """
        Compute energy density from G distribution function.

        Parameters
        ----------
        G : torch.Tensor
            G distribution function (Q, Y, X)

        Returns
        -------
        torch.Tensor
            Energy density (Y, X)
        """
        rho_E = torch.sum(G, dim=0).to(self.device)
        return rho_E

    def get_macroscopic(self, F, G):
        """
        Compute macroscopic variables from distribution functions.

        Parameters
        ----------
        F : torch.Tensor
            F distribution function (Q, Y, X)
        G : torch.Tensor
            G distribution function (Q, Y, X)

        Returns
        -------
        tuple
            (rho, ux, uy, E) macroscopic variables (Y, X)
        """
        rho = self.get_density(F)
        inv_rho = 1 / rho
        rho_ux, rho_uy = self.get_momentum(F)
        ux = rho_ux * inv_rho
        uy = rho_uy * inv_rho
        E = self.get_energy_density(G) * 0.5 * inv_rho
        del inv_rho, rho_ux, rho_uy
        return rho, ux, uy, E

    def get_w(self, T):
        """
        Compute weights for G distribution function.

        Parameters
        ----------
        T : torch.Tensor
            Temperature (Y, X)

        Returns
        -------
        torch.Tensor
            Weights (Q, Y, X)
        """
        w = torch.zeros((self.Qn, self.Y, self.X)).to(self.device)
        one_minus_T = 1 - T
        w[:4, :, :] = one_minus_T * T * 0.5
        w[4:8, :, :] = T**2 * 0.25
        w[8, :, :] = one_minus_T**2
        del one_minus_T
        return w

    def get_relaxation_time(self, rho, T, F, Feq):
        """
        Compute relaxation times for collision operator.

        Uses adaptive relaxation based on deviation from equilibrium.

        Parameters
        ----------
        rho : torch.Tensor
            Density (Y, X)
        T : torch.Tensor
            Temperature (Y, X)
        F : torch.Tensor
            F distribution function (Q, Y, X)
        Feq : torch.Tensor
            Equilibrium F distribution function (Q, Y, X)

        Returns
        -------
        tuple
            (omega, omegaT) relaxation frequencies (Q, Y, X)
        """
        # Compute base relaxation time
        # muy may be calculated (Cylinder) or provided directly (SOD)
        tau_DL = self.muy / (rho * T) + 0.5

        # Compute deviation from equilibrium
        # Add small epsilon to Feq to prevent division by zero
        Feq_safe = Feq + 1e-10
        diff = torch.abs(F - Feq) / Feq_safe
        EPS = diff.mean(dim=0)

        # Adaptive relaxation parameter
        alpha = torch.ones_like(EPS)
        alpha = torch.where(EPS < 0.01, torch.tensor(1.0, device=EPS.device), alpha)
        alpha = torch.where(
            EPS < 0.1, torch.tensor(self.alpha01, device=EPS.device), alpha
        )
        alpha = torch.where(
            EPS < 1, torch.tensor(self.alpha1, device=EPS.device), alpha
        )
        alpha = torch.where(EPS >= 1, (1 / tau_DL).clone().detach(), alpha)

        tau_EPS = alpha * tau_DL
        tau = tau_EPS.reshape(1, self.Y, self.X).expand(self.Qn, self.Y, self.X)
        tauT = 0.5 + (tau - 0.5) / self.Pr
        omega = 1 / tau
        omegaT = 1 / tauT
        return omega, omegaT

    def get_Feq(self, rho, ux, uy, T, Q=None):
        """
        Compute equilibrium F distribution function.

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
        Q : int, optional
            Number of discrete velocities. If None, uses self.Qn (default: None)

        Returns
        -------
        torch.Tensor
            Equilibrium F distribution function (Q, Y, X)
        """
        if Q is None:
            Q = self.Qn
        Feq = F_pop_torch.compute_Feq(rho, ux, self.Uax, uy, self.Uay, T, Q=Q)
        return Feq

    def get_Geq_Newton_solver(
        self, rho, ux, uy, T, khi, zetax, zetay, sparse_format="csr"
    ):
        """
        Compute equilibrium G distribution function using Newton-Raphson method.

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
        khi : numpy.ndarray
            Lagrange multiplier for density (Y, X)
        zetax : numpy.ndarray
            Lagrange multiplier for x-velocity (Y, X)
        zetay : numpy.ndarray
            Lagrange multiplier for y-velocity (Y, X)
        sparse_format : str, optional
            Sparse matrix format for multinv: "csr" or "csc" (default: "csr")

        Returns
        -------
        tuple
            (Geq, khi, zetax, zetay) where Geq is torch.Tensor (Q, Y, X)
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
            detach(self.ex),
            detach(self.ey),
            ux_np,
            uy_np,
            T_np,
            rho_np,
            self.Cv,
            self.Qn,
            khi,
            zetax,
            zetay,
            sparse_format=sparse_format,
        )

        # Convert back to torch tensors
        Geq = torch.tensor(Geq_np, dtype=torch.float32, device=self.device)
        return Geq, khi, zetax, zetay

    def get_maxwellian_pressure_tensor(self, rho, ux, uy, T):
        """
        Compute Maxwellian pressure tensor components.

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
        tuple
            (P_Maxw_xx, P_Maxw_yy, P_Maxw_xy) pressure tensor components (Y, X)
        """
        momentumx = rho * ux
        momentumy = rho * uy
        rhoT = rho * T
        P_Maxw_xx = momentumx * ux + rhoT
        P_Maxw_yy = momentumy * uy + rhoT
        P_Maxw_xy = momentumx * uy
        return P_Maxw_xx, P_Maxw_yy, P_Maxw_xy

    def get_pressure_tensor(self, F):
        """
        Compute pressure tensor from F distribution function.

        Parameters
        ----------
        F : torch.Tensor
            F distribution function (Q, Y, X)

        Returns
        -------
        tuple
            (P_xx, P_yy, P_xy) pressure tensor components (Y, X)
        """
        P_xx = torch.tensordot(self.ex2, F, dims=([0], [0])).to(self.device)
        P_yy = torch.tensordot(self.ey2, F, dims=([0], [0])).to(self.device)
        P_xy = torch.tensordot(self.exey, F, dims=([0], [0])).to(self.device)
        return P_xx, P_yy, P_xy

    def get_pressure(self, T, rho):
        """
        Compute pressure from temperature and density.

        Parameters
        ----------
        T : torch.Tensor
            Temperature (Y, X)
        rho : torch.Tensor
            Density (Y, X)

        Returns
        -------
        torch.Tensor
            Pressure P = R * rho * T (Y, X)
        """
        P = self.R * rho * T
        return P

    def get_qs(self, F, rho, ux, uy, T):
        """
        Compute non-equilibrium heat flux components.

        Parameters
        ----------
        F : torch.Tensor
            F distribution function (Q, Y, X)
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
        tuple
            (qsx, qsy) non-equilibrium heat flux components (Y, X)
        """
        P_eqxx, P_eqyy, P_eqxy = self.get_maxwellian_pressure_tensor(rho, ux, uy, T)
        P_xx, P_yy, P_xy = self.get_pressure_tensor(F)
        diff_xy = P_xy - P_eqxy
        qsx = 2 * ux * (P_xx - P_eqxx) + 2 * uy * diff_xy
        qsy = 2 * uy * (P_yy - P_eqyy) + 2 * ux * diff_xy
        return qsx, qsy

    def from_macro_to_lattice_Gis(self, F, rho, ux, uy, T):
        """
        Convert macroscopic variables to lattice G distribution function.

        Parameters
        ----------
        F : torch.Tensor
            F distribution function (Q, Y, X)
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
            G distribution function (Q, Y, X)
        """
        w = self.get_w(T)
        qsx, qsy = self.get_qs(F, rho, ux, uy, T)
        Gis = (
            w
            * (qsx * self.ex[:, None, None] + qsy * self.ey[:, None, None])
            / T[None, :, :]
        )
        return Gis

    def interpolate_domain(self, Fo, Go):
        """
        Apply inverse distance interpolation for velocity shift.

        Parameters
        ----------
        Fo : torch.Tensor
            F distribution before interpolation (Q, Y, X)
        Go : torch.Tensor
            G distribution before interpolation (Q, Y, X)

        Returns
        -------
        tuple
            (Fo1, Go1) interpolated distributions (Q, Y, X)
        """
        div = 1 + 2 * self.Uax
        Fo1 = torch.zeros((self.Qn, self.Y, self.X)).to(self.device)
        Go1 = torch.zeros((self.Qn, self.Y, self.X)).to(self.device)
        Fo1[:, :, 1 : self.X] = (
            Fo[:, :, 1 : self.X] * (1 - self.Uax) + Fo[:, :, 0 : self.X - 1] * self.Uax
        )
        Go1[:, :, 1 : self.X] = (
            Go[:, :, 1 : self.X] * (1 - self.Uax) + Go[:, :, 0 : self.X - 1] * self.Uax
        )
        Fo1[:, :, 0] = (Fo[:, :, 1] * self.Uax + Fo[:, :, 0] * (1 + self.Uax)) / div
        Go1[:, :, 0] = (Go[:, :, 1] * self.Uax + Go[:, :, 0] * (1 + self.Uax)) / div
        return Fo1, Go1

    def collision(self, F, G, Feq, Geq, rho, ux, uy, T):
        """
        Apply collision operator.

        Parameters
        ----------
        F : torch.Tensor
            F distribution function (Q, Y, X)
        G : torch.Tensor
            G distribution function (Q, Y, X)
        Feq : torch.Tensor
            Equilibrium F distribution function (Q, Y, X)
        Geq : torch.Tensor
            Equilibrium G distribution function (Q, Y, X)
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
        tuple
            (F_pos_collision, G_pos_collision) post-collision distributions (Q, Y, X)
        """
        omega, omegaT = self.get_relaxation_time(rho, T, F, Feq)
        Gis = self.from_macro_to_lattice_Gis(F, rho, ux, uy, T)
        F_pos_collision = F - omega * (F - Feq)
        G_pos_collision = G - omega * (G - Geq) + (omega - omegaT) * Gis
        return F_pos_collision, G_pos_collision

    def shift_operator(self, F, G):
        """
        Apply shift operator for streaming.

        Parameters
        ----------
        F : torch.Tensor
            F distribution function (Q, Y, X)
        G : torch.Tensor
            G distribution function (Q, Y, X)

        Returns
        -------
        tuple
            (Fi, Gi) shifted distributions (Q, Y, X)
        """
        Fi = F[self.q_indices, self.Y_indices, self.X_indices]
        Gi = G[self.q_indices, self.Y_indices, self.X_indices]
        return Fi, Gi

    def streaming(self, F_pos_coll, G_pos_coll):
        """
        Apply streaming operator.

        This is a base implementation. Subclasses should override if they need
        case-specific behavior (e.g., inline BC application for SOD).

        Parameters
        ----------
        F_pos_coll : torch.Tensor
            Post-collision F distribution (Q, Y, X)
        G_pos_coll : torch.Tensor
            Post-collision G distribution (Q, Y, X)

        Returns
        -------
        tuple
            (Fi, Gi) streamed distributions (Q, Y, X)
        """
        Fo1, Go1 = self.interpolate_domain(F_pos_coll, G_pos_coll)
        Fi, Gi = self.shift_operator(Fo1, Go1)
        return Fi, Gi
