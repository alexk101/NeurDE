"""
Unified F equilibrium distribution function computation.

This module provides unified implementations of the F equilibrium distribution
functions used across Cylinder, Cylinder_faster, and SOD_shock_tube cases.

Usage by case:
-------------
Cylinder:
    - compute_Feq: Use defaults (Q=9, implicit)
    - compute_Feq_obstacle: Use defaults (Q=9, implicit)
    - compute_Feq_BC: Use defaults (Q=9, implicit)
    - forward: Use defaults (Q=9, implicit)

Cylinder_faster:
    - compute_Feq: Use defaults (Q=9, implicit)
    - compute_Feq_obstacle: Use defaults (Q=9, implicit)
    - compute_Feq_BC: Use defaults (Q=9, implicit)
    - forward: Use defaults (Q=9, implicit)

SOD_shock_tube:
    - compute_Feq: Use Q parameter explicitly (e.g., Q=self.Qn)
    - compute_Feq_obstacle: Not used
    - compute_Feq_BC: Not used
    - forward: Use Q parameter explicitly
"""

import torch
import torch.nn as nn


class F_pop_torch(nn.Module):
    def __init__(self):
        super(F_pop_torch, self).__init__()
        self.Q = 9

    @staticmethod
    def _compute_phi(ux_diff, uy_diff, T):
        ux_diff_sq = ux_diff**2
        uy_diff_sq = uy_diff**2
        return {
            "mx": (-ux_diff + ux_diff_sq + T) * 0.5,
            "my": (-uy_diff + uy_diff_sq + T) * 0.5,
            "0x": 1 - (ux_diff_sq + T),
            "0y": 1 - (uy_diff_sq + T),
            "px": (ux_diff + ux_diff_sq + T) * 0.5,
            "py": (uy_diff + uy_diff_sq + T) * 0.5,
        }

    @staticmethod
    def _compute_feq_core(rho, Phi, shape, device, Q=9):
        Feq = torch.zeros((Q, *shape), device=device)
        Feq[0] = rho * Phi["px"] * Phi["0y"]
        Feq[1] = rho * Phi["0x"] * Phi["py"]
        Feq[2] = rho * Phi["mx"] * Phi["0y"]
        Feq[3] = rho * Phi["0x"] * Phi["my"]
        Feq[4] = rho * Phi["px"] * Phi["py"]
        Feq[5] = rho * Phi["mx"] * Phi["py"]
        Feq[6] = rho * Phi["mx"] * Phi["my"]
        Feq[7] = rho * Phi["px"] * Phi["my"]
        Feq[8] = rho * Phi["0x"] * Phi["0y"]
        return Feq

    @staticmethod
    def compute_Feq(rho, ux, Uax, uy, Uay, T, Q=9):
        ux_diff = ux - Uax
        uy_diff = uy - Uay
        Phi = F_pop_torch._compute_phi(
            ux_diff, uy_diff, T
        )  # Call static method with class name
        return F_pop_torch._compute_feq_core(rho, Phi, T.shape, T.device, Q)

    @staticmethod
    def compute_Feq_obstacle(rho, ux, Uax, uy, Uay, T, obstacle, Q=9):
        ux_diff = ux[obstacle] - Uax
        uy_diff = uy[obstacle] - Uay
        Phi = F_pop_torch._compute_phi(ux_diff, uy_diff, T[obstacle])
        return F_pop_torch._compute_feq_core(
            rho[obstacle], Phi, (obstacle.sum(),), T.device, Q
        )

    @staticmethod
    def compute_Feq_BC(rho, ux, Uax, uy, Uay, T, row, col, Q=9):
        ux_diff = ux[row, col] - Uax
        uy_diff = uy[row, col] - Uay
        Phi = F_pop_torch._compute_phi(ux_diff, uy_diff, T[row, col])
        return F_pop_torch._compute_feq_core(
            rho[row, col], Phi, (row.shape[0],), T.device, Q
        )

    def forward(self, rho, ux, Uax, uy, Uay, T, obstacle=None, bc_indices=None, Q=9):
        if obstacle is None and bc_indices is None:
            return self.compute_Feq(rho, ux, Uax, uy, Uay, T, Q)  # Call static method
        elif obstacle is not None:
            return self.compute_Feq_obstacle(
                rho, ux, Uax, uy, Uay, T, obstacle, Q
            )  # Call static method
        elif bc_indices is not None:
            row, col = bc_indices
            return self.compute_Feq_BC(
                rho, ux, Uax, uy, Uay, T, row, col, Q
            )  # Call static method
        else:
            obstacle_Feq = self.compute_Feq_obstacle(
                rho, ux, Uax, uy, Uay, T, obstacle, Q
            )
            row, col = bc_indices
            bc_Feq = self.compute_Feq_BC(rho, ux, Uax, uy, Uay, T, row, col, Q)
            return obstacle_Feq, bc_Feq
