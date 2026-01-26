"""
PyTorch Dataset classes for training.

This module provides unified Dataset classes used across all cases (Cylinder,
Cylinder_faster, and SOD_shock_tube) for both training stages.

Usage by case:
-------------
All cases use the same dataset classes:
    - EquilibriumDataset: Stage 1 training (replaces CylinderDataset, SodDataset_stage1)
    - Stage2Dataset: Stage 2 training (replaces Cylinder_stage2, SodDataset_stage2)
    - RolloutBatchDataset: Rollout training sequences
"""

import torch
from torch.utils.data import Dataset


class EquilibriumDataset(Dataset):
    """
    Dataset for Stage 1 training (equilibrium state prediction).

    This unified class replaces:
    - CylinderDataset (Cylinder case)
    - SodDataset_stage1 (SOD_shock_tube case)

    All cases use identical structure for equilibrium state data.

    Parameters
    ----------
    rho : array-like
        Density values
    ux : array-like
        x-velocity values
    uy : array-like
        y-velocity values
    T : array-like
        Temperature values
    Geq : array-like
        Equilibrium G distribution function values
    """

    def __init__(self, rho, ux, uy, T, Geq):
        # Handle both tensors and numpy arrays
        # Use detach().clone() for tensors to avoid warnings
        if isinstance(rho, torch.Tensor):
            self.rho = rho.detach().clone().to(dtype=torch.float32)
            self.ux = ux.detach().clone().to(dtype=torch.float32)
            self.uy = uy.detach().clone().to(dtype=torch.float32)
            self.T = T.detach().clone().to(dtype=torch.float32)
            self.Geq = Geq.detach().clone().to(dtype=torch.float32)
        else:
            # numpy arrays or lists - use torch.tensor
            self.rho = torch.tensor(rho, dtype=torch.float32)
            self.ux = torch.tensor(ux, dtype=torch.float32)
            self.uy = torch.tensor(uy, dtype=torch.float32)
            self.T = torch.tensor(T, dtype=torch.float32)
            self.Geq = torch.tensor(Geq, dtype=torch.float32)

    def __len__(self):
        return len(self.rho)

    def __getitem__(self, idx):
        return self.rho[idx], self.ux[idx], self.uy[idx], self.T[idx], self.Geq[idx]


class Stage2Dataset(Dataset):
    """
    Dataset for Stage 2 training (distribution function prediction).

    This unified class replaces:
    - Cylinder_stage2 (Cylinder case)
    - SodDataset_stage2 (SOD_shock_tube case)

    All cases use identical structure for Stage 2 data.

    Parameters
    ----------
    F : array-like
        F distribution function values (Fi0)
    G : array-like
        G distribution function values (Gi0)
    Feq : array-like
        Equilibrium F distribution function values
    Geq : array-like
        Equilibrium G distribution function values
    """

    def __init__(self, F, G, Feq, Geq):
        # Handle both tensors and numpy arrays
        # Use detach().clone() for tensors to avoid warnings
        if isinstance(F, torch.Tensor):
            self.F = F.detach().clone().to(dtype=torch.float32)
            self.G = G.detach().clone().to(dtype=torch.float32)
            self.Feq = Feq.detach().clone().to(dtype=torch.float32)
            self.Geq = Geq.detach().clone().to(dtype=torch.float32)
        else:
            # numpy arrays or lists - use torch.tensor
            self.F = torch.tensor(F, dtype=torch.float32)
            self.G = torch.tensor(G, dtype=torch.float32)
            self.Feq = torch.tensor(Feq, dtype=torch.float32)
            self.Geq = torch.tensor(Geq, dtype=torch.float32)

    def __len__(self):
        return len(self.F)

    def __getitem__(self, idx):
        return self.F[idx], self.G[idx], self.Feq[idx], self.Geq[idx]


class RolloutBatchDataset(Dataset):
    """
    Dataset for rollout training sequences.

    Creates sequences of distribution functions for rollout training.
    Used identically across all cases (Cylinder, Cylinder_faster, SOD_shock_tube).

    Parameters
    ----------
    all_Fi : array-like
        All F distribution function sequences
    all_Gi : array-like
        All G distribution function sequences
    all_Feq : array-like
        All equilibrium F distribution function sequences
    all_Geq : array-like
        All equilibrium G distribution function sequences
    number_of_rollout : int
        Length of each rollout sequence
    """

    def __init__(self, all_Fi, all_Gi, all_Feq, all_Geq, number_of_rollout):
        self.all_Fi = all_Fi
        self.all_Gi = all_Gi
        self.all_Feq = all_Feq
        self.all_Geq = all_Geq
        self.number_of_rollout = number_of_rollout
        self.num_sequences = len(all_Fi) - number_of_rollout + 1

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # Handle both numpy arrays (from HDF5) and lists of tensors
        # Input: list/array of (Q, Y, X) arrays or tensors
        # Output: (number_of_rollout, Q, Y, X) tensor
        Fi_slice = self.all_Fi[idx : idx + self.number_of_rollout]
        Gi_slice = self.all_Gi[idx : idx + self.number_of_rollout]
        Feq_slice = self.all_Feq[idx : idx + self.number_of_rollout]
        Geq_slice = self.all_Geq[idx : idx + self.number_of_rollout]

        # Convert to tensor - handles both numpy arrays and tensors
        # If it's a list of tensors, use stack; if numpy arrays, use tensor
        if isinstance(Fi_slice[0], torch.Tensor):
            Fi_sequence = torch.stack(Fi_slice).float()
            Gi_sequence = torch.stack(Gi_slice).float()
            Feq_targets = torch.stack(Feq_slice).float()
            Geq_targets = torch.stack(Geq_slice).float()
        else:
            # numpy arrays or lists - use torch.tensor
            Fi_sequence = torch.tensor(Fi_slice, dtype=torch.float32)
            Gi_sequence = torch.tensor(Gi_slice, dtype=torch.float32)
            Feq_targets = torch.tensor(Feq_slice, dtype=torch.float32)
            Geq_targets = torch.tensor(Geq_slice, dtype=torch.float32)

        return Fi_sequence, Gi_sequence, Feq_targets, Geq_targets
