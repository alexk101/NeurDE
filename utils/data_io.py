"""
Data I/O utilities for loading and saving simulation data.

This module provides functions for reading HDF5 files containing simulation data
for both training stages (equilibrium state and stage 2 data).
"""

import h5py


def load_equilibrium_state(file_path):
    """
    Load equilibrium state data from HDF5 file (Stage 1 training data).

    Loads the macroscopic variables and equilibrium distribution function
    needed for Stage 1 training.

    Parameters
    ----------
    file_path : str
        Path to HDF5 file containing equilibrium state data

    Returns
    -------
    tuple
        (all_rho, all_ux, all_uy, all_T, all_Geq) where:
        - all_rho: Density array
        - all_ux: x-velocity array
        - all_uy: y-velocity array
        - all_T: Temperature array
        - all_Geq: Equilibrium G distribution function array
    """
    with h5py.File(file_path, "r") as f:
        all_rho = f["rho"][:]
        all_ux = f["ux"][:]
        all_uy = f["uy"][:]
        all_T = f["T"][:]
        all_Geq = f["Geq"][:]
        return all_rho, all_ux, all_uy, all_T, all_Geq


def load_data_stage_2(file_path):
    """
    Load Stage 2 training data from HDF5 file.

    Loads the distribution functions and their equilibrium values
    needed for Stage 2 training.

    Parameters
    ----------
    file_path : str
        Path to HDF5 file containing Stage 2 data

    Returns
    -------
    tuple
        (all_F, all_G, all_Feq, all_Geq) where:
        - all_F: F distribution function array (Fi0)
        - all_G: G distribution function array (Gi0)
        - all_Feq: Equilibrium F distribution function array
        - all_Geq: Equilibrium G distribution function array
    """
    with h5py.File(file_path, "r") as f:
        all_F = f["Fi0"][:]
        all_G = f["Gi0"][:]
        all_Feq = f["Feq"][:]
        all_Geq = f["Geq"][:]
        return all_F, all_G, all_Feq, all_Geq
