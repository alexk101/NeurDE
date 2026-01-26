"""
Unified Levermore equilibrium distribution function computation.

This module provides unified implementations of the Levermore equilibrium
distribution functions used across Cylinder, Cylinder_faster, and SOD_shock_tube cases.

Usage by case:
-------------
Cylinder:
    - levermore_Geq: Use defaults (zero_threshold=1e-6, sparse_format="csr")
    - levermore_Geq_BCs: Use use_dense_inv=True (default)
    - levermore_Geq_Obs: Use use_dense_inv=True (default)

Cylinder_faster:
    - levermore_Geq: Use defaults (zero_threshold=1e-6), sparse_format="csc"
    - levermore_Geq_BCs: Use defaults (zero_threshold=1e-6), sparse_format="csc", use_dense_inv=False
    - levermore_Geq_Obs: Use defaults (zero_threshold=1e-6), sparse_format="csc", use_dense_inv=False

SOD_shock_tube:
    - levermore_Geq: Use defaults (zero_threshold=1e-6, sparse_format="csr")
    - levermore_Geq_BCs: Not used
    - levermore_Geq_Obs: Not used
"""

import numpy as np
from .multinv import multinv

# ============================================================================
# Constants and Hyperparameters
# ============================================================================

# Default thresholds for numerical stability
DEFAULT_ZERO_THRESHOLD = (
    1e-6  # Used by all cases (Cylinder, Cylinder_faster, SOD_shock_tube)
)

# Convergence threshold for Newton-Raphson iterations
DEFAULT_CONVERGENCE_THRESHOLD = 1e-6

# Relative error epsilon for BC and Obs functions (prevents division by zero)
DEFAULT_RELATIVE_EPS = 1e-12

# Maximum number of Newton-Raphson iterations
MAX_ITERATIONS = 20

# Default sparse matrix format for multinv
DEFAULT_SPARSE_FORMAT = "csr"  # Used by Cylinder and SOD_shock_tube
CYLINDER_FASTER_SPARSE_FORMAT = "csc"  # Used by Cylinder_faster

# Default reshape order for IJ matrix
DEFAULT_RESHAPE_ORDER = "F"


# ============================================================================
# Main Levermore Equilibrium Function
# ============================================================================


def levermore_Geq(
    ex,
    ey,
    ux,
    uy,
    T,
    rho,
    Cv,
    Qn,
    khi,
    zetax,
    zetay,
    zero_threshold=DEFAULT_ZERO_THRESHOLD,
    convergence_threshold=DEFAULT_CONVERGENCE_THRESHOLD,
    sparse_format=DEFAULT_SPARSE_FORMAT,
    reshape_order=DEFAULT_RESHAPE_ORDER,
):
    """
    Calculates the Levermore equilibrium distribution function (optimized).

    This is the main function used by all three cases. The differences are:
    - Cylinder: zero_threshold=1e-6 (default), sparse_format="csr", reshape_order="F"
    - Cylinder_faster: zero_threshold=1e-6 (default), sparse_format="csc", reshape_order="C" (default)
    - SOD_shock_tube: zero_threshold=1e-6 (default), sparse_format="csr", reshape_order="F"

    Parameters
    ----------
    ex, ey : array-like
        Discrete velocity components
    ux, uy : array-like
        Macroscopic velocity components
    T : array-like
        Temperature
    rho : array-like
        Density
    Cv : float
        Specific heat at constant volume
    Qn : int
        Number of discrete velocities
    khi, zetax, zetay : array-like
        Lagrange multipliers (modified in-place)
    zero_threshold : float, optional
        Threshold for zeroing small values (default: 1e-6)
    convergence_threshold : float, optional
        Threshold for convergence check (default: 1e-6)
    sparse_format : str, optional
        Sparse matrix format for multinv: "csr" or "csc" (default: "csr")
    reshape_order : str, optional
        Order for reshaping IJ matrix: "C" or "F" (default: "F")

    Returns
    -------
    Feq : ndarray
        Equilibrium distribution function
    khi, zetax, zetay : ndarray
        Updated Lagrange multipliers
    """
    # Zero out small values for numerical stability
    ux[np.abs(ux) < zero_threshold] = 0
    uy[np.abs(uy) < zero_threshold] = 0
    T[np.abs(T) < zero_threshold] = 0
    rho[np.abs(rho) < zero_threshold] = 0

    Y, X = ux.shape
    Qn = int(Qn)
    ex = ex.squeeze()
    ey = ey.squeeze()

    # Precompute common terms
    uu = ux**2 + uy**2
    E = T * Cv + 0.5 * uu
    H = E + T

    # Precompute weights (optimization from Cylinder/Cylinder_faster)
    w = np.zeros((Qn, Y, X))
    one_minus_T = 1 - T
    w[:4, :, :] = one_minus_T * T * 0.5
    w[4:8, :, :] = T**2 * 0.25
    w[8, :, :] = one_minus_T**2

    f = np.zeros((Qn, Y, X))
    F = np.zeros((3, Y, X))
    J = np.zeros((3, 3, Y, X))

    # Newton-Raphson iteration
    for _ in range(MAX_ITERATIONS):
        # Zero out small Lagrange multipliers
        khi[np.abs(khi) < zero_threshold] = 0
        zetax[np.abs(zetax) < zero_threshold] = 0
        zetay[np.abs(zetay) < zero_threshold] = 0

        # Compute distribution function
        f = w * np.exp(
            khi[None, :, :]
            + zetax[None, :, :] * ex[:, None, None]
            + zetay[None, :, :] * ey[:, None, None]
        )

        # Compute residual vector F (optimization: precompute f_sum for Cylinder_faster)
        f_sum = f.sum(axis=0)
        F[0, :, :] = f_sum - 2 * E
        F[1, :, :] = (ex[:, None, None] * f).sum(axis=0) - 2 * ux * H
        F[2, :, :] = (ey[:, None, None] * f).sum(axis=0) - 2 * uy * H

        # Compute Jacobian matrix J
        J[0, 0, :, :] = f_sum
        J[0, 1, :, :] = np.einsum("q,qyx->yx", ex, f)
        J[0, 2, :, :] = np.einsum("q,qyx->yx", ey, f)
        J[1, 0, :, :] = J[0, 1, :, :]
        J[1, 1, :, :] = np.einsum("q,qyx->yx", ex**2, f)
        J[1, 2, :, :] = np.einsum("q,qyx->yx", ex * ey, f)
        J[2, 0, :, :] = J[0, 2, :, :]
        J[2, 1, :, :] = J[1, 2, :, :]
        J[2, 2, :, :] = np.einsum("q,qyx->yx", ey**2, f)

        # Compute inverse Jacobian using sparse solver
        IJ = multinv(J, sparse_format=sparse_format)
        if reshape_order == "F":
            IJ = IJ.reshape((3, 3, Y, X), order="F")
        else:
            IJ = IJ.reshape((3, 3, Y, X))

        # Store previous values for convergence check
        khi1 = khi.copy()
        zetax1 = zetax.copy()
        zetay1 = zetay.copy()

        # Update Lagrange multipliers using Newton step
        khi -= IJ[0, 0] * F[0] + IJ[0, 1] * F[1] + IJ[0, 2] * F[2]
        zetax -= IJ[1, 0] * F[0] + IJ[1, 1] * F[1] + IJ[1, 2] * F[2]
        zetay -= IJ[2, 0] * F[0] + IJ[2, 1] * F[1] + IJ[2, 2] * F[2]

        # Check convergence
        dkhi = np.abs(khi - khi1)
        dzetax = np.abs(zetax - zetax1)
        dzetay = np.abs(zetay - zetay1)

        mx = np.max(np.array([np.max(dkhi), np.max(dzetax), np.max(dzetay)]))
        if mx < convergence_threshold:
            break

    # Compute final equilibrium distribution
    Feq = (
        w
        * rho[None, :, :]
        * np.exp(
            khi[None, :, :]
            + zetax[None, :, :] * ex[:, None, None]
            + zetay[None, :, :] * ey[:, None, None]
        )
    )

    return Feq, khi, zetax, zetay


# ============================================================================
# Boundary Conditions Function
# ============================================================================


def levermore_Geq_BCs(
    ex,
    ey,
    ux,
    uy,
    T,
    rho,
    Cv,
    Qn,
    khi,
    zetax,
    zetay,
    row,
    col,
    zero_threshold=DEFAULT_ZERO_THRESHOLD,
    convergence_threshold=DEFAULT_CONVERGENCE_THRESHOLD,
    relative_eps=DEFAULT_RELATIVE_EPS,
    sparse_format=DEFAULT_SPARSE_FORMAT,
    use_dense_inv=True,  # Cylinder uses np.linalg.inv, Cylinder_faster uses multinv
):
    """
    Calculates Levermore equilibrium for boundary conditions (optimized).

    Usage by case:
    --------------
    Cylinder:
        use_dense_inv=True (default), zero_threshold=1e-6, sparse_format="csr"
        Uses np.linalg.inv() for matrix inversion

    Cylinder_faster:
        use_dense_inv=False, zero_threshold=1e-5, sparse_format="csc"
        Uses multinv() with CSC format for matrix inversion

    Parameters
    ----------
    ex, ey : array-like
        Discrete velocity components
    ux, uy : array-like
        Macroscopic velocity components
    T : array-like
        Temperature
    rho : array-like
        Density
    Cv : float
        Specific heat at constant volume
    Qn : int
        Number of discrete velocities
    khi, zetax, zetay : array-like
        Lagrange multipliers (modified in-place)
    row, col : array-like
        Boundary condition indices
    zero_threshold : float, optional
        Threshold for zeroing small values (default: 1e-6)
    convergence_threshold : float, optional
        Threshold for convergence check (default: 1e-6)
    relative_eps : float, optional
        Epsilon for relative error calculation (default: 1e-12)
    sparse_format : str, optional
        Sparse matrix format for multinv: "csr" or "csc" (default: "csr")
    use_dense_inv : bool, optional
        If True, use np.linalg.inv() (Cylinder behavior)
        If False, use multinv() (Cylinder_faster behavior) (default: True)

    Returns
    -------
    Feq : ndarray
        Equilibrium distribution function at boundary points
    khi, zetax, zetay : ndarray
        Updated Lagrange multipliers
    """
    # Zero out small values for numerical stability
    ux[np.abs(ux) < zero_threshold] = 0
    uy[np.abs(uy) < zero_threshold] = 0
    T[np.abs(T) < zero_threshold] = 0
    rho[np.abs(rho) < zero_threshold] = 0

    Y, X = ux.shape
    Qn = int(Qn)
    ex = ex.squeeze()
    ey = ey.squeeze()

    # Precompute common terms
    uu = ux**2 + uy**2
    E = T * Cv + 0.5 * uu
    H = E + T

    # Precompute weights for boundary points
    w = np.zeros((Qn, Y, X))
    one_minus_T = 1 - T[row, col]
    w[:4, row, col] = one_minus_T * T[row, col] * 0.5
    w[4:8, row, col] = T[row, col] ** 2 * 0.25
    w[8, row, col] = one_minus_T**2

    R = len(row)
    F = np.zeros((3, R))
    if use_dense_inv:
        # Cylinder: J has shape (3, 3, R)
        J = np.zeros((3, 3, R))
    else:
        # Cylinder_faster: J has shape (3, 3, Y) - indexed with row
        J = np.zeros((3, 3, Y))

    # Newton-Raphson iteration
    for _ in range(MAX_ITERATIONS):
        # Zero out small Lagrange multipliers
        khi[np.abs(khi) < zero_threshold] = 0
        zetax[np.abs(zetax) < zero_threshold] = 0
        zetay[np.abs(zetay) < zero_threshold] = 0

        # Calculate exponential term
        exponent = (
            khi[None, row, col]
            + zetax[None, row, col] * ex[:, None]
            + zetay[None, row, col] * ey[:, None]
        )
        f = w[:, row, col] * np.exp(exponent)

        # Calculate F and J using vectorized operations
        f_sum = f.sum(axis=0)
        F[0] = f_sum - 2 * E[row, col]
        # Precompute ex and ey dot f
        ex_dot_f = np.dot(ex, f)
        ey_dot_f = np.dot(ey, f)

        F[1] = ex_dot_f - 2 * ux[row, col] * H[row, col]
        F[2] = ey_dot_f - 2 * uy[row, col] * H[row, col]

        if use_dense_inv:
            # Cylinder: J has shape (3, 3, R)
            J[0, 0] = f_sum
            J[0, 1] = ex_dot_f
            J[0, 2] = ey_dot_f
            J[1, 0] = ex_dot_f
            J[1, 1] = np.dot(ex**2, f)
            J[1, 2] = np.dot(ex * ey, f)
            J[2, 0] = ey_dot_f
            J[2, 1] = J[1, 2]
            J[2, 2] = np.dot(ey**2, f)
        else:
            # Cylinder_faster: J has shape (3, 3, Y), indexed with row
            J[0, 0, row] = f_sum
            J[0, 1, row] = ex_dot_f
            J[0, 2, row] = ey_dot_f
            J[1, 0, row] = ex_dot_f
            J[1, 1, row] = np.dot(ex**2, f)
            J[1, 2, row] = np.dot(ex * ey, f)
            J[2, 0, row] = ey_dot_f
            J[2, 1, row] = J[1, 2, :]
            J[2, 2, row] = np.dot(ey**2, f)

        # Compute inverse Jacobian
        if use_dense_inv:
            # Cylinder behavior: use np.linalg.inv() with batched transpose
            IJ = np.linalg.inv(J.transpose(2, 0, 1))
            d_params = -np.matmul(IJ, F).transpose(0, 2, 1)[:, 0, :]
        else:
            # Cylinder_faster behavior: use multinv() with J shape (3, 3, Y)
            IJ = multinv(J, sparse_format=sparse_format)
            # IJ has shape (3, 3, Y), we use it directly with row indexing

        # Store previous values for convergence check
        khi_old = khi[row, col].copy()
        zetax_old = zetax[row, col].copy()
        zetay_old = zetay[row, col].copy()

        # Update Lagrange multipliers
        if use_dense_inv:
            khi[row, col] += d_params[:, 0]
            zetax[row, col] += d_params[:, 1]
            zetay[row, col] += d_params[:, 2]
        else:
            # Cylinder_faster uses subtraction with IJ indexed by row
            khi[row, col] = khi[row, col] - (
                IJ[0, 0, row] * F[0] + IJ[0, 1, row] * F[1] + IJ[0, 2, row] * F[2]
            )
            zetax[row, col] = zetax[row, col] - (
                IJ[1, 0, row] * F[0] + IJ[1, 1, row] * F[1] + IJ[1, 2, row] * F[2]
            )
            zetay[row, col] = zetay[row, col] - (
                IJ[2, 0, row] * F[0] + IJ[2, 1, row] * F[1] + IJ[2, 2, row] * F[2]
            )

        # Calculate convergence criteria (relative error)
        dkhi = np.abs((khi[row, col] - khi_old) / (khi_old + relative_eps))
        dzetax = np.abs((zetax[row, col] - zetax_old) / (zetax_old + relative_eps))
        dzetay = np.abs((zetay[row, col] - zetay_old) / (zetay_old + relative_eps))

        max_diff = max(np.max(dkhi), np.max(dzetax), np.max(dzetay))

        if max_diff < convergence_threshold:
            break

    # Calculate final equilibrium distribution
    exponent = (
        khi[None, row, col]
        + zetax[None, row, col] * ex[:, None]
        + zetay[None, row, col] * ey[:, None]
    )
    Feq = w[:, row, col] * rho[row, col] * np.exp(exponent)

    return Feq, khi, zetax, zetay


# ============================================================================
# Obstacle Function
# ============================================================================


def levermore_Geq_Obs(
    ex,
    ey,
    ux,
    uy,
    T,
    rho,
    Cv,
    Qn,
    khi,
    zetax,
    zetay,
    Obs,
    zero_threshold=DEFAULT_ZERO_THRESHOLD,
    convergence_threshold=DEFAULT_CONVERGENCE_THRESHOLD,
    relative_eps=DEFAULT_RELATIVE_EPS,
    sparse_format=DEFAULT_SPARSE_FORMAT,
    use_dense_inv=True,  # Cylinder uses np.linalg.inv, Cylinder_faster uses multinv
):
    """
    Calculates Levermore equilibrium for obstacle points (optimized).

    Usage by case:
    --------------
    Cylinder:
        use_dense_inv=True (default), zero_threshold=1e-5, sparse_format="csr"
        Uses np.linalg.inv() for matrix inversion
        Convergence check uses np.max() of stacked arrays

    Cylinder_faster:
        use_dense_inv=False, zero_threshold=1e-5, sparse_format="csc"
        Uses multinv() with CSC format for matrix inversion
        Convergence check uses np.min() of individual arrays (different logic)

    Parameters
    ----------
    ex, ey : array-like
        Discrete velocity components
    ux, uy : array-like
        Macroscopic velocity components
    T : array-like
        Temperature
    rho : array-like
        Density
    Cv : float
        Specific heat at constant volume
    Qn : int
        Number of discrete velocities
    khi, zetax, zetay : array-like
        Lagrange multipliers (modified in-place)
    Obs : array-like
        Boolean mask for obstacle points
    zero_threshold : float, optional
        Threshold for zeroing small values (default: 1e-6)
    convergence_threshold : float, optional
        Threshold for convergence check (default: 1e-6)
    relative_eps : float, optional
        Epsilon for relative error calculation (default: 1e-12)
    sparse_format : str, optional
        Sparse matrix format for multinv: "csr" or "csc" (default: "csr")
    use_dense_inv : bool, optional
        If True, use np.linalg.inv() (Cylinder behavior)
        If False, use multinv() (Cylinder_faster behavior) (default: True)

    Returns
    -------
    Feq : ndarray
        Equilibrium distribution function at obstacle points
    khi, zetax, zetay : ndarray
        Updated Lagrange multipliers
    """
    # Zero out small values for numerical stability
    ux[np.abs(ux) < zero_threshold] = 0
    uy[np.abs(uy) < zero_threshold] = 0
    T[np.abs(T) < zero_threshold] = 0
    rho[np.abs(rho) < zero_threshold] = 0

    ex = ex.squeeze()
    ey = ey.squeeze()

    uu = ux[Obs] ** 2 + uy[Obs] ** 2
    E = T[Obs] * Cv + uu / 2
    H = E + T[Obs]
    L = np.arange(len(uu))

    # Precompute weights
    w = np.zeros((Qn, len(L)))
    one_minus_T = 1 - T[Obs]
    w[:4, L] = one_minus_T * T[Obs] * 0.5
    w[4:8, L] = T[Obs] ** 2 * 0.25
    w[8, L] = one_minus_T**2

    # Initialize arrays for convergence tracking
    if use_dense_inv:
        # Cylinder: use full-size arrays with order="F"
        dkhi = np.zeros_like(khi, order="F")
        dzetax = np.zeros_like(zetax, order="F")
        dzetay = np.zeros_like(zetay, order="F")
    else:
        # Cylinder_faster: use default order
        dkhi = np.zeros_like(khi)
        dzetax = np.zeros_like(zetax)
        dzetay = np.zeros_like(zetay)

    F = np.zeros((3, len(L)))
    J = np.zeros((3, 3, len(L)))

    # Newton-Raphson iteration
    for _ in range(MAX_ITERATIONS):
        # Zero out small Lagrange multipliers
        khi[np.abs(khi) < zero_threshold] = 0
        zetax[np.abs(zetax) < zero_threshold] = 0
        zetay[np.abs(zetay) < zero_threshold] = 0

        # Compute distribution function with numerical stability
        exponent = (
            khi[Obs]
            + zetax[None, Obs] * ex[:, None]
            + zetay[None, Obs] * ey[:, None]
        )
        f = w * np.exp(exponent)

        # Precompute ex and ey dot f
        ex_dot_f = np.dot(ex, f)
        ey_dot_f = np.dot(ey, f)

        f_sum = f.sum(axis=0)
        F[0, L] = f_sum - 2 * E
        F[1, L] = ex_dot_f - 2 * ux[Obs] * H
        F[2, L] = ey_dot_f - 2 * uy[Obs] * H

        J[0, 0, L] = f_sum
        J[0, 1, L] = ex_dot_f
        J[0, 2, L] = ey_dot_f
        J[1, 0, L] = ex_dot_f
        J[1, 1, L] = np.dot(ex**2, f)
        J[1, 2, L] = np.dot(ex * ey, f)
        J[2, 0, L] = ey_dot_f
        J[2, 1, L] = J[1, 2, L]
        J[2, 2, L] = np.dot(ey**2, f)

        # Compute inverse Jacobian
        if use_dense_inv:
            # Cylinder behavior: use np.linalg.inv()
            IJ = np.linalg.inv(J.transpose(2, 0, 1))
            d_params = -np.matmul(IJ, F).transpose(0, 2, 1)[:, 0, :]

            # Store previous values
            khi_old = khi[Obs].copy()
            zetax_old = zetax[Obs].copy()
            zetay_old = zetay[Obs].copy()

            # Update Lagrange multipliers
            khi[Obs] += d_params[:, 0]
            zetax[Obs] += d_params[:, 1]
            zetay[Obs] += d_params[:, 2]

            # Calculate convergence criteria (Cylinder: max of stacked arrays)
            dkhi[Obs] = np.abs((khi[Obs] - khi_old) / (khi_old + relative_eps))
            dzetax[Obs] = np.abs((zetax[Obs] - zetax_old) / (zetax_old + relative_eps))
            dzetay[Obs] = np.abs((zetay[Obs] - zetay_old) / (zetay_old + relative_eps))

            max_diff = np.max(
                np.stack((dkhi[Obs], dzetax[Obs], dzetay[Obs])), axis=0
            ).max()

        else:
            # Cylinder_faster behavior: use multinv()
            IJ = multinv(J, sparse_format=sparse_format)

            # Store previous values
            khi1 = khi[Obs].copy()
            zetax1 = zetax[Obs].copy()
            zetay1 = zetay[Obs].copy()

            # Pre-compute F slices
            sz = F[0, L].shape[0]
            F_0L = F[0, L]
            F_1L = F[1, L]
            F_2L = F[2, L]

            # Update Lagrange multipliers (Cylinder_faster uses subtraction)
            khi[Obs] -= (
                IJ[0, 0, L].reshape(sz) * F_0L
                + IJ[0, 1, L].reshape(sz) * F_1L
                + IJ[0, 2, L].reshape(sz) * F_2L
            )

            zetax[Obs] -= (
                IJ[1, 0, L].reshape(sz) * F_0L
                + IJ[1, 1, L].reshape(sz) * F_1L
                + IJ[1, 2, L].reshape(sz) * F_2L
            )

            zetay[Obs] -= (
                IJ[2, 0, L].reshape(sz) * F_0L
                + IJ[2, 1, L].reshape(sz) * F_1L
                + IJ[2, 2, L].reshape(sz) * F_2L
            )

            # Calculate convergence criteria (Cylinder_faster: min of individual arrays)
            np.abs((khi[Obs] - khi1) / (khi1 + relative_eps), out=dkhi[Obs])
            np.abs((zetax[Obs] - zetax1) / (zetax1 + relative_eps), out=dzetax[Obs])
            np.abs((zetay[Obs] - zetay1) / (zetay1 + relative_eps), out=dzetay[Obs])

            mkhi = np.min(dkhi[Obs])
            mzetax = np.min(dzetax[Obs])
            mzetay = np.min(dzetay[Obs])

            max_diff = min(mkhi, mzetax, mzetay)

        if max_diff < convergence_threshold:
            break

    # Compute final equilibrium distribution
    Feq = (
        w
        * rho[None, Obs]
        * np.exp(
            khi[None, Obs]
            + zetax[None, Obs] * ex[:, None]
            + zetay[None, Obs] * ey[:, None]
        )
    )

    return Feq, khi, zetax, zetay
