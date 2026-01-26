"""
Unified multiple matrix inversion using sparse solvers.

This module provides unified implementations of matrix inversion used across
Cylinder, Cylinder_faster, and SOD_shock_tube cases.

Usage by case:
-------------
Cylinder:
    - Use defaults: multinv(J) or multinv(J, sparse_format="csr")
    - Uses CSR format (Compressed Sparse Row)

Cylinder_faster:
    - Use: multinv(J, sparse_format="csc")
    - Uses CSC format (Compressed Sparse Column) for better performance

SOD_shock_tube:
    - Use defaults: multinv(J) or multinv(J, sparse_format="csr")
    - Uses CSR format (Compressed Sparse Row)
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix


def multinv(M, sparse_format="csr"):
    """
    Compute the inverse of multiple matrices using sparse solvers.

    Computes the inverse of each m x m matrix slice in an array of shape (m, n, ...).
    The first two dimensions must be square (m == n).

    Parameters
    ----------
    M : numpy.ndarray
        Input array of shape (m, n, ...) where m == n. Each slice along the
        remaining dimensions is treated as a separate m x m matrix to invert.
    sparse_format : str, optional
        Sparse matrix format to use. Options:
        - "csr": Compressed Sparse Row format (default, used by Cylinder and SOD_shock_tube)
        - "csc": Compressed Sparse Column format (used by Cylinder_faster)

    Returns
    -------
    numpy.ndarray
        Array of inverses with shape (n, m, ...), matching the input dimensions.

    Raises
    ------
    ValueError
        If the first two dimensions are not square (m != n).
    """
    sn = M.shape
    m = sn[0]
    n = sn[1]
    if m != n:
        raise ValueError("The first two dimensions of M must be m x m slices.")

    # Handle additional dimensions by reshaping
    p = np.prod(M.shape[2:])
    M = M.reshape((int(m), int(n), int(p)), order="F")

    # Build sparse matrix
    # Generate index arrays for sparse matrix construction
    I = np.reshape(np.arange(0, m * p), (m, 1, p), order="F")
    I = np.tile(I, (1, n, 1))
    J = np.reshape(np.arange(0, n * p), (1, n, p), order="F")
    J = np.tile(J, (m, 1, 1))

    # Flatten the arrays
    ii = I.flatten()
    jj = J.flatten()
    mm = M.flatten()

    # Create the sparse matrix in COO format
    sparse_matrix = coo_matrix((mm, (ii, jj)))

    # Convert to requested format
    if sparse_format.lower() == "csc":
        sparse_matrix = sparse_matrix.tocsc()
    else:  # default to CSR
        sparse_matrix = sparse_matrix.tocsr()

    # Prepare RHS as repeated identity matrices
    RHS = np.tile(np.eye(m), (int(p), 1))

    # Solve the system
    X = spsolve(sparse_matrix, RHS)

    # Reshape the result back to the original dimensions with the inverse for each slice
    X = np.reshape(X, (int(n), int(p), int(m)), order="F")
    X = X.transpose(0, 2, 1)
    X = np.reshape(X, ((n, m) + sn[2:]), order="F")

    return X
