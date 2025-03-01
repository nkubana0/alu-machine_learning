#!/usr/bin/env python3
"""Calculates the correlation matrix from a covariance matrix."""
import numpy as np


def correlation(C):
    """
    Calculates the correlation matrix from a covariance matrix.

    Parameters:
    C (numpy.ndarray): Covariance matrix of shape (d, d).

    Returns:
    numpy.ndarray: Correlation matrix of shape (d, d).

    Raises:
    TypeError: If C is not a numpy.ndarray.
    ValueError: If C is not a 2D square matrix.
    """

    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    std_devs = np.sqrt(np.diag(C))

    if np.any(std_devs == 0):
        raise ValueError("Covariance matrix contains zero variance")

    corr_matrix = C / np.outer(std_devs, std_devs)

    return corr_matrix
