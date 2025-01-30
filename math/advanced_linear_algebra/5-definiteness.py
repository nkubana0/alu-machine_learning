#!/usr/bin/env python3
"""
Module for definiteness
"""

import numpy as np


def definiteness(matrix):
    """
    Definiteness of a matrix.
    """
    # Check if matrix is a numpy ndarray
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # Check if matrix is square and not empty
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.size == 0:
        return None

    # Calculate the eigenvalues of the matrix
    eigenvalues = np.linalg.eigvals(matrix)

    # Determine the definiteness based on the eigenvalues
    if np.all(eigenvalues > 0):
        return "Positive definite"
    elif np.all(eigenvalues >= 0):
        return "Positive semi-definite"
    elif np.all(eigenvalues < 0):
        return "Negative definite"
    elif np.all(eigenvalues <= 0):
        return "Negative semi-definite"
    else:
        return "Indefinite"
