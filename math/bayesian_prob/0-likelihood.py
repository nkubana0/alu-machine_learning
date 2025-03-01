#!/usr/bin/env python3
"""Calculates the likelihood of obtaining the data
given various probabilities."""
import numpy as np
from scipy.stats import binom


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining observed
    data for various probabilities.

    Parameters:
    x (int): Number of patients with severe side effects.
    n (int): Total number of patients observed.
    P (numpy.ndarray): 1D array of hypothetical probabilities.

    Returns:
    numpy.ndarray: 1D array of likelihoods.

    Raises:
    ValueError: If n is not a positive integer,
    x is not a non-negative integer,
    x is greater than n, or P contains values outside [0, 1].
    TypeError: If P is not a 1D numpy.ndarray.
    """

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    return binom.pmf(x, n, P)
