#!/usr/bin/env python3
"""Calculates the mean and covariance of a dataset."""
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a dataset.

    Parameters:
    X (numpy.ndarray): Dataset of shape (n, d).

    Returns:
    tuple: (mean, cov)
           - mean: numpy.ndarray of shape (1, d)
           - cov: numpy.ndarray of shape (d, d)

    Raises:
    TypeError: If X is not a 2D numpy.ndarray.
    ValueError: If X contains less than 2 data points.
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape

    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0).reshape(1, d)
    X_centered = X - mean
    cov = np.dot(X_centered.T, X_centered) / (n - 1)

    return mean, cov
