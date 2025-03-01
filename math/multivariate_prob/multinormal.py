#!/usr/bin/env python3
"""Multinomal Class"""
import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution.
    """

    def __init__(self, data):
        """
        Initializes the MultiNormal class.

        Parameters:
        data (numpy.ndarray): A 2D numpy array of shape (d, n).

        Raises:
        TypeError: If data is not a 2D numpy.ndarray.
        ValueError: If the number of data points n is less than 2.
        """

        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.d = d
        self.mean = np.mean(data, axis=1, keepdims=True)
        data_centered = data - self.mean
        self.cov = np.dot(data_centered, data_centered.T) / (n - 1)
        self.cov_inv = np.linalg.inv(self.cov)
        self.cov_det = np.linalg.det(self.cov)

    def pdf(self, x):
        """
        Calculates the PDF at a given data point x.

        Parameters:
        x (numpy.ndarray): A numpy array of shape (d, 1).

        Returns:
        float: The value of the PDF at the given data point.

        Raises:
        TypeError: If x is not a numpy.ndarray.
        ValueError: If x does not have the shape (d, 1).
        """

        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        if x.shape != (self.d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(self.d))

        diff = x - self.mean
        exponent = -0.5 * np.dot(np.dot(diff.T, self.cov_inv), diff)
        denominator = np.sqrt((2 * np.pi) ** self.d * self.cov_det)
        pdf_value = (1 / denominator) * np.exp(exponent)

        return pdf_value[0, 0]
