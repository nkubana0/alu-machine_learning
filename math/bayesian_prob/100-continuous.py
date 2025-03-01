#!/usr/bin/env python3
"""
Posterior probability
"""
from scipy import special


def posterior(x, n, p1, p2):
    """Calculates the posterior probability
    p is within [p1, p2] given x and n."""
    # Input validation
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(p1, float) or not (0 <= p1 <= 1):
        raise ValueError("p1 must be a float in the range [0, 1]")
    if not isinstance(p2, float) or not (0 <= p2 <= 1):
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    # Beta CDF for Bayesian posterior
    alpha = x + 1  # Posterior alpha
    beta = n - x + 1  # Posterior beta

    # Compute the cumulative distribution function for Beta at p2 and p1
    cdf_p2 = special.btdtr(alpha, beta, p2)
    cdf_p1 = special.btdtr(alpha, beta, p1)

    # Posterior probability is the difference between the two CDF values
    return cdf_p2 - cdf_p1
