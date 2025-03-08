#!/usr/bin/env python3
"""
Function that performs a valid convolution on grayscale images
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Function that performs a valid convolution on grayscale images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    new_h, new_w = h - kh + 1, w - kw + 1
    output = np.zeros((m, new_h, new_w))

    for i in range(new_h):
        for j in range(new_w):
            output[:, i, j] = np.sum(
                images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2))

    return output
