#!/usr/bin/env python3
"""
Function that performs a same convolution on grayscale images
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Function that performs a same convolution on grayscale images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    padded_images = np.pad(
        images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')
    new_h, new_w = h + 2 * ph - kh + 1, w + 2 * pw - kw + 1
    output = np.zeros((m, new_h, new_w))

    for i in range(new_h):
        for j in range(new_w):
            output[:, i, j] = np.sum(
                padded_images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2))

    return output
