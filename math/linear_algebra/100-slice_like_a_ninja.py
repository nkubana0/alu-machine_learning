#!/usr/bin/env python3

def np_slice(matrix, axes={}):
    """
    Slices a matrix along specific axes.

    Args:
        matrix: The matrix to slice (assumed to be a numpy.ndarray).
        axes: A dictionary where the key is an axis to slice along,
              and the value is a tuple representing the slice to make.

    Returns:
        A sliced version of the input matrix.
    """
    slices = [slice(None)] * \
        len(matrix.shape)  # Create a default slice for all axes
    for axis, slice_tuple in axes.items():
        # Replace the slice for the specified axis
        slices[axis] = slice(*slice_tuple)
    return matrix[tuple(slices)]
