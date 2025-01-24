#!/usr/bin/env python3

def np_slice(matrix, axes={}):
    """Slices a matrix (nested lists) along specific axes.

    Args:
        matrix (list): The matrix to slice, represented as nested lists.
        axes (dict): A dictionary where:
                     - Key: The axis to slice along (int).
                     - Value: A tuple representing the slice for that axis.

    Returns:
        list: A new sliced matrix (nested lists).

    Example:
        >>> mat = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        >>> np_slice(mat, {0: (1, 3), 1: (0, 2)})
        [[4, 5], [7, 8]]
    """
    def slice_recursive(mat, axis, slices):
        """Recursively slices the matrix along the specified axis."""
        if axis == 0:
            # Apply slicing at the current level
            return mat[slices[0]:slices[1]:slices[2]]
        # Recursively slice submatrices
        return [slice_recursive(sub, axis - 1, slices) for sub in mat]

    # Apply slices along specified axes
    for axis, slice_params in axes.items():
        # Normalize slice parameters to (start, stop, step)
        slices = slice_params + (None,) * (3 - len(slice_params))
        matrix = slice_recursive(matrix, axis, slices)

    return matrix
