#!/usr/bin/env python3

def np_slice(matrix, axes={}):
    """Slices a matrix (nested lists) along specific axes.
    """
    def slice_recursive(mat, axis, slices):
        """Recursively applies slicing along the specified axis."""
        if axis == 0:
            # Apply slicing at the current depth
            return mat[slices[0]:slices[1]:slices[2]]
        # Apply slicing to each sub-matrix
        return [slice_recursive(sub, axis - 1, slices) for sub in mat]

    # Iterate over axes and apply slices
    for axis, slice_params in axes.items():
        # Normalize the slice parameters
        slices = slice_params + (None,) * (3 - len(slice_params))  # Fill missing values with None
        matrix = slice_recursive(matrix, axis, slices)
    
    return matrix
