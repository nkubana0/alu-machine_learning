#!/usr/bin/env python3


def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis.
    """
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None

    if axis == 0:
        # Ensure all sub-matrices in the same axis have the same shape
        if len(mat1) > 0 and len(mat2) > 0 and len(mat1[0]) == len(mat2[0]):
            return mat1 + mat2
        return None

    # Concatenate along deeper axes
    if len(mat1) != len(mat2):
        return None

    result = []
    for sub_mat1, sub_mat2 in zip(mat1, mat2):
        concatenated = cat_matrices(sub_mat1, sub_mat2, axis=axis - 1)
        if concatenated is None:
            return None
        result.append(concatenated)

    return result
