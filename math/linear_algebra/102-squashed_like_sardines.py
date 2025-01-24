#!/usr/bin/env python3
"""
Module `102-squashed_like_sardines`
"""


def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis.

    Args:
        mat1: The first matrix (list of lists of ints/floats).
        mat2: The second matrix (list of lists of ints/floats).
        axis: The axis along which to concatenate (default is 0).

    Returns:
        A new matrix representing the concatenation of mat1 and mat2 along the specified axis,
        or None if the matrices cannot be concatenated.
    """
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None

    # Base case: Axis is 0, concatenate at the top level
    if axis == 0:
        # Ensure both inputs have the same inner dimensions if they are nested
        if isinstance(mat1[0], list) and isinstance(mat2[0], list):
            if len(mat1[0]) != len(mat2[0]):
                return None
        elif isinstance(mat1[0], list) or isinstance(mat2[0], list):
            # If one is a list and the other is not, they cannot be concatenated
            return None
        return mat1 + mat2

    # Recursive case: Axis > 0
    if len(mat1) != len(mat2):
        return None

    result = []
    for sub_mat1, sub_mat2 in zip(mat1, mat2):
        concatenated = cat_matrices(sub_mat1, sub_mat2, axis=axis - 1)
        if concatenated is None:
            return None
        result.append(concatenated)

    return result
