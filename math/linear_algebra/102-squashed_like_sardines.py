#!/usr/bin/env python3
"""
Module `102-squashed_like_sardines`
"""


def cat_matrices(mat1, mat2, axis=0):
    """Concatenate two matrices along a specific axis."""
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None  # Both inputs must be lists
    
    if axis == 0:
        # Check if the inner dimensions match
        if all(isinstance(row, list) for row in mat1) and all(isinstance(row, list) for row in mat2):
            if len(mat1[0]) == len(mat2[0]):  # Inner dimensions must match
                return mat1 + mat2
        return None
    
    if axis > 0:
        # Recursively concatenate sublists
        if len(mat1) != len(mat2):
            return None
        return [
            cat_matrices(sub1, sub2, axis=axis - 1)
            for sub1, sub2 in zip(mat1, mat2)
        ]
    
    return None  # Default case for invalid inputs
