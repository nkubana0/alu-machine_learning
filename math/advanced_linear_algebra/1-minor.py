#!/usr/bin/env python3
"""
Module for 1-minor
"""


def minor(matrix):
    # Check if the input is a list of lists
    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Check if the matrix is non-empty and square
    if len(matrix) == 0 or len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")

    # Function to get the minor of a matrix
    def get_minor(matrix, row, col):
        return [r[:col] + r[col + 1:]
                for r in (matrix[:row] + matrix[row + 1:])]

    # Calculate the minor matrix
    return [[get_minor(matrix, i, j) for j in range(len(matrix))]
            for i in range(len(matrix))]
