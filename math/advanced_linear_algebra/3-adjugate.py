#!/usr/bin/env python3
"""
Module 3-adjugate.py
"""


def adjugate(matrix):
    """
    A function that calculates the adjugate of a matrix.
    """

    if not isinstance(matrix, list) or not all(
            isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if len(matrix) == 0 or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    def minor(matrix, i, j):
        """Get the minor of the matrix by removing row i and column j."""
        return [row[:j] + row[j + 1:] for row in (matrix[:i] + matrix[i + 1:])]

    def determinant(matrix):
        """Recursively compute the determinant of a matrix."""
        if len(matrix) == 1:
            return matrix[0][0]
        if len(matrix) == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        det = 0
        for col in range(len(matrix)):
            det += ((-1) ** col) * matrix[0][col] * \
                determinant(minor(matrix, 0, col))
        return det

    # Step 1: Calculate the cofactor matrix
    cofactor_matrix = []
    for i in range(len(matrix)):
        cofactor_row = []
        for j in range(len(matrix)):
            minor_matrix = minor(matrix, i, j)
            cofactor_row.append(((-1) ** (i + j)) * determinant(minor_matrix))
        cofactor_matrix.append(cofactor_row)

    # Step 2: Transpose the cofactor matrix to get the adjugate matrix
    adjugate_matrix = list(map(list, zip(*cofactor_matrix)))

    return adjugate_matrix
