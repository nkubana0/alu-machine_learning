#!/usr/bin/env python3
"""
Module for 0-determinant
"""


def determinant(matrix):
    """
    Calculates the determinant of a square matrix.
    """

    
    # Validate input is a list of lists
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    
    # Validate square matrix
    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")
    
    # Base case: 0x0 matrix
    if matrix == [[]]:
        return 1
    
    # Base case: 1x1 matrix
    if len(matrix) == 1:
        return matrix[0][0]
    
    # Base case: 2x2 matrix
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    # Recursive case: Use Laplace expansion
    det = 0
    for col in range(len(matrix)):
        sub_matrix = [row[:col] + row[col+1:] for row in matrix[1:]]  # Create minor matrix
        det += ((-1) ** col) * matrix[0][col] * determinant(sub_matrix)
    
    return det
