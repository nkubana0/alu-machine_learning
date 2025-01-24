import numpy as np

def np_slice(matrix, axes={}):
    """Slices a numpy matrix along specific axes.
    
    Args:
        matrix (numpy.ndarray): The matrix to slice.
        axes (dict): A dictionary where the key is an axis to slice along, and
                     the value is a tuple representing the slice for that axis.
                     
    Returns:
        numpy.ndarray: The sliced matrix.
    """
    # Create a list of slices for each axis
    slices = [slice(None)] * matrix.ndim  # Default slice for all axes
    
    # Update slices based on the axes dictionary
    for axis, slice_params in axes.items():
        slices[axis] = slice(*slice_params)
    
    # Apply the slices to the matrix
    return matrix[tuple(slices)]
