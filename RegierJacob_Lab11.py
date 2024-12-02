#lab 11  - Jacob Regier
import numpy as np
import matplotlib.pyplot as plt

def create_tridiagonal_matrix(size, below_diag, diag, above_diag):
    matrix = np.zeros((size, size))
    np.fill_diagonal(matrix, diag)
    np.fill_diagonal(matrix[1:], below_diag)
    np.fill_diagonal(matrix[:, 1:], above_diag)
    return matrix

def compute_spectral_radius(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    return np.max(np.abs(eigenvalues))

