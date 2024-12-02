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

def initialize_wavepacket(sigma, k, grid):
    return np.exp(-grid**2 / (2 * sigma**2)) * np.cos(k * grid)

def advection1d(method, nspace, ntime, tau_rel, params):
    L, c = params
    h = L / nspace
    tau = tau_rel * h / c

    x = np.linspace(-L / 2, L / 2, nspace)
    t = np.linspace(0, tau * ntime, ntime)
    a = np.zeros((nspace, ntime))

    #initial conditions
    sigma, k = 0.2, 35  
    a[:, 0] = initialize_wavepacket(sigma, k, x)

    #calculate A matrix depending on method
    if method == "ftcs":
        A = create_tridiagonal_matrix(nspace, c * tau / (2 * h), 1, -c * tau / (2 * h))
    elif method == "lax":
        A = create_tridiagonal_matrix(nspace, 0.5 * c * tau / h, 0, -0.5 * c * tau / h)
        np.fill_diagonal(A, 1)

    for n in range(1, ntime):
        a[:, n] = np.dot(A, a[:, n - 1])

    if method == "ftcs" and compute_spectral_radius(A) > 1:
        print("Warning: FTCS method is unstable for this configuration.")

    return a, x, t, A
