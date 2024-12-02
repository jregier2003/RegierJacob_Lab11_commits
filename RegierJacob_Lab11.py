#lab 11  - Jacob Regier
import numpy as np
import matplotlib.pyplot as plt

def create_tridiagonal_matrix(size, below_diag, diag, above_diag):
    """
    Create a tridiagonal matrix of a given size.

    Parameters:
        size (int): Size of the matrix (number of rows/columns).
        below_diag (float): Value to fill the lower diagonal.
        diag (float): Value to fill the main diagonal.
        above_diag (float): Value to fill the upper diagonal.

    Returns:
        numpy.ndarray: A tridiagonal matrix with specified values.
    """
    matrix = np.zeros((size, size))
    np.fill_diagonal(matrix, diag)
    np.fill_diagonal(matrix[1:], below_diag)
    np.fill_diagonal(matrix[:, 1:], above_diag)
    return matrix

def compute_spectral_radius(matrix):
    """
    Compute the spectral radius (largest absolute eigenvalue) of a matrix.

    Parameters:
        matrix (numpy.ndarray): The matrix whose spectral radius is to be calculated.

    Returns:
        float: The spectral radius of the matrix.
    """
    eigenvalues = np.linalg.eigvals(matrix)
    return np.max(np.abs(eigenvalues))

def initialize_wavepacket(sigma, k, grid):
    """
    Initialize a Gaussian wave packet modulated by a cosine.

    Parameters:
        sigma (float): Standard deviation of the Gaussian envelope.
        k (float): Wave number of the cosine modulation.
        grid (numpy.ndarray): Spatial grid on which to initialize the wave packet.

    Returns:
        numpy.ndarray: Initial wave packet values on the grid.
    """
    return np.exp(-grid**2 / (2 * sigma**2)) * np.cos(k * grid)

def advection1d(method, nspace, ntime, tau_rel, params):
    """
    Solve the 1D advection equation using specified numerical methods.

    Parameters:
        method (str): The numerical method to use ("ftcs" or "lax").
        nspace (int): Number of spatial grid points.
        ntime (int): Number of time steps.
        tau_rel (float): Relative time step size (tau_rel = tau / h).
        params (list): A list containing [L, c], where L is the domain length
                       and c is the wave propagation speed.

    Returns:
        tuple: (a, x, t, A)
            a (numpy.ndarray): Solution array of shape (nspace, ntime).
            x (numpy.ndarray): Spatial grid.
            t (numpy.ndarray): Time grid.
            A (numpy.ndarray): Amplification matrix used for the method.
    """
    L, c = params
    h = L / nspace
    tau = tau_rel * h / c

    x = np.linspace(-L / 2, L / 2, nspace)
    t = np.linspace(0, tau * ntime, ntime)
    a = np.zeros((nspace, ntime))

    #initial conditions
    sigma, k = 0.2, 35  
    a[:, 0] = initialize_wavepacket(sigma, k, x)
    print("Initial wavepacket:", a[:, 0])

    #calculate A matrix depending on method
    if method == "ftcs":
        A = create_tridiagonal_matrix(nspace, c * tau / (2 * h), 1, -c * tau / (2 * h))
    elif method == "lax":
        A = np.zeros((nspace, nspace))

        for i in range(nspace):
            A[i, (i - 1) % nspace] = 0.5 
            A[i, (i + 1) % nspace] = 0.5 
        np.fill_diagonal(A, 0)

    for n in range(1, ntime):
        a[:, n] = np.dot(A, a[:, n - 1])

    if method == "ftcs" and compute_spectral_radius(A) > 1:
        print("Warning: FTCS method is unstable for this configuration.")

    return a, x, t, A

def plot_wave(a, x, t, plotskip=50):
    """
    Visualize wave propagation over time.

    Parameters:
        a (numpy.ndarray): Solution array of shape (nspace, ntime).
        x (numpy.ndarray): Spatial grid.
        t (numpy.ndarray): Time grid.
        plotskip (int, optional): Number of time steps to skip between plots. Default is 50.
        filename (str, optional): File name to save the plot. Default is "wave_plot.png".

    Returns:
        None
    """
    fig, ax = plt.subplots()
    yoffset = a[:, 0].max() - a[:, 0].min()
    for i in range(len(t) - 1, -1, -plotskip):
        ax.plot(x, a[:, i] + yoffset * i / plotskip, label=f"t = {t[i]:.3f}")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Amplitude [offset]")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # Test parameters
    nspace, ntime = 300, 500
    tau_rel, L, c = 1, 5, 1
    params = [L, c]

    a, x, t, A = advection1d("lax", nspace, ntime, tau_rel, params)

    #Visualize results
    plot_wave(a, x, t)
