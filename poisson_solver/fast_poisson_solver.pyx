import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_poisson_solver(np.ndarray[np.float64_t, ndim=2] n_e, 
                        np.ndarray[np.float64_t, ndim=2] n_h, 
                        np.ndarray[np.float64_t, ndim=1] x, 
                        np.ndarray[np.float64_t, ndim=1] y, 
                        double epsilon_0, 
                        np.ndarray[np.float64_t, ndim=2] phi, 
                        double tol, 
                        int max_iter, 
                        np.ndarray[np.float64_t, ndim=2] eps):
    cdef int i, j, iter
    cdef double dx, dy, qe, diff, max_diff
    cdef int nx = x.shape[0]
    cdef int ny = y.shape[0]
    
    # Declare variables for better memory access patterns
    cdef np.ndarray[np.float64_t, ndim=2] rho = np.zeros((ny, nx), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] V = np.zeros((ny, nx), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] V_old = np.zeros((ny, nx), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] epsilon_y1 = np.zeros((ny-2, nx-2), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] epsilon_y2 = np.zeros((ny-2, nx-2), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] epsilon_x1 = np.zeros((ny-2, nx-2), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] epsilon_x2 = np.zeros((ny-2, nx-2), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] denominator = np.zeros((ny-2, nx-2), dtype=np.float64)
    
    # Constants
    qe = -1.60217662e-19  # Elementary charge in Coulombs
    
    # Define the charge density rho = qe * (n_e - n_h)
    for i in range(ny):
        for j in range(nx):
            rho[i, j] = qe * (n_e[i, j] - n_h[i, j])
    
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    # Initialize the potential with the initial guess
    V[:, :] = phi[:, :]
    
    # Set boundary conditions (V = 0 at the boundaries)
    V[:, 0] = 0   # Left boundary
    V[:, -1] = 0  # Right boundary
    V[0, :] = 0   # Bottom boundary
    V[-1, :] = 0  # Top boundary
    
    # Iterative solver (Gauss-Seidel method)
    for iter in range(max_iter):
        # Copy V to V_old
        np.copyto(V_old, V)
        
        # Calculate epsilon at half-grid points and denominator for all inner points
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                epsilon_y1[i-1, j-1] = 0.5 * (eps[i, j] + eps[i+1, j])
                epsilon_y2[i-1, j-1] = 0.5 * (eps[i, j] + eps[i-1, j])
                epsilon_x1[i-1, j-1] = 0.5 * (eps[i, j] + eps[i, j+1])
                epsilon_x2[i-1, j-1] = 0.5 * (eps[i, j] + eps[i, j-1])
                denominator[i-1, j-1] = (epsilon_x1[i-1, j-1] / dx**2 +
                                         epsilon_x2[i-1, j-1] / dx**2 +
                                         epsilon_y1[i-1, j-1] / dy**2 +
                                         epsilon_y2[i-1, j-1] / dy**2)
        
        # Update V in the inner region
        max_diff = 0.0  # To check for convergence
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                V[i, j] = (1 / denominator[i-1, j-1]) * (
                    epsilon_y1[i-1, j-1] * V_old[i+1, j] / dy**2 +
                    epsilon_y2[i-1, j-1] * V_old[i-1, j] / dy**2 +
                    epsilon_x1[i-1, j-1] * V_old[i, j+1] / dx**2 +
                    epsilon_x2[i-1, j-1] * V_old[i, j-1] / dx**2 +
                    rho[i, j] / epsilon_0
                )
                # Calculate the maximum difference for convergence check
                diff = abs(V[i, j] - V_old[i, j])
                if diff > max_diff:
                    max_diff = diff

        # Check for convergence
        if max_diff < tol:
            break
    
    # Calculate the electric field components (negative gradient of potential)
    E_y, E_x = np.gradient(-V, dy, dx)
    
    return E_x, E_y, V, max_diff < tol
