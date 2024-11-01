import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import matplotlib.pyplot as plt 

from scipy.special import gammaincc

def potential_theoretical_gaussian(X, Y, sigma):
    # Constants
    pi = np.pi
    r = np.sqrt(X**2 + Y**2)
    # Calculate the potential
    term1 = -1 / (2 * pi) * np.log(r)
    term2 = -1 / (2 * pi) * 0.5 * gammaincc(0, r**2 / (2 * sigma**2))
    
    return term1 + term2

def poisson_2d_matrix_with_permittivity_varying_h(eps, hx, hy, nx, ny):
    """
    Constructs the Poisson matrix A for a 2D problem with varying permittivity and grid spacings hx, hy,
    while imposing Dirichlet boundary conditions (phi = 0) at all boundaries.
    
    Parameters:
    eps : 2D array of shape (nx, ny) representing the permittivity ε(x, y).
    hx  : Grid spacing in the x direction (can be constant or varying).
    hy  : Grid spacing in the y direction (can be constant or varying).
    nx  : Number of grid points in the x direction.
    ny  : Number of grid points in the y direction.
    
    Returns:
    A   : Sparse matrix representing the 2D Poisson operator with permittivity and varying hx, hy.
    """
    N = nx * ny
    A = sp.lil_matrix((N, N))  # Use sparse format for efficiency

    # Flatten the permittivity array
    eps = eps.flatten()
    
    # Compute shifted indices for neighbor grid points in the matrix
    def index(i, j):
        return i * ny + j
    
    # Loop over all points and set up the finite difference equations
    for i in range(nx):
        for j in range(ny):
            ind = index(i, j)
            
            # Boundary conditions (Dirichlet: phi = 0)
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                # For boundary points, set A[ind, ind] = 1 and the rest to 0
                A[ind, :] = 0  # Clear the entire row first
                A[ind, ind] = 1.0
            else:
                # Internal points: use finite difference method to fill the matrix
                eps_x_p = eps[index(i + 1, j)] if i < nx - 1 else 0  # Permittivity for +x neighbor
                eps_x_m = eps[index(i - 1, j)] if i > 0 else 0  # Permittivity for -x neighbor
                eps_y_p = eps[index(i, j + 1)] if j < ny - 1 else 0  # Permittivity for +y neighbor
                eps_y_m = eps[index(i, j - 1)] if j > 0 else 0  # Permittivity for -y neighbor
                
                # Main diagonal element (current point)
                main_diag = -((eps_x_p / hx**2) + (eps_x_m / hx**2) + 
                              (eps_y_p / hy**2) + (eps_y_m / hy**2))
                A[ind, ind] = main_diag
                
                # Off-diagonals for neighbors (set the corresponding neighbors in the matrix)
                if i < nx - 1:
                    A[ind, index(i + 1, j)] = eps_x_p / hx**2  # +x neighbor
                if i > 0:
                    A[ind, index(i - 1, j)] = eps_x_m / hx**2  # -x neighbor
                if j < ny - 1:
                    A[ind, index(i, j + 1)] = eps_y_p / hy**2  # +y neighbor
                if j > 0:
                    A[ind, index(i, j - 1)] = eps_y_m / hy**2  # -y neighbor
    
    return A.tocsc()  # Convert to CSC for efficient LU decomposition

def lu_decomposition_of_poisson_matrix(A):
    """
    Performs LU decomposition on the sparse matrix A.

    Parameters:
    A : Sparse matrix (nx*ny x nx*ny) representing the system.

    Returns:
    lu : Object representing the LU decomposition of the matrix A.
    """
    lu = splinalg.splu(A)
    return lu

def solve_with_lu_decomposition(lu, f, hx, hy, nx, ny):
    """
    Solves the Poisson equation for a given right-hand side f using precomputed LU decomposition.

    Parameters:
    lu  : LU decomposition of the matrix A.
    f   : 2D array (nx x ny) representing the source term f(x, y).
    hx  : grid spacing in the x direction.
    hy  : grid spacing in the y direction.
    nx  : number of grid points in the x direction.
    ny  : number of grid points in the y direction.

    Returns:
    phi : 2D array (nx x ny) representing the solution phi(x, y).
    """

    # Flatten f and scale by grid spacings hx and hy
    b = f.flatten() / (hx * hy)

    # Solve the system using the LU decomposition
    phi_flat = lu.solve(b)

    # Reshape the solution back to 2D
    phi = phi_flat.reshape((nx, ny))

    return phi

# Example usage
nx, ny = 502, 502  # Grid size
# hx, hy = 0.01, 0.01  # Different grid spacings
# x = np.linspace(0, hx * (nx - 1), nx)
# y = np.linspace(0, hy * (ny - 1), ny)
# X, Y = np.meshgrid(x, y)

# Define spatially varying permittivity
# eps = np.ones((nx, ny))
# eps[nx // 4:nx // 2, ny // 4:ny // 2] = 5.0  # High permittivity in a region

# Construct the sparse matrix A for the Poisson equation with permittivity and varying hx, hy




# Example of a Gaussian source
# sigma = 3  # Spread of the source
# source_x1, source_y1 = nx // 2, ny // 2
# source_x2, source_y2 = 2 * nx // 4, 2 * ny // 4

# Create grid using meshgrid
Lx = 150              # Length in x-direction
Ly = 150            # Length in y-direction
# x = np.linspace(-Lx, Lx, nx)
y = np.linspace(-Ly, Ly, ny)
x = y
X, Y = np.meshgrid(x, y)
hx = x[1] - x[0]
hy = y[1] - y[0]
# Initial condition
eps = np.ones(X.shape)
# eps[X > 0] *= 12

# X, Y = np.meshgrid(np.linspace(0, nx-1, nx), np.linspace(0, ny-1, ny))
A = poisson_2d_matrix_with_permittivity_varying_h(eps, hx, hy, nx, ny)
# Perform LU decomposition of A (done once)
lu = lu_decomposition_of_poisson_matrix(A)

# Calculate source term
alpha = 7
sigma_y = 5/np.sqrt(8 * np.log(2))
# source = np.exp(-alpha * X) * np.exp(-(Y**2) / (2 * sigma_y**2))
source =  np.exp(-(Y**2 + X**2) / (2 * sigma_y**2)) / (2 * np.pi * sigma_y**2)
# source[nx//2, ny//2] = 1
# source[X < 0] = 0

# Set the source to zero at all boundaries
source[0, :] = 0        # Bottom boundary (y=0)
source[-1, :] = 0       # Top boundary (y=ny-1)
source[:, 0] = 0        # Left boundary (x=0)
source[:, -1] = 0       # Right boundary (x=nx-1)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot the permittivity (eps)
im1 = axs[0].contourf(X, Y, eps, cmap='plasma')
fig.colorbar(im1, ax=axs[0])
axs[0].set_title('Permittivity (eps)')

# Plot the source term
im2 = axs[1].contourf(X, Y, source, cmap='viridis')
fig.colorbar(im2, ax=axs[1])
axs[1].set_title('Source Term (f) - Symmetry Check')

plt.tight_layout()
# plt.show()

# Solve the Poisson equation for each f using the same LU decomposition
phi1 = solve_with_lu_decomposition(lu, -source, hx, hy, nx, ny)
# phi2 = solve_with_lu_decomposition(lu, -source, hx, hy, nx, ny)
phi_theory = potential_theoretical_gaussian(X, Y, sigma_y)

# Plot the results
import matplotlib.pyplot as plt
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.contourf(X, Y, phi1, levels=50, cmap='viridis')
plt.colorbar(label='Potential φ for f1')
plt.title('Solution for f1')

plt.subplot(1, 3, 2)
plt.contourf(X, Y, phi_theory, levels=50, cmap='viridis')
plt.colorbar(label='Potential φ for theory')
plt.title('Solution for theory')

plt.subplot(1, 3, 3)
plt.contourf(X, Y, phi_theory + phi1, levels=50, cmap='viridis')
plt.colorbar(label='Potential φ for theory')
plt.title('Solution for theory')

# plt.show()

plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.plot(x, phi_theory[nx//2, :])
plt.subplot(1, 3, 2)
plt.plot(x, phi1[nx//2, :])
plt.subplot(1, 3, 3)
plt.plot(x, phi1[nx//2, :] * phi_theory[nx//2, :])
# plt.show()

def compute_electric_field(phi, hx, hy):
    """
    Computes the electric field components E_x and E_y from the potential phi.

    Parameters:
    phi : 2D array (nx x ny) representing the potential φ(x, y).
    hx  : grid spacing in the x direction.
    hy  : grid spacing in the y direction.

    Returns:
    Ex  : 2D array (nx-1 x ny) representing the x-component of the electric field.
    Ey  : 2D array (nx x ny-1) representing the y-component of the electric field.
    """
    # Compute the electric field components using central finite differences
    Ex = -(phi[1:, :] - phi[:-1, :]) / hx  # E_x = -dφ/dx (centered difference along x)
    Ey = -(phi[:, 1:] - phi[:, :-1]) / hy  # E_y = -dφ/dy (centered difference along y)
    
    return Ex, Ey

# # Example usage
# phi = solve_with_lu_decomposition(lu, f1 + f2, hx, hy, nx, ny)  # Solve Poisson for some f

# Compute electric field components

Ex, Ey = compute_electric_field(phi1 , hx, hy)
Ey_theory, Ex_theory = compute_electric_field(-phi_theory , hx, hy)

# Ensure Ex and Ey have the same dimensions by slicing
min_shape_x, min_shape_y = min(Ex.shape[0], Ey.shape[0]), min(Ex.shape[1], Ey.shape[1])
Ex_resized = Ex[:min_shape_x, :min_shape_y]
Ey_resized = Ey[:min_shape_x, :min_shape_y]

# Compute the magnitude of the electric field
magnitude = np.sqrt(Ex_resized**2 + Ey_resized**2)

# Create the quiver plot
plt.figure(figsize=(8, 6))
plt.quiver(X[:min_shape_x, :min_shape_y], Y[:min_shape_x, :min_shape_y], Ex_resized, Ey_resized, 
           scale=1e-7, scale_units='xy', width=0.003, headwidth=3)
plt.title('Electric Field Vector Field (Ex, Ey) with Matching Dimensions')
plt.xlabel('x')
plt.ylabel('y')
# plt.show()



# Plot the results
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot Ex
im1 = axs[0].imshow(Ex.T, extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower', cmap='RdBu')
axs[0].set_title('Electric Field Component Ex')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
fig.colorbar(im1, ax=axs[0], label='Ex')

# Plot Ey
im2 = axs[1].imshow(Ey.T, extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower', cmap='RdBu')
axs[1].set_title('Electric Field Component Ey')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
fig.colorbar(im2, ax=axs[1], label='Ey')

plt.tight_layout()



# fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# # Plot Ex
# im1 = axs[0].imshow(Ex_theory, extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower', cmap='RdBu')
# axs[0].set_title('Electric Field Component Ex_th')
# axs[0].set_xlabel('x')
# axs[0].set_ylabel('y')
# fig.colorbar(im1, ax=axs[0], label='Ex')

# # Plot Ey
# im2 = axs[1].imshow(Ey_theory, extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower', cmap='RdBu')
# axs[1].set_title('Electric Field Component Ey_th')
# axs[1].set_xlabel('x')
# axs[1].set_ylabel('y')
# fig.colorbar(im2, ax=axs[1], label='Ey')

# plt.tight_layout()
# plt.show()

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot Ex
im1 = axs[0].imshow(Ex_theory , extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower', cmap='RdBu')
axs[0].set_title('Electric Field Component Ex_th')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
fig.colorbar(im1, ax=axs[0], label='Ex')

# Plot Ey
im2 = axs[1].imshow(Ey_theory , extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower', cmap='RdBu')
axs[1].set_title('Electric Field Component Ey_th')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
fig.colorbar(im2, ax=axs[1], label='Ey')

plt.tight_layout()
# plt.show()

fig, axs = plt.subplots(1, 1, figsize=(12, 6))
# im1 = axs.plot(x,Ex[nx//2 , :])
im2 = axs.plot(x,-Ex_theory[ :, nx//2 ] /Ex[nx//2 , :], 'r--')
axs.set_ylim([0, 0.01])
plt.show()
