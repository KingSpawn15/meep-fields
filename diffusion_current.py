import numpy as np
import matplotlib.pyplot as plt

# Parameters
nc = 4.6e3          # number density n0
alpha = 7           # Decay constant
gamma = 3.3         # Decay constant for source term
sigma_y = 20
D = 0.1             # Diffusion coefficient
t_r = 4             # Recombination time
Lx = 5              # Length in x-direction
Ly = 200            # Length in y-direction
T = 3               # Total time
Nx = 49             # Number of grid points in x-direction
Ny = 500            # Number of grid points in y-direction
Nt = 1000           # Number of time steps
t0 = 0.1
sigma_t = 0.02
mu = 4

dx = Lx / (Nx - 1)  # Grid spacing in x-direction
dy = Ly / (Ny - 1)  # Grid spacing in y-direction
dt = T / Nt         # Time step size

# Stability condition check
if D * dt / dx**2 > 0.5 or D * dt / dy**2 > 0.5:
    raise ValueError('The solution is unstable. Reduce dt or increase dx and dy.')

# Create grid using meshgrid
x = np.linspace(-Lx, Lx, Nx)
y = np.linspace(-Ly, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Initial condition
eps = np.ones(X.shape)
eps[X < 0] *= 12
n_e = nc * np.exp(-alpha * X) * np.exp(-(Y**2) / (2 * sigma_y**2)) * np.exp(-t0**2 / (2 * sigma_t**2))
n_e[X < 0] = 0
n_h = np.copy(n_e)
div_neE = np.zeros_like(n_e)
phi_old = div_neE + 1e-6
matEy = []

# Visualization setup
fig, ax = plt.subplots(2, 3, figsize=(12, 6))

# Function to compute divergence and field
import numpy as np

import numpy as np

def poisson_solver(n_e, n_h, x, y, epsilon_0, phi, tol, max_iter, eps):
    # Constants
    qe = -1.60217662e-19  # Elementary charge in Coulombs

    # Define the charge density
    rho = qe * (n_e - n_h)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Initialize the potential with the initial guess
    V = phi.copy()

    # Set boundary conditions (V = 0 at the boundaries)
    V[:, 0] = 0   # Left boundary
    V[:, -1] = 0  # Right boundary
    V[0, :] = 0   # Bottom boundary
    V[-1, :] = 0  # Top boundary

    # Iterative solver (Gauss-Seidel method)
    converged = False
    for iter in range(max_iter):
        V_old = V.copy()

        # Calculate epsilon at half-grid points
        epsilon_x1 = (eps[:-1, 1:-1] + eps[1:, 1:-1]) / 2  # between i and i+1
        epsilon_x2 = (eps[1:, 1:-1] + eps[:-1, 1:-1]) / 2  # between i and i-1
        epsilon_y1 = (eps[1:-1, :-1] + eps[1:-1, 1:]) / 2  # between j and j+1
        epsilon_y2 = (eps[1:-1, 1:] + eps[1:-1, :-1]) / 2  # between j and j-1

        # Calculate the denominator
        denominator = (epsilon_x1 / dx**2 + epsilon_x2 / dx**2 +
                    epsilon_y1 / dy**2 + epsilon_y2 / dy**2)

        # Update V in the inner region
        V[1:-1, 1:-1] = (1 / denominator) * (
            epsilon_x1 * V[2:, 1:-1] / dx**2 +
            epsilon_x2 * V[:-2, 1:-1] / dx**2 +
            epsilon_y1 * V[1:-1, 2:] / dy**2 +
            epsilon_y2 * V[1:-1, :-2] / dy**2 +
            rho[1:-1, 1:-1] / epsilon_0
        )


        # Check for convergence
        if np.max(np.abs(V - V_old)) < tol:
            converged = True
            print(f"Converged in {iter+1} iterations")
            break

    # Calculate the electric field components (negative gradient of potential)
    E_y, E_x = np.gradient(-V, dy, dx)

    # Return the electric field, potential, and convergence flag
    return E_x, E_y, V, converged


def compute_divergence_and_field(n_e, n_h, x, y, phi, eps):
    # Constants
    epsilon_0 = 8.854e-18  # Permittivity of free space

    # Set tolerance and maximum iterations for the Poisson solver
    tol = 1e-5
    max_iter = 5000
    
    # Call the Poisson solver (assumed to be defined elsewhere)
    Ex, Ey, phi, converged = poisson_solver(n_e, n_h, x, y, epsilon_0, phi, tol, max_iter, eps)
    
    if not converged:
        print("Warning: Poisson solver not converged")

    # Compute grid spacing
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Compute the gradients of the electron density n_e
    d_n_e_dy, d_n_e_dx = np.gradient(n_e, dy, dx)

    # Compute the gradients of the electric field components Ex and Ey
    dE_x_dy, dE_x_dx = np.gradient(Ex, dy, dx)
    dE_y_dy, dE_y_dx = np.gradient(Ey, dy, dx)

    # Apply the product rule to compute the divergence of n_e * E
    div_neE = Ex * d_n_e_dx + Ey * d_n_e_dy + n_e * (dE_x_dx + dE_y_dy)

    return Ex, Ey, div_neE, phi



# Function to compute current
def compute_current(n_e, Ex, Ey, D, mu, dx, dy):
    q = 1.6e-19
    d_n_e_dx, d_n_e_dy = np.gradient(n_e, dx, dy)
    jx = -q * D * d_n_e_dx + q * mu * n_e * Ex
    jy = -q * D * d_n_e_dy + q * mu * n_e * Ey
    return jx, jy

import matplotlib.pyplot as plt
import numpy as np

def plot_results(params):
    

    # Clear the current plot to prepare for a new update
    plt.clf()

    # Create a subplot for n_e
    plt.subplot(2, 3, 1)
    plt.imshow(params['n_e'], extent=[params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]], origin='lower', aspect='auto')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"Electron Density at t = {params['n'] * params['dt']}")
    plt.colorbar()
    plt.axvline(x=0, color='w', linewidth=0.5, linestyle='-')  # Vertical white line at x=0

    # Create a subplot for n_h
    plt.subplot(2, 3, 2)
    plt.imshow(params['n_h'], extent=[params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]], origin='lower', aspect='auto')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"Hole Density at t = {params['n'] * params['dt']}")
    plt.colorbar()
    plt.axvline(x=0, color='w', linewidth=0.5, linestyle='-')  # Vertical white line at x=0

    # Create a subplot for source_term
    plt.subplot(2, 3, 3)
    plt.imshow(params['source_term'], extent=[params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]], origin='lower', aspect='auto')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"Source Density at t = {params['n'] * params['dt']}")
    plt.colorbar()
    plt.clim([0, params['nc']])
    plt.axvline(x=0, color='w', linewidth=0.5, linestyle='-')  # Vertical white line at x=0

    # Create a subplot for Jx
    plt.subplot(2, 3, 4)
    plt.imshow(params['jx'], extent=[params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]], origin='lower', aspect='auto')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"Jx at t = {params['n'] * params['dt']}")
    plt.colorbar()
    plt.axvline(x=0, color='w', linewidth=0.5, linestyle='-')  # Vertical white line at x=0

    # Create a subplot for Jy
    plt.subplot(2, 3, 5)
    plt.imshow(params['jy'], extent=[params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]], origin='lower', aspect='auto')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"Jy at t = {params['n'] * params['dt']}")
    plt.colorbar()
    plt.axvline(x=0, color='w', linewidth=0.5, linestyle='-')  # Vertical white line at x=0

    # Create a subplot for dn_dt (Ey)
    # plt.subplot(2, 3, 6)
    # X, Y = np.meshgrid(params['x'], params['y'])  # Create meshgrid for surface plot
    # plt.pcolormesh(X, Y, params['dn_dt'].T, shading='auto')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title(f"Ey at t = {params['n'] * params['dt']}")
    # plt.colorbar()
    # plt.axvline(x=0, color='w', linewidth=0.5, linestyle='-')  # Vertical white line at x=0

    # Redraw the plot and pause to update
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)  # Short pause to allow interactive updates






# Time-stepping loop
for n in range(Nt):
    n_e_new = np.copy(n_e)
    n_h_new = np.copy(n_h)

    # Calculate source term
    source_term = nc * (1 / np.sqrt(2 * np.pi * sigma_t**2)) * np.exp(-alpha * X) * \
                  np.exp(-(Y**2) / (2 * sigma_y**2)) * np.exp(-((n * dt - t0)**2) / (2 * sigma_t**2))
    source_term[X < 0] = 0

    # Finite difference for second-order derivatives
    d2_n_e_dx2 = (n_e[2:, 1:-1] - 2 * n_e[1:-1, 1:-1] + n_e[:-2, 1:-1]) / dx**2
    d2_n_e_dy2 = (n_e[1:-1, 2:] - 2 * n_e[1:-1, 1:-1] + n_e[1:-1, :-2]) / dy**2

    # Update electron density
    dn_dt = np.zeros_like(n_e)
    dn_dt[1:-1, 1:-1] = D * (d2_n_e_dx2 + d2_n_e_dy2) + source_term[1:-1, 1:-1] - \
                        (n_h[1:-1, 1:-1] * n_e[1:-1, 1:-1] / t_r) + mu * div_neE[1:-1, 1:-1] - \
                        n_e[1:-1, 1:-1] * gamma
    n_e_new[1:-1, 1:-1] += dt * dn_dt[1:-1, 1:-1]

    n_h_new[1:-1, 1:-1] += dt * (-n_h[1:-1, 1:-1] * n_e[1:-1, 1:-1] / t_r + source_term[1:-1, 1:-1])

    # Reflecting boundary at x = 0
    n_e_new[Nx//2, :] = n_e_new[Nx//2 + 1, :]
    n_h_new[Nx//2, :] = n_h_new[Nx//2 + 1, :]

    # Absorbing boundary conditions
    n_e_new[0, :] = n_e_new[-1, :] = 0
    n_e_new[:, 0] = n_e_new[:, -1] = 0
    n_h_new[0, :] = n_h_new[-1, :] = 0
    n_h_new[:, 0] = n_h_new[:, -1] = 0

    n_e_new[0:Nx//2, :] = 0
    n_h_new[0:Nx//2, :] = 0
    # Update n_e and n_h for the next time step
    n_e = n_e_new
    n_h = n_h_new

    # Compute divergence for the next iteration
    Ex, Ey, div_neE, phi_old = compute_divergence_and_field(n_e, n_h, x, y, phi_old, eps)
    jx, jy = compute_current(n_e, Ex, Ey, D, mu, dx, dy)

    # Collect results and plot every 20 steps
    if n % 20 == 0:
        params = {
            'n': n, 'dt': dt, 'x': x, 'y': y, 'n_e': n_e, 'n_h': n_h,
            'source_term': source_term, 'Ex': Ex, 'Ey': Ey, 'X': X, 'Y': Y,
            'nc': nc, 'jx': jx, 'jy': jy, 'dn_dt': dn_dt
        }
        plot_results(params)


