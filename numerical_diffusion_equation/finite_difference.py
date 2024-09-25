import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# Parameters
nx, nz = 50, 50  # Number of grid points
lx, lz = 1, 300  # Domain size
dx, dz = lx / (nx - 1), lz / (nz - 1)  # Grid spacing
D = 0.08  # Diffusion coefficient
dt = 0.0005  # Time step
nt = 2000  # Number of time steps

# Directory to save the images
output_dir = "numerical_diffusion_equation/saved_currents/"
os.makedirs(output_dir, exist_ok=True)

# Gaussian and exponential parameters
x0, z0 = 0, 0  # Center of the Gaussian
sigma_z = 40 / (np.sqrt(8 * np.log(2)))  # Standard deviation
alpha = 7  # Exponential decay coefficient

# Initial condition: Gaussian distribution in z and exponential decay in x
x = np.linspace(0, lx, nx)
z = np.linspace(-lz / 2, lz / 2, nz)
X, Z = np.meshgrid(x, z, indexing='xy')
u = np.exp(- (Z - z0) ** 2 / (2 * sigma_z ** 2)) * np.exp(- alpha * (X - x0))

# Create a figure with 1x3 subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
cax1 = axs[0].imshow(u, cmap='hot', interpolation='nearest', origin='lower', extent=[0, lx, -lz / 2, lz / 2])
cbar1 = fig.colorbar(cax1, ax=axs[0], fraction=0.046, pad=0.08)
cbar1.set_label('u')

# Set axis labels and titles for the main plot
axs[0].set_xlabel(r'x, $\mu m$')
axs[0].set_ylabel(r'z, $\mu m$')
axs[0].set_title('Charge density')

# Prepare subplots for derivatives
cax2 = axs[1].imshow(np.zeros((nz, nx)), cmap='coolwarm', interpolation='nearest', origin='lower', extent=[0, lx, -lz / 2, lz / 2])
cbar2 = fig.colorbar(cax2, ax=axs[1], fraction=0.046, pad=0.08)
cbar2.set_label(r'$J_x$')

# axs[1].set_xlabel(r'x, $\mu m$')
# axs[1].set_ylabel(r'z, $\mu m$')

cax3 = axs[2].imshow(np.zeros((nz, nx)), cmap='coolwarm', interpolation='nearest', origin='lower', extent=[0, lx, -lz / 2, lz / 2])
cbar3 = fig.colorbar(cax3, ax=axs[2], fraction=0.046, pad=0.08)
cbar3.set_label(r'$J_z$')

# axs[2].set_xlabel(r'x, $\mu m$')
# axs[2].set_ylabel(r'z, $\mu m$')

# Increase gap between subplots
plt.subplots_adjust(wspace=0.4)  # Adjust this value to increase/decrease the gap

# Initialize colorbar limits
du_dx_min, du_dx_max = np.inf, -np.inf
du_dz_min, du_dz_max = np.inf, -np.inf

def update(frame):
    global u, du_dx_min, du_dx_max, du_dz_min, du_dz_max
    un = u.copy()
    
    # Finite difference scheme
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] +
                     D * dt / dx**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]) +
                     D * dt / dz**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]))
    
    # Neumann boundary condition on the left wall (du/dx = 0)
    u[:, 0] = u[:, 1]  # Set the gradient at the boundary to zero

    # Dirichlet boundary conditions on the other walls
    u[0, :] = 0
    u[:, -1] = 0
    u[-1, :] = 0
    
    # Compute derivatives
    du_dx = np.zeros_like(u)
    du_dz = np.zeros_like(u)
    
    du_dz[1:-1, 1:-1] = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dz)
    du_dx[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)
    
    # Update colorbar limits based on current frame
    du_dx_min = min(du_dx_min, np.min(du_dx))
    du_dx_max = max(du_dx_max, np.max(du_dx))
    du_dz_min = min(du_dz_min, np.min(du_dz))
    du_dz_max = max(du_dz_max, np.max(du_dz))
    
    # Update plots
    cax1.set_data(u)
    axs[0].set_title(f'Concentration u\nTime: {frame * dt:.3f} ps')

    cax2.set_data(du_dx)
    cax2.set_clim(du_dx_min, du_dx_max)
    axs[1].set_title(r'$J_x$')

    cax3.set_data(du_dz)
    cax3.set_clim(du_dz_min, du_dz_max)
    axs[2].set_title(r'$J_z$')

    # Ensure square subplots
    for ax in axs:
        ax.set_aspect(lx / lz)

    # Save the plot at every 500 steps
    if frame < 500:
        if frame % 50 == 0:
            plt.savefig(os.path.join(output_dir, f"plot_step_{frame:04d}.png"))
    elif frame % 500 == 0:
        plt.savefig(os.path.join(output_dir, f"plot_step_{frame:04d}.png"))

    # Force redraw of the canvas to update the plots
    fig.canvas.draw()
    plt.pause(0.01)

# Animation
ani = FuncAnimation(fig, update, frames=nt, blit=False, repeat=False, interval=50)
plt.show()
