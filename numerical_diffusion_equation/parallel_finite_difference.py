from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Parameters
nx, ny = 50, 50  # Number of grid points
lx, ly = 0.5, 150  # Domain size
dx, dy = lx / (nx - 1), ly / (ny - 1)  # Grid spacing
D = 0.1  # Diffusion coefficient
dt = 0.0002  # Time step
nt = 500  # Number of time steps

# Gaussian and exponential parameters
x0, y0 = 0, ly / 2  # Center of the Gaussian
sigma_y = 16  # Standard deviation
alpha = 7  # Exponential decay coefficient

# Initialize MPI domain
sub_ny = ny // size
sub_ny_r = ny % size

if rank < sub_ny_r:
    local_ny = sub_ny + 1
    sub_start_y = rank * local_ny
else:
    local_ny = sub_ny
    sub_start_y = rank * local_ny + sub_ny_r

sub_end_y = sub_start_y + local_ny

x = np.linspace(0, lx, nx)
y = np.linspace(sub_start_y * dy, sub_end_y * dy, local_ny)
X, Y = np.meshgrid(x, y, indexing='xy')

# Initial condition: Gaussian distribution in y and exponential decay in x
u = np.exp(- (Y - y0) ** 2 / (2 * sigma_y ** 2)) * np.exp(- alpha * (X - x0))

# Create a figure with 1x3 subplots (only on rank 0)
if rank == 0:
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    cax1 = axs[0].imshow(u, cmap='hot', interpolation='nearest', origin='lower', extent=[0, lx, 0, ly])
    cbar1 = fig.colorbar(cax1, ax=axs[0])
    cbar1.set_label('u')

    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title('Concentration u')

    cax2 = axs[1].imshow(np.zeros((local_ny, nx)), cmap='coolwarm', interpolation='nearest', origin='lower', extent=[0, lx, 0, ly])
    cbar2 = fig.colorbar(cax2, ax=axs[1])
    cbar2.set_label('du/dx')

    axs[1].set_title('du/dx')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')

    cax3 = axs[2].imshow(np.zeros((local_ny, nx)), cmap='coolwarm', interpolation='nearest', origin='lower', extent=[0, lx, 0, ly])
    cbar3 = fig.colorbar(cax3, ax=axs[2])
    cbar3.set_label('du/dy')

    axs[2].set_title('du/dy')
    axs[2].set_xlabel('x')
    axs[2].set_ylabel('y')

# Initialize colorbar limits
du_dx_min, du_dx_max = np.inf, -np.inf
du_dy_min, du_dy_max = np.inf, -np.inf

def update(frame):
    global u, du_dx_min, du_dx_max, du_dy_min, du_dy_max

    # Finite difference scheme
    un = u.copy()

    u[1:-1, 1:-1] = (un[1:-1, 1:-1] +
                     D * dt / dx**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]) +
                     D * dt / dy**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]))

    # Boundary conditions
    if rank == 0:
        u[0, :] = 0
    if rank == size - 1:
        u[-1, :] = 0
    u[:, 0] = u[:, 1]
    u[:, -1] = 0

    # Communicate boundaries with neighbors
    if rank > 0:
        comm.send(u[1, :], dest=rank - 1, tag=11)
        u_recv = comm.recv(source=rank - 1, tag=12)
        u[0, :] = u_recv
    if rank < size - 1:
        comm.send(u[-2, :], dest=rank + 1, tag=12)
        u_recv = comm.recv(source=rank + 1, tag=11)
        u[-1, :] = u_recv

    # Compute derivatives
    du_dx = np.zeros_like(u)
    du_dy = np.zeros_like(u)

    du_dy[1:-1, 1:-1] = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy)
    du_dx[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)

    # Update colorbar limits based on current frame
    du_dx_min = min(du_dx_min, np.min(du_dx))
    du_dx_max = max(du_dx_max, np.max(du_dx))
    du_dy_min = min(du_dy_min, np.min(du_dy))
    du_dy_max = max(du_dy_max, np.max(du_dy))

    # Update plots (only on rank 0)
    if rank == 0:
        cax1.set_data(u)
        axs[0].set_title(f'Concentration u\nTime: {frame * dt:.3f} s')

        cax2.set_data(du_dx)
        cax2.set_clim(du_dx_min, du_dx_max)
        axs[1].set_title('du/dx')

        cax3.set_data(du_dy)
        cax3.set_clim(du_dy_min, du_dy_max)
        axs[2].set_title('du/dy')

        # Ensure square subplots
        for ax in axs:
            ax.set_aspect(lx / ly)

        # Force redraw of the canvas to update the plots
        fig.canvas.draw()
        plt.pause(0.01)

# Animation (only on rank 0)
if rank == 0:
    ani = FuncAnimation(fig, update, frames=nt, blit=False, repeat=False, interval=50)
    plt.show()

# Finalize MPI
MPI.Finalize()
