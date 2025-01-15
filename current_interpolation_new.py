import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator
from interp3d.interp3d import interp_3d

def setup_interpolators(nc_formatted, sigma_y_formatted, base_dir="current_densities", subdirs=("eps12", "diffusion")):
    """
    Sets up interpolators for current densities using the stored .npy file.

    Args:
        nc_formatted (str): Formatted string for the simulation case (e.g., "0001").
        sigma_y_formatted (str): Formatted string for sigma_y (e.g., "0.50").
        base_dir (str): Base directory where current densities are stored.
        subdirs (tuple): Subdirectories specifying the storage path.

    Returns:
        tuple: Interpolators for jx and jy.
    """
    # Construct the full path to the .npy file
    file_path = os.path.join(base_dir, *subdirs, f"currents_nc_{nc_formatted}_sigma_{sigma_y_formatted}_t0_0.29.npy")


    # Load the stored currents dictionary from the .npy file
    stored_currents = np.load(file_path, allow_pickle=True).item()
    
    # Extract grid and data parameters from the dictionary
    jx_storage = stored_currents['j_x']
    jy_storage = stored_currents['j_y']
    x_grid = stored_currents['x']
    y_grid = stored_currents['y']
    T = stored_currents['T']
    Nt = jx_storage.shape[2]  # Infer Nt from the shape of j_x

    # Ensure x_grid and y_grid are constrained to relevant ranges
    x_grid = x_grid[x_grid >= 0]
    y_grid = y_grid[abs(y_grid) < 100]

    # Create time grid
    t_grid = np.linspace(0, T, Nt)

    # Create interpolators for jx and jy
    jx_interpolator = interp_3d.Interp3D(jx_storage, y_grid, x_grid, t_grid)
    jy_interpolator = interp_3d.Interp3D(jy_storage, y_grid, x_grid, t_grid)

    return jx_interpolator, jy_interpolator

def get_jx_at(x, y, t, jx_interpolator):
    """
    Returns interpolated jx and jy values at the given (x, y, t).

    Args:
        x (float): x-coordinate.
        y (float): y-coordinate.
        t (float): Time.
        jx_interpolator: Interpolator for jx.
        jy_interpolator: Interpolator for jy.

    Returns:
        tuple: Interpolated values for jx and jy.
    """
    jx = jx_interpolator((y, x, t))

    return jx

def get_jy_at(x, y, t, jy_interpolator):
    """
    Returns interpolated jx and jy values at the given (x, y, t).

    Args:
        x (float): x-coordinate.
        y (float): y-coordinate.
        t (float): Time.
        jx_interpolator: Interpolator for jx.
        jy_interpolator: Interpolator for jy.

    Returns:
        tuple: Interpolated values for jx and jy.
    """
    jy = jy_interpolator((y, x, t))
    return jy


# import matplotlib.pyplot as plt
# # Example simulation case identifier (e.g., "0001")
# nc_formatted = "4.00e+10"

# # Set up interpolators
# jx_interpolator, jy_interpolator = setup_interpolators(nc_formatted)

# # Define the time and ranges
# t = 0.1
# x_range = np.linspace(0, 1, 100)  # x range from 0 to 1
# y_range = np.linspace(-50, 50, 100)  # y range from -50 to 50

# # Create meshgrid for plotting
# X, Y = np.meshgrid(x_range, y_range)

# # Initialize arrays for jx and jy
# jx_values = np.zeros_like(X)
# jy_values = np.zeros_like(X)

# # Interpolate jx and jy values over the grid
# for i in range(len(x_range)):
#     for j in range(len(y_range)):
#         jx_values[j, i], jy_values[j, i] = get_jx_jy_at(x_range[i], y_range[j], t, jx_interpolator, jy_interpolator)

# # Plot jx
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.contourf(X, Y, jx_values, cmap='viridis')
# plt.colorbar(label="jx")
# plt.title(f"jx at t = {t}")
# plt.xlabel("x")
# plt.ylabel("y")

# # Plot jy
# plt.subplot(1, 2, 2)
# plt.contourf(X, Y, jy_values, cmap='viridis')
# plt.colorbar(label="jy")
# plt.title(f"jy at t = {t}")
# plt.xlabel("x")
# plt.ylabel("y")

# # Show the plots
# plt.tight_layout()
# plt.show()