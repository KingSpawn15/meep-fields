import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator
from interp3d.interp3d import interp_3d

def setup_interpolators():
    data_dir = './current_densities/'
    Nx = 49
    Ny = 500
    Nt = 1000
    x_min, x_max = -5, 5
    y_min, y_max = -150, 150
    T = 3

    # Load the stored 3D arrays from .npy files
    jx_storage = np.load(os.path.join(data_dir, 'jx_storage.npy'))
    jy_storage = np.load(os.path.join(data_dir, 'jy_storage.npy'))

    # Define the grid points
    x_grid = np.linspace(x_min, x_max, Nx)
    x_grid = x_grid[x_grid >= 0]
    y_grid = np.linspace(y_min, y_max, Ny)
    y_grid = y_grid[abs(y_grid) < 100]
    t_grid = np.linspace(0, T, Nt)

    # Create interpolators for jx and jy
    jx_interpolator = interp_3d.Interp3D(jx_storage, y_grid, x_grid, t_grid)
    jy_interpolator = RegularGridInterpolator((y_grid, x_grid, t_grid), jy_storage, bounds_error=False, fill_value=0)

    return jx_interpolator, jy_interpolator

def get_jx_jy_at(x, y, t, jx_interpolator):
    """Returns interpolated jx value at the given (x, y, t)."""
    return jx_interpolator((y, x, t))