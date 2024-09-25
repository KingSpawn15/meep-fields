import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# paramters from FDTD simuation setup
Time_Sec_To_MEEP = (1e-6 / 3e8)
Time_MEEP_To_Sec = 1 / Time_Sec_To_MEEP
sx = 200
sy = 50
simulation_end_time_meep = 1000
time_range = simulation_end_time_meep / Time_MEEP_To_Sec * 1e12

outdir = 'tomography/air'
path = 'saved_matrices/' + outdir 
# Load the saved 3D NumPy arrays for Ex and Ey fields
ex_fields_3d = np.load(path + "ex_fields_3d.npy")
ey_fields_3d = np.load(path + "ey_fields_3d.npy")

print(f'Loaded Ex fields shape: {ex_fields_3d.shape}')
print(f'Loaded Ey fields shape: {ey_fields_3d.shape}')

xx = np.linspace(0,25,127,endpoint=True)
rr = np.linspace(-100,100,1002,endpoint=True)
tt = np.linspace(0,time_range,500, endpoint=True)

def interpolation_loaded_fields(arrtxyz):
    """
    Interpolates electric field components (E_x, E_y, E_z) based on given spatial coordinates.

    Parameters:
        arrtxyz (ndarray): A 2D array where each row represents a point in the form 
                           [t, x, y, z]. The columns correspond to time and spatial coordinates.

    Returns:
        tuple: A tuple containing three arrays:
            - E_x (ndarray): Interpolated electric field component along the x-axis.
            - E_y (ndarray): Interpolated electric field component along the y-axis.
            - E_z (ndarray): Interpolated electric field component along the z-axis.
    """
    
    # Create interpolators for the electric field components
    interpolator_r = RegularGridInterpolator((tt, rr, xx), ex_fields_3d, bounds_error=False, fill_value=0)
    interpolator_x = RegularGridInterpolator((tt, rr, xx), ey_fields_3d, bounds_error=False, fill_value=0)

    # Extract spatial coordinates from the input array
    tpoints = arrtxyz[:, 0]
    xpoints = arrtxyz[:, 1]
    ypoints = arrtxyz[:, 2]
    zpoints = arrtxyz[:, 3]

    # Calculate polar coordinates directly for interpolation
    rpoints = np.sqrt(ypoints**2 + zpoints**2)
    phi = np.arctan2(ypoints, zpoints)

    # Prepare points for interpolation
    points_to_interpolate = np.vstack((tpoints, rpoints, xpoints)).T

    # Perform interpolation for electric field components
    interpolated_r = interpolator_r(points_to_interpolate)
    interpolated_ex = interpolator_x(points_to_interpolate)

    # Compute E_y and E_z components using vectorized operations
    interpolated_ey = interpolated_r * np.sin(phi)
    interpolated_ez = interpolated_r * np.cos(phi)

    return interpolated_ex, interpolated_ey, interpolated_ez


# EXAMPLE fields on a plane example 



# ttg, rrg, zzg = np.meshgrid(tt, rr, xx,indexing='ij')



# Create the arrays directly
zp = np.array([0])  # Single value array
tp = np.linspace(0, time_range, 500)
yp = np.linspace(-100, 100, 1002)
xp = np.array([10])  # Single value array

# Create the meshgrid with broadcasting
tpp, xpp, ypp, zpp = np.meshgrid(tp, xp, yp, zp, indexing='ij')

# Stack the meshgrid directly to form the input array for find_fields_exeyez
input_array = np.empty((tpp.size, 4))
input_array[:, 0] = tpp.flatten()
input_array[:, 1] = xpp.flatten()
input_array[:, 2] = ypp.flatten()
input_array[:, 3] = zpp.flatten()

# Call the function
ex, ey, ez = interpolation_loaded_fields(input_array)

# Reshape the output directly
ex_slice = ex.reshape((500, 1, 1002, 1))
ey_slice = ey.reshape((500, 1, 1002, 1))
ez_slice = ez.reshape((500, 1, 1002, 1))

# Create a figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns of subplots

# Plot for ex_slice
im1 = axes[0].imshow(ex_slice[:, 0, :, 0].T, 
                     cmap='bwr', 
                     aspect=time_range / sx, 
                     extent=[0, time_range, -sx/2, sx/2])
axes[0].set_title('Ex Slice')
fig.colorbar(im1, ax=axes[0])  # Add colorbar to the subplot

# Plot for ey_slice
im2 = axes[1].imshow(ey_slice[:, 0, :, 0].T, 
                     cmap='bwr', 
                     aspect=time_range / sx, 
                     extent=[0, time_range, -sx/2, sx/2])
axes[1].set_title('Ey Slice')
fig.colorbar(im2, ax=axes[1])  # Add colorbar to the subplot

# Plot for ez_slice
im3 = axes[2].imshow(ez_slice[:, 0, :, 0].T, 
                     cmap='bwr', 
                     aspect=time_range / sx, 
                     extent=[0, time_range, -sx/2, sx/2])
axes[2].set_title('Ez Slice')
fig.colorbar(im3, ax=axes[2])  # Add colorbar to the subplot

# Show the plots
plt.tight_layout()  # Adjust spacing between subplots
plt.show()

print()