import sys
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time
import imageio
from io import BytesIO
from poisson_solver.fast_poisson_solver import fast_poisson_solver
from matplotlib.colors import Normalize
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import splu
import os

output_dir = './current_densities/eps12/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

nc = 4e10
alpha = 7
gamma = 3.3
sigma_y = 40/np.sqrt(8 * np.log(2))
diff = 0.1
t_r = 4e3
Lx = 100
Ly = 200
T = 1
Nx = 100
Ny = 250
Nt = 5000
t0 = 0.1
sigma_t = 50/np.sqrt(8 * np.log(2)) * 1e-3
mu = 4

dt = T / Nt


x = np.linspace(0, Lx, Nx)
xe = np.linspace(-Lx, Lx, 2 * Nx  - 1)
y = np.linspace(-Ly, Ly, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
if diff * dt / dx**2 > 0.5 or diff * dt / dy**2 > 0.5:
    raise ValueError('The solution is unstable. Reduce dt or increase dx and dy.')
X, Y = np.meshgrid(x, y)
XE, YE = np.meshgrid(xe, y)

eps = np.ones(XE.shape)
eps[XE > 0] *= 12

n_e = nc * alpha / (2 * np.sqrt(2) * np.pi**(3/2) * sigma_y**2 * sigma_t) * np.exp(-alpha * X) * \
                  np.exp(-(Y**2) / (2 * sigma_y**2)) * np.exp(-(( - t0)**2) / (2 * sigma_t**2)) * 0

# n_e[X < 0] = 0

n_h = np.copy(n_e) * 0
# div_neE = np.zeros_like(n_e)
# phi_old = div_neE + 1e-16
# matEy = []

nrows, ncols = XE.shape
hx = x[1] - x[0]
hy = y[1] - y[0]

def assemble_poisson_matrix(nrows, ncols, hx, hy, eps_in):
    A = lil_matrix((nrows * ncols, nrows * ncols))
    eps = eps_in.ravel()[:]
    for i in range(nrows):
        for j in range(ncols):
            k = i * ncols + j
            if i == 0 or i == nrows - 1 or j == 0 or j == ncols - 1:
                A[k, k] = 1
                continue
            A[k, k] = -0.5 * (eps[k-1] + eps[k+1] + 2 * eps[k]) / hx**2 - 0.5 * (eps[k-ncols] + eps[k+ncols] + 2 * eps[k]) / hy**2
            A[k, k + 1] = 1 / hx**2 * (eps[k] + eps[k+1]) * 0.5
            A[k, k - 1] = 1 / hx**2 * (eps[k] + eps[k-1]) * 0.5
            A[k, k + ncols] = 1 / hy**2 * (eps[k] + eps[k+ncols]) * 0.5
            A[k, k - ncols] = 1 / hy**2 * (eps[k] + eps[k-ncols]) * 0.5
    A = A.tocsc()
    lu_A = splu(A)
    return lu_A

def poisson_solver_traditional(n_e, n_h, x, y, epsilon_0, phi, tol, max_iter, eps):
    qe = 1.6e-19
    rho = qe * (n_e - n_h)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    V = phi.copy()
    V[:, 0] = 0
    V[:, -1] = 0
    V[0, :] = 0
    V[-1, :] = 0
    for iter in range(max_iter):
        V_old = V.copy()
        epsilon_y1 = (eps[1:-1, 1:-1] + eps[2:, 1:-1]) / 2
        epsilon_y2 = (eps[1:-1, 1:-1] + eps[:-2, 1:-1]) / 2
        epsilon_x1 = (eps[1:-1, 1:-1] + eps[1:-1, 2:]) / 2
        epsilon_x2 = (eps[1:-1, 1:-1] + eps[1:-1, :-2]) / 2
        denominator = (epsilon_x1 / dx**2 + epsilon_x2 / dx**2 + epsilon_y1 / dy**2 + epsilon_y2 / dy**2)
        V[1:-1, 1:-1] = (1 / denominator) * (
            epsilon_y1 * V[2:, 1:-1] / dy**2 +
            epsilon_y2 * V[:-2, 1:-1] / dy**2 +
            epsilon_x1 * V[1:-1, 2:] / dx**2 +
            epsilon_x2 * V[1:-1, :-2] / dx**2 +
            rho[1:-1, 1:-1] / epsilon_0
        )
        if np.max(np.abs(V - V_old)) < tol:
            break
    E_y, E_x = np.gradient(-V, dy, dx)
    return E_x, E_y, V, True

def compute_divergence_and_field(lu_A, n_e_in, n_h_in, x, y):
    epsilon_0 = 8.854e-18
    q_e = 1.60217662e-19
    n_e = n_e_in
    n_h = n_h_in

    n_e[:,0] = n_e[:,1]
    rho_s = (n_e - n_h) * q_e 

    zeroblock = np.zeros((Nx - 1, Ny))
    rho = np.concatenate((zeroblock, rho_s.T), axis=0).T


    phi = lu_A.solve(rho.ravel()[:] / epsilon_0).reshape(XE.shape)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    Ey_f, Ex_f = np.gradient(-phi, dy, dx)

    Ey = Ey_f[:, Nx -1 :]
    Ex = Ex_f[:, Nx -1 :]
    
    dE_x_dy, dE_x_dx = np.gradient(Ex, dy, dx)
    dE_y_dy, dE_y_dx = np.gradient(Ey, dy, dx)

    d_n_e_dy, d_n_e_dx = np.gradient(n_e, dy, dx)


    div_neE = (Ex * d_n_e_dx + Ey * d_n_e_dy + n_e * (dE_x_dx + dE_y_dy)) 
    return Ex, Ey, div_neE, phi

def compute_current(n_e, Ex, Ey, diff, mu, dx, dy):

    q = 1.6e-19
    d_n_e_dy, d_n_e_dx = np.gradient(n_e, dy, dx)
    jx = -q * diff * d_n_e_dx + q * mu * n_e * Ex
    jy = -q * diff * d_n_e_dy + q * mu * n_e * Ey

    return jx, jy



def initialize_plots(params):
    global fig, ax_list, im_list

    # Create figure and subplots
    fig, ax_list = plt.subplots(2, 3, figsize=(12, 8))
    im_list = []  # Clear the list before creating new images
    step = 25
    # Electron density plot
    ax = ax_list[0, 0]
    im_n_e = ax.imshow(params['n_e'], extent=[params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]], origin='lower', aspect=0.5*Lx/Ly, cmap='RdPu')
    im_n_e.set_clim(0, max_n_e)
    ax_list[0, 0].quiver(X[::step, ::step], Y[::step, ::step], Ex[::step, ::step], Ey[::step, ::step])
    ax.set_ylabel('z (µm)')
    ax.set_title(f"Electron Density at t = {params['t']} (ps)")
    fig.colorbar(im_n_e, ax=ax)
    im_list.append(im_n_e)

    # Hole density plot
    ax = ax_list[0, 1]
    im_n_h = ax.imshow(params['n_h'], extent=[params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]], origin='lower', aspect=0.5*Lx/Ly, cmap='RdPu')
    ax.set_title(f"Hole Density at t = {params['t']} (ps)")
    fig.colorbar(im_n_h, ax=ax)
    im_list.append(im_n_h)

    # Source density plot
    ax = ax_list[0, 2]
    im_source = ax.imshow(params['source_term'], extent=[params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]], origin='lower', aspect=0.5*Lx/Ly, cmap='hot')
    im_source.set_clim(0, max_source)
    ax.set_title(f"Source Density at t = {params['t']} (ps)")
    fig.colorbar(im_source, ax=ax)
    im_list.append(im_source)

    # Jx plot
    ax = ax_list[1, 0]
    im_jx = ax.imshow(params['jx'], extent=[params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]], origin='lower', aspect=0.5*Lx/Ly, cmap='Spectral')
    im_jx.set_clim(-max_jx, max_jx)
    ax.set_xlabel('x (µm)')
    ax.set_ylabel('z (µm)')
    ax.set_title(f"Jx at t = {params['t']} (ps)")
    fig.colorbar(im_jx, ax=ax)
    im_list.append(im_jx)

    # Jy plot
    ax = ax_list[1, 1]
    im_jy = ax.imshow(params['jy'], extent=[params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]], origin='lower', aspect=0.5*Lx/Ly, cmap='Spectral')
    im_jy.set_clim(-max_jy, max_jy)
    ax.set_xlabel('x (µm)')
    ax.set_title(f"Jz at t = {params['t']} (ps)")
    fig.colorbar(im_jy, ax=ax)
    im_list.append(im_jy)

    plt.tight_layout()


def plot_results(params):
    global im_list

    # Clear output for dynamic updates
    clear_output(wait=True)

    for i, key in enumerate(['n_e', 'n_h', 'source_term', 'jx', 'jy']):
        im_list[i].set_data(params[key])
        im_list[i].set_extent([params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]])
        ax_list[i // 3, i % 3].set_title(f"{key.replace('_', ' ').title()} at t = {params['t']:.2f} ps")


# Initialize figure and axis objects globally
fig, ax_list = None, None
im_list = []  # To store image objects
frames = []
# plt.ion()
# Time-stepping loop


# Initialize storage for jx and jy
# Xr, Yr = np.meshgrid(x[x>=0], y[abs(y)<100])
# jx_storage = np.zeros((*np.shape(Xr), Nt))
# jy_storage = np.zeros((*np.shape(Xr), Nt))

max_n_e=16272.605894719136
max_n_h=0
max_source=2898455409.8756537
max_jx=3.687142144646383e-13
max_jy=4.1733494316689525e-13

lu_A = assemble_poisson_matrix(nrows, ncols, hx, hy, eps)

def dndt(t, n_e):
    # Calculate source term
    source_term = nc * alpha / (2 * np.sqrt(2) * np.pi**(3/2) * sigma_y**2 * sigma_t) * np.exp(-alpha * X) * \
                  np.exp(-(Y**2) / (2 * sigma_y**2)) * np.exp(-((t - t0)**2) / (2 * sigma_t**2))

    # Finite difference for second-order derivatives
    d2_n_e_dy2 = (n_e[2:, 1:-1] - 2 * n_e[1:-1, 1:-1] + n_e[:-2, 1:-1]) / dy**2
    d2_n_e_dx2 = (n_e[1:-1, 2:] - 2 * n_e[1:-1, 1:-1] + n_e[1:-1, :-2]) / dx**2    
    dn_dt = np.zeros_like(n_e)
    _, _, div_neE, _ =  compute_divergence_and_field(lu_A, n_e, n_e*0, x, y)
    dn_dt[1:-1, 1:-1] = diff * (d2_n_e_dx2 + d2_n_e_dy2)  + mu * div_neE[1:-1, 1:-1] - gamma * n_e[1:-1, 1:-1] + source_term[1:-1, 1:-1]
    # dn_dt[1:-1, 1:-1] =  mu * div_neE[1:-1, 1:-1]

    dn_dt[0, :] = dn_dt[-1, :] = 0
  
    dn_dt[:, -1] = 0

    # neumann boundary
    dn_dt[:, 0] = dn_dt[:, 1]
    return dn_dt


# Perform the RK4 integration
t = 0.0
for n in range(Nt):
    

    k1 = dt * dndt(t, n_e)
    k2 = dt * dndt(t + dt / 2, n_e + k1 / 2)
    k3 = dt * dndt(t + dt / 2, n_e + k2 / 2)
    k4 = dt * dndt(t + dt, n_e + k3)
    
    n_e = n_e + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    t += dt
    
        # Collect results and plot every 20 steps
    if n % 125 == 0:
        
        print(t , 'ps')
        source_term = nc * alpha / (2 * np.sqrt(2) * np.pi**(3/2) * sigma_y**2 * sigma_t) * np.exp(-alpha * X) * \
                    np.exp(-(Y**2) / (2 * sigma_y**2)) * np.exp(-((t - t0)**2) / (2 * sigma_t**2))


        Ex, Ey, _, _ =  compute_divergence_and_field(lu_A, n_e, n_e*0, x, y)
        jx, jy = compute_current(n_e, Ex, Ey, diff, mu, dx, dy)

        max_n_e = max(max_n_e, np.max(n_e))
        max_n_h = max(max_n_h, np.max(n_h))
        max_source = max(max_source, np.max(source_term))
        max_jx = max(max_jx, np.max(np.abs(jx)))
        max_jy = max(max_jy, np.max(jy))

        params = {
            't': t, 'x': x, 'y': y, 'n_e': n_e, 'n_h': n_h,
            'source_term': source_term, 'Ex': Ex, 'Ey': Ey, 'X': X, 'Y': Y,
            'nc': nc, 'jx': jx, 'jy': jy
        }
        if n == 0:
            print("init")
            initialize_plots(params)
            buf = BytesIO()
        else:
            plot_results(params)
            buf = BytesIO()

        

        
        plt.savefig(buf, format='png', dpi=400)
        buf.seek(0)
        
        # Read the image from the buffer and append it to the frames list
        frames.append(imageio.v2.imread(buf))
        buf.close()
        


        
print(f'max_n_e={max_n_e}')
print(f'max_n_h={max_n_h}')
print(f'max_source={max_source}')
print(f'max_jx={max_jx}')
print(f'max_jy={max_jy}')

imageio.mimsave(os.path.join(output_dir,f'simulation_sigma_t_{sigma_t:.2g}.gif'), frames, fps=5)


