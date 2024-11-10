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
from scipy.sparse.linalg import splu, bicg, spsolve
import os
from scipy.special import erf
import warnings

output_dir = './current_densities/eps12/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

nc = 4e10
alpha = 7
gamma = 3.3
sigma_y = 40/np.sqrt(8 * np.log(2))
diff = 0.1
t_r = 4e3




Lx = 2
Ly = 200
T = 1
Nx = 7
Ny = 7
Nt = 5000
t0 = 0.1
sigma_t = 50/np.sqrt(8 * np.log(2)) * 1e-3
mu = 4

dt = T / Nt


x = np.linspace(0, Lx, Nx)
xe = np.linspace(-Lx, Lx, 2 * Nx  - 1)
y = np.linspace(-Ly, Ly, Ny)
hx = x[1] - x[0]
hy = y[1] - y[0]
if diff * dt / hx**2 > 0.5 or diff * dt / hy**2 > 0.5:
    raise ValueError('The solution is unstable. Reduce dt or increase dx and dy.')
X, Y = np.meshgrid(x, y)
# XE, YE = np.meshgrid(xe, y)

# eps = np.ones(XE.shape)
# eps[XE > 0] *= 12
# nrows, ncols = XE.shape

eps = np.ones(X.shape) * 12
nrows, ncols = X.shape



def cumulative_generated_carriers(X, alpha, Y, sigma_y, nc, T, t0, sigma_t):
    # Calculate the expression
    result = (np.exp(-X * alpha - Y**2 / (2 * sigma_y**2)) * nc * alpha * (1 + erf((T - t0) / (np.sqrt(2) * sigma_t)))) / (4 * np.pi * sigma_y**2)
    return result


n_e = cumulative_generated_carriers(X, alpha, Y, sigma_y, nc, 0, t0, sigma_t)

n_h = cumulative_generated_carriers(X, alpha, Y, sigma_y, nc, 0, t0, sigma_t)



# def assemble_poisson_matrix(nrows, ncols, hx, hy, eps_in):
#     A = lil_matrix((nrows * ncols, nrows * ncols))
#     eps = eps_in.ravel()[:]
#     for i in range(nrows):
#         for j in range(ncols):
#             k = i * ncols + j
#             if i == 0 or i == nrows - 1 or j == 0 or j == ncols - 1:
#                 A[k, k] = 1
#                 continue
#             A[k, k] = -0.5 * (eps[k-1] + eps[k+1] + 2 * eps[k]) / hx**2 - 0.5 * (eps[k-ncols] + eps[k+ncols] + 2 * eps[k]) / hy**2
#             A[k, k + 1] = 1 / hx**2 * (eps[k] + eps[k+1]) * 0.5
#             A[k, k - 1] = 1 / hx**2 * (eps[k] + eps[k-1]) * 0.5
#             A[k, k + ncols] = 1 / hy**2 * (eps[k] + eps[k+ncols]) * 0.5
#             A[k, k - ncols] = 1 / hy**2 * (eps[k] + eps[k-ncols]) * 0.5
#     A = A.tocsc()
#     lu_A = splu(A)
#     return lu_A


def assemble_poisson_matrix_neumann(nrows, ncols, hx, hy, eps_in):
    A = lil_matrix((nrows * ncols, nrows * ncols))
    eps = eps_in.ravel()[:]
    for i in range(0,nrows):
        for j in range(0,ncols):
            k = i * ncols + j


            if j == 0 and (i != 0 and i != nrows - 1):
                A[k, k] = -0.5 * (eps[k+1] + eps[k+1] + 2 * eps[k]) / hx**2 - 0.5 * (eps[k-ncols] + eps[k+ncols] + 2 * eps[k]) / hy**2
                A[k, k + 1] = 2 / hx**2 * (eps[k] + eps[k+1]) * 0.5
                A[k, k + ncols] = 1 / hy**2 * (eps[k] + eps[k+ncols]) * 0.5
                A[k, k - ncols] = 1 / hy**2 * (eps[k] + eps[k-ncols]) * 0.5
                continue

            if (i == 0 or i == nrows - 1 or j == ncols - 1):
                A[k, k] = 1
                continue
                
            A[k, k] = -0.5 * (eps[k-1] + eps[k+1] + 2 * eps[k]) / hx**2 - 0.5 * (eps[k-ncols] + eps[k+ncols] + 2 * eps[k]) / hy**2
            A[k, k + 1] = 1 / hx**2 * (eps[k] + eps[k+1]) * 0.5
            A[k, k - 1] = 1 / hx**2 * (eps[k] + eps[k-1]) * 0.5
            A[k, k + ncols] = 1 / hy**2 * (eps[k] + eps[k+ncols]) * 0.5
            A[k, k - ncols] = 1 / hy**2 * (eps[k] + eps[k-ncols]) * 0.5


    A = A.tocsc()
    lu_A = splu(A)
    return lu_A, A


# def compute_divergence_and_field(lu_A, n_e_in, n_h_in, x, y, phi_in):


#     # if n % 50 == 0:

#     #     print(n)

#     epsilon_0 = 8.854e-18
#     q_e = 1.60217662e-19
#     n_e = n_e_in
#     n_h = n_h_in

#     # n_h[:,0] = n_h[:,1]
#     # n_e[:,0] = n_e[:,1]
#     rho_s = (n_e - n_h) * q_e 
     

#     # zeroblock = np.zeros((Nx - 1, Ny))
#     # rho = np.concatenate((zeroblock, rho_s.T), axis=0).T
#     rho = rho_s
#     rho[:,0] = rho[:,1]
#     rho[:,-1] = 0
#     rho[0,:] = 0
#     rho[-1,:] = 0
    
#     b = rho.ravel()[:] / epsilon_0 
#     # phi = lu_A.solve(rho.ravel()[:] / epsilon_0).reshape(X.shape)
#     x0 = n_e.ravel()[:] / epsilon_0
    
#     phi = spsolve(A, b)
#     max_iterations = 10
#     # for i in range(max_iterations):

#     #     # Compute the residual
#     #     res = b - A.dot(phi)
        
#     #     # Solve for the correction term
#     #     delta_phi = spsolve(A, res)
        
#     #     # Update the solution
#     #     phi += delta_phi
        
#     #     # Check the residual norm
#     #     residual_norm = np.linalg.norm(res)
#     #     print(f"Iteration {i+1}, Residual norm: {residual_norm}")
        
#     #     if residual_norm < 1e-2:  # Stopping criterion
#     #         break
#     # phi, exitCode = bicg(A, b, atol=1e-5)

#     # if exitCode != 0:
#     #     print("Bicg exited with error code", exitCode)

#     phi = phi.reshape(X.shape)
#     # phi[0,0] = phi[1,1]
#     # phi[-1,-1] = phi[-2,-2]

#     # phi[-1,0] = phi[-2,1]
#     # phi[0,-1] = phi[1,-2]
    

#     dx = x[1] - x[0]
#     dy = y[1] - y[0]
#     Ey, Ex = np.gradient(-phi, dy, dx)

#     # Ey = Ey_f[:, Nx -1 :]
#     # Ex = Ex_f[:, Nx -1 :]
    
#     dE_x_dy, dE_x_dx = np.gradient(Ex, dy, dx)
#     dE_y_dy, dE_y_dx = np.gradient(Ey, dy, dx)

#     d_n_e_dy, d_n_e_dx = np.gradient(n_e, dy, dx)

#     d_n_e_dx[:,0] = 0
#     dE_x_dx[:,0] = 0
#     div_neE = (Ex * d_n_e_dx + Ey * d_n_e_dy + n_e * (dE_x_dx + dE_y_dy)) 
#     div_neE[:,0] = 0
#     return Ex, Ey, div_neE, phi

# def compute_divergence_and_field(lu_A, n_e_in, n_h_in, x, y):


#     # if n % 50 == 0:

#     #     print(n)

#     epsilon_0 = 8.854e-18
#     q_e = 1.60217662e-19
#     n_e = n_e_in
#     n_h = n_h_in

#     n_h[:,0] = n_h[:,1]
#     n_e[:,0] = n_e[:,1]
#     rho_s = (n_e - n_h) * q_e 

#     zeroblock = np.zeros((Nx - 1, Ny))
#     rho = np.concatenate((zeroblock, rho_s.T), axis=0).T


#     phi = lu_A.solve(rho.ravel()[:] / epsilon_0).reshape(XE.shape)
#     dx = x[1] - x[0]
#     dy = y[1] - y[0]
#     Ey_f, Ex_f = np.gradient(-phi, dy, dx)

#     Ey = Ey_f[:, Nx -1 :]
#     Ex = Ex_f[:, Nx -1 :]
    
#     dE_x_dy, dE_x_dx = np.gradient(Ex, dy, dx)
#     dE_y_dy, dE_y_dx = np.gradient(Ey, dy, dx)

#     d_n_e_dy, d_n_e_dx = np.gradient(n_e, dy, dx)

#     d_n_e_dx[:,0] = 0
#     dE_x_dx[:,0] = 0
#     div_neE = (Ex * d_n_e_dx + Ey * d_n_e_dy + n_e * (dE_x_dx + dE_y_dy)) 
#     div_neE[:,0] = 0
#     return Ex, Ey, div_neE, phi

def compute_current(n_e, Ex, Ey, diff, mu, dx, dy):

    q = 1.6e-19
    d_n_e_dy, d_n_e_dx = np.gradient(n_e, dy, dx)
 
    jx = {'drift': + q * mu * n_e * Ex, 'diffusion': -q * diff * d_n_e_dx}
    jy = {'drift': + q * mu * n_e * Ey, 'diffusion': -q * diff * d_n_e_dy}
    return jx, jy



def initialize_plots(params):
    global fig, ax_list, im_list, quiver

    # Create figure and subplots
    fig, ax_list = plt.subplots(3, 3, figsize=(15, 10))
    im_list = []  # Clear the list before creating new images
    fig.suptitle(f"n0={nc:.2g}")
    step = 10
    
    # Electron density plot
    ax = ax_list[0, 0]
    im_n_e = ax.imshow(params['n_e'], extent=[params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]], 
                       origin='lower', aspect=0.5*Lx/Ly, cmap='RdPu')
    im_n_e.set_clim(0, max_n_e)
    
    # Store quiver plot in global variable
    sx = 2
    quiver = ax.quiver(X[::step * sx, ::step], Y[::step * sx, ::step], Ex[::step * sx, ::step]*500, Ey[::step * sx, ::step]*500)

    ax.set_ylabel('z (µm)')
    ax.set_title(f"Electron Density at t = {params['t']} (ps)")
    fig.colorbar(im_n_e, ax=ax)
    im_list.append(im_n_e)

    # Hole density plot
    ax = ax_list[0, 1]
    im_n_h = ax.imshow(params['n_h'], extent=[params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]], 
                       origin='lower', aspect=0.5*Lx/Ly, cmap='RdPu')
    im_n_h.set_clim(0, max_n_h)
    ax.set_title(f"Hole Density at t = {params['t']} (ps)")
    fig.colorbar(im_n_h, ax=ax)
    im_list.append(im_n_h)

    # Source density plot
    ax = ax_list[0, 2]
    im_source = ax.imshow(params['source_term'], extent=[params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]], 
                          origin='lower', aspect=0.5*Lx/Ly, cmap='hot')
    im_source.set_clim(0, max_source)
    ax.set_title(f"Source Density at t = {params['t']} (ps)")
    fig.colorbar(im_source, ax=ax)
    im_list.append(im_source)

    # Drift and Diffusion components of Jx
    ax = ax_list[1, 0]
    im_jx_drift = ax.imshow(params['jx_drift'], extent=[params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]], 
                            origin='lower', aspect=0.5*Lx/Ly, cmap='Spectral')
    im_jx_drift.set_clim(-max_jx_drift, max_jx_drift)
    ax.set_xlabel('x (µm)')
    ax.set_ylabel('z (µm)')
    ax.set_title(f"Jx Drift at t = {params['t']} (ps)")
    fig.colorbar(im_jx_drift, ax=ax)
    im_list.append(im_jx_drift)

    ax = ax_list[1, 1]
    im_jx_diffusion = ax.imshow(params['jx_diffusion'], extent=[params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]], 
                                origin='lower', aspect=0.5*Lx/Ly, cmap='Spectral')
    im_jx_diffusion.set_clim(-max_jx_diffusion, max_jx_diffusion)
    ax.set_xlabel('x (µm)')
    ax.set_title(f"Jx Diffusion at t = {params['t']} (ps)")
    fig.colorbar(im_jx_diffusion, ax=ax)
    im_list.append(im_jx_diffusion)

    ax = ax_list[1, 2]
    im_jx_total = ax.imshow(params['jx_total'], extent=[params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]], 
                            origin='lower', aspect=0.5*Lx/Ly, cmap='Spectral')
    im_jx_total.set_clim(-max_jx_total, max_jx_total)
    ax.set_xlabel('x (µm)')
    ax.set_title(f"Jx Total at t = {params['t']} (ps)")
    fig.colorbar(im_jx_total, ax=ax)
    im_list.append(im_jx_total)

    # Drift and Diffusion components of Jy
    ax = ax_list[2, 0]
    im_jy_drift = ax.imshow(params['jy_drift'], extent=[params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]], 
                            origin='lower', aspect=0.5*Lx/Ly, cmap='Spectral')
    im_jy_drift.set_clim(-max_jy_drift, max_jy_drift)
    ax.set_xlabel('x (µm)')
    ax.set_ylabel('z (µm)')
    ax.set_title(f"Jy Drift at t = {params['t']} (ps)")
    fig.colorbar(im_jy_drift, ax=ax)
    im_list.append(im_jy_drift)

    ax = ax_list[2, 1]
    im_jy_diffusion = ax.imshow(params['jy_diffusion'], extent=[params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]], 
                                origin='lower', aspect=0.5*Lx/Ly, cmap='Spectral')
    im_jy_diffusion.set_clim(-max_jy_diffusion, max_jy_diffusion)
    ax.set_xlabel('x (µm)')
    ax.set_title(f"Jy Diffusion at t = {params['t']} (ps)")
    fig.colorbar(im_jy_diffusion, ax=ax)
    im_list.append(im_jy_diffusion)

    ax = ax_list[2, 2]
    im_jy_total = ax.imshow(params['jy_total'], extent=[params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]], 
                            origin='lower', aspect=0.5*Lx/Ly, cmap='Spectral')
    im_jy_total.set_clim(-max_jy_total, max_jy_total)
    ax.set_xlabel('x (µm)')
    ax.set_title(f"Jy Total at t = {params['t']} (ps)")
    fig.colorbar(im_jy_total, ax=ax)
    im_list.append(im_jy_total)

    plt.tight_layout()

def plot_results(params):
    global im_list, plot_config, quiver  # Ensure plot_config and quiver are accessible

    # Clear output for dynamic updates
    clear_output(wait=True)

    # Update the quiver plot for Ex and Ey
    max_E_magnitude = np.max(np.sqrt(params['Ex']**2 + params['Ey']**2))
    dynamic_scale = 0.5 if max_E_magnitude == 0 else 0.1 / max_E_magnitude
    step = 10
    sx = 2
    scale = 0.0025 / max_E_magnitude
    quiver.set_UVC(params['Ex'][::step * sx, ::step] * scale, params['Ey'][::step * sx, ::step] * scale)

    # Loop over each item in the plot configuration and update the corresponding plot
    for key, config in plot_config.items():
        i = config["index"]

        # # Check if we're dealing with a component plot (drift/diffusion/total)
        # if key in ['jx_drift', 'jx_diffusion', 'jx_total', 'jy_drift', 'jy_diffusion', 'jy_total']:
        #     # Determine the component type and the main current direction (jx or jy)
        #     component_type = key.split('_')[-1]  # 'drift', 'diffusion', or 'total'
        #     current_key = key.split('_')[0]      # 'jx' or 'jy'

        #     # Retrieve the appropriate component data
        #     if component_type == 'drift':
        #         data = params[f'{current_key}_drift']
        #     elif component_type == 'diffusion':
        #         data = params[f'{current_key}_diffusion']
        #     else:  # 'total' combines drift and diffusion
        #         data = params[f'{current_key}_drift'] + params[f'{current_key}_diffusion']

        #     # Update the plot data for this component
        #     im_list[i].set_data(data)
        #     im_list[i].set_extent([params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]])
        #     ax_list[i // 3, i % 3].set_title(f"{config['title']} at t = {params['t']:.2f} ps")
        
        # else:
        #     # For non-component keys, update plots directly
        im_list[i].set_data(params[key])
        im_list[i].set_extent([params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]])
        ax_list[i // 3, i % 3].set_title(f"{config['title']} at t = {params['t']:.2f} ps")

    # Redraw updated plots
    plt.draw()

def dhdt(t, n_h):
    # Calculate source term
    source_term = nc * alpha / (2 * np.sqrt(2) * np.pi**(3/2) * sigma_y**2 * sigma_t) * np.exp(-alpha * X) * \
                  np.exp(-(Y**2) / (2 * sigma_y**2)) * np.exp(-((t - t0)**2) / (2 * sigma_t**2))


    dh_dt = np.zeros_like(n_h)


    dh_dt[1:-1, 1:-1] = - gamma * n_h[1:-1, 1:-1] + source_term[1:-1, 1:-1]
    # dn_dt[1:-1, 1:-1] =  mu * div_neE[1:-1, 1:-1]

    dh_dt[0, :] = dh_dt[-1, :] = 0
  
    dh_dt[:, -1] = 0

    # neumann boundary
    dh_dt[:, 0] = dh_dt[:, 1]
    return dh_dt

# def diffusion_current():

def compute_electric_field(n_e, n_h, params):
    # compute electric field
    q_e = 1.6e-19
    epsilon_0 = 8.854e-18

    rho = (n_e - n_h) * q_e
    b = rho.ravel()[:] / epsilon_0 
    
    phi = spsolve(params['A'], b).reshape(params['X'].shape)

    ey, ex = np.gradient(-phi, params['hy'], params['hx'])

    return ex, ey


def compute_current_gradient_term(n_e, n_h, params):

    # compute electric field
    ex, ey = compute_electric_field(n_e, n_h, params)
    # calculate j
    ne_dy, ne_dx = np.gradient(n_e, params['hy'], params['hx'])
    j_x = params['diff'] * ne_dx + n_e * params['mu'] * ex
    j_y = params['diff'] * ne_dy + n_e * params['mu'] * ey

    j_x[0, :] = j_x[1, :]

    _, grad_jx_dx = np.gradient(j_x, params['hy'], params['hx'])
    grad_jy_dy, _ = np.gradient(j_y, params['hy'], params['hx'])

    return grad_jx_dx + grad_jy_dy

def compute_source_term(t, params):

    # Unpack parameters
    nc = params['nc']
    alpha = params['alpha']
    sigma_y = params['sigma_y']
    sigma_t = params['sigma_t']
    t0 = params['t0']
    X = params['X']
    Y = params['Y']
    # Calculate the source term
    source_term = (nc * alpha / (2 * np.sqrt(2) * np.pi**(3/2) * sigma_y**2 * sigma_t) *
                   np.exp(-alpha * X) *
                   np.exp(-(Y**2) / (2 * sigma_y**2)) *
                   np.exp(-((t - t0)**2) / (2 * sigma_t**2)))
    
    return source_term

def compute_decay_term(n, params):

    # compute decay term
    return params['gamma'] * n



def dndt(t, n_e, n_h, params):

    current_gradient_term = compute_current_gradient_term(n_e, n_h, params)

    generation_term = compute_source_term(t, params)

    decay_term = compute_decay_term(n_e, params)

    return current_gradient_term + generation_term - decay_term


def dpdt(t, n_h, params):


    generation_term = compute_source_term(t, params)

    decay_term = compute_decay_term(n_h, params)

    return generation_term - decay_term

# def dndt(t, n_e, n_h, phi_in):
#     # Calculate source term

#     source_term = (nc * alpha / (2 * np.sqrt(2) * np.pi**(3/2) * sigma_y**2 * sigma_t) *
#                    np.exp(-alpha * X) *
#                    np.exp(-(Y**2) / (2 * sigma_y**2)) *
#                    np.exp(-((t - t0)**2) / (2 * sigma_t**2)))
    


#     # Finite difference for second-order derivatives
#     d2_n_e_dy2 = (n_e[2:, 1:-1] - 2 * n_e[1:-1, 1:-1] + n_e[:-2, 1:-1]) / hy**2
#     d2_n_e_dx2 = (n_e[1:-1, 2:] - 2 * n_e[1:-1, 1:-1] + n_e[1:-1, :-2]) / hx**2    
#     dn_dt = np.zeros_like(n_e)


#     _, _, div_neE, phi_in =  compute_divergence_and_field(lu_A, n_e, n_h, x, y , phi_in)
#     dn_dt[1:-1, 1:-1] = diff * (d2_n_e_dx2 + d2_n_e_dy2)  + mu * div_neE[1:-1, 1:-1] - gamma * n_e[1:-1, 1:-1] + source_term[1:-1, 1:-1]
#     # dn_dt[1:-1, 1:-1] =  mu * div_neE[1:-1, 1:-1]

#     dn_dt[0, :] = dn_dt[-1, :] = 0
  
#     dn_dt[:, -1] = 0

#     # neumann boundary
#     dn_dt[:, 0] = dn_dt[:, 1]
#     return dn_dt, phi_in


# def dndt(t, n_e):
#     # Calculate source term
#     source_term = nc * alpha / (2 * np.sqrt(2) * np.pi**(3/2) * sigma_y**2 * sigma_t) * np.exp(-alpha * X) * \
#                   np.exp(-(Y**2) / (2 * sigma_y**2)) * np.exp(-((t - t0)**2) / (2 * sigma_t**2))

#     # # Finite difference for second-order derivatives
#     # d2_n_e_dy2 = (n_e[2:, 1:-1] - 2 * n_e[1:-1, 1:-1] + n_e[:-2, 1:-1]) / hy**2
#     # d2_n_e_dx2 = (n_e[1:-1, 2:] - 2 * n_e[1:-1, 1:-1] + n_e[1:-1, :-2]) / hx**2    
#     dn_dt = np.zeros_like(n_e)


#     # _, _, div_neE, _ =  compute_divergence_and_field(lu_A, n_e, n_h, x, y)
#     dn_dt[1:-1, 1:-1] = - gamma * n_e[1:-1, 1:-1] + source_term[1:-1, 1:-1]
#     # dn_dt[1:-1, 1:-1] =  mu * div_neE[1:-1, 1:-1]

#     dn_dt[0, :] = dn_dt[-1, :] = 0
  
#     dn_dt[:, -1] = 0

#     # neumann boundary
#     dn_dt[:, 0] = dn_dt[:, 1]
#     return dn_dt


fig, ax_list = None, None
im_list = []  # To store image objects
frames = []

max_n_e=0
max_n_h=0
max_source=0
max_jx_drift=0
max_jx_diffusion=0
max_jx_total=0
max_jy_drift=0
max_jy_diffusion=0
max_jy_total=0
n_e_mat = []
n_h_mat = []
source_mat = []
jx_drift_mat = []
jx_diffusion_mat = []
jx_total_mat = []
jy_drift_mat = []
jy_diffusion_mat = []
jy_total_mat = []
t_mat = []
Ex_mat = []
Ey_mat = []


plot_config = {
    'n_e': {'index': 0, 'title': 'Electron Density'},
    'n_h': {'index': 1, 'title': 'Hole Density'},
    'source_term': {'index': 2, 'title': 'Source Density'},
    'jx_drift': {'index': 3, 'title': 'Jx (Drift)'},
    'jx_diffusion': {'index': 4, 'title': 'Jx (Diffusion)'},
    'jx_total': {'index': 5, 'title': 'Jx (Total)'},
    'jy_drift': {'index': 6, 'title': 'Jy (Drift)'},
    'jy_diffusion': {'index': 7, 'title': 'Jy (Diffusion)'},
    'jy_total': {'index': 8, 'title': 'Jy (Total)'}
}


# Perform the RK4 integration
t = 0.0

max_e = 0

lu_A, A = assemble_poisson_matrix_neumann(nrows, ncols, hx, hy, eps)

phi_in = (n_e - n_h - 1e5)  * 1.6e19 / 8.854e-18 * 100
for n in range(Nt//2):
    

    k1, phi_in = dt * dndt(t, n_e, n_h, phi_in)
    p1 = dt * dhdt(t, n_h)

    k2, phi_in = dt * dndt(t + dt / 2, n_e + k1 / 2, n_h + p1 / 2, phi_in)
    p2 = dt * dhdt(t + dt / 2, n_h + p1 / 2)

    k3, phi_in = dt * dndt(t + dt / 2, n_e + k2 / 2, n_h + p2 / 2, phi_in)
    p3 = dt * dhdt(t + dt / 2, n_h + p2 / 2)

    k4, phi_in = dt * dndt(t + dt, n_e + k3, n_h + p3, phi_in)
    p4 = dt * dhdt(t + dt, n_h + p3)
    
    n_e = n_e + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    n_h = n_h + (p1 + 2 * p2 + 2 * p3 + p4) / 6

    # k1 = dt * dndt(t, n_e, n_h)
    # p1 = dt * dhdt(t, n_h)

    # k2 = dt * dndt(t + dt / 2, n_e + k1 / 2)
    # p2 = dt * dhdt(t + dt / 2, n_h + p1 / 2)

    # k3 = dt * dndt(t + dt / 2, n_e + k2 / 2)
    # p3 = dt * dhdt(t + dt / 2, n_h + p2 / 2)

    # k4 = dt * dndt(t + dt, n_e + k3)
    # p4 = dt * dhdt(t + dt, n_h + p3)
    
    # n_e = n_e + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    # n_h = n_h + (p1 + 2 * p2 + 2 * p3 + p4) / 6


        # Replace invalid values in n_e_in
    if np.any(n_e < 0):
        warnings.warn("Warning: n_e contained values less than zero, setting them to 0.", stacklevel=2)
        n_e[n_e < 0] = 0
    if np.any(np.isinf(n_e)):
        warnings.warn("Warning: n_e contained infinity (inf) values, setting them to 0.", stacklevel=2)
        n_e[np.isinf(n_e)] = 0
    if np.any(np.isnan(n_e)):
        warnings.warn("Warning: n_e contained NaN values, setting them to 0.", stacklevel=2)
        n_e[np.isnan(n_e)] = 0

    t += dt
    
        # Collect results and plot every 20 steps
    if n % 125 == 0:

        Ex, Ey, _, _ =  compute_divergence_and_field(lu_A, n_e, n_h, x, y, phi_in)
        jx, jy = compute_current(n_e, Ex, Ey, diff, mu, hx,hy)

        source_term = nc * alpha / (2 * np.sqrt(2) * np.pi**(3/2) * sigma_y**2 * sigma_t) * np.exp(-alpha * X) * \
                    np.exp(-(Y**2) / (2 * sigma_y**2)) * np.exp(-((t - t0)**2) / (2 * sigma_t**2))
        
        t_mat.append(t)
        Ex_mat.append(Ex)
        Ey_mat.append(Ey)
        jx_drift_mat.append(jx['drift'])
        jx_diffusion_mat.append(jx['diffusion'])
        jx_total_mat.append(jx['drift'] + jx['diffusion'])
        jy_drift_mat.append(jy['drift'])
        jy_diffusion_mat.append(jy['diffusion'])
        jy_total_mat.append(jy['drift'] + jy['diffusion'])
        n_e_mat.append(n_e)
        n_h_mat.append(n_h)
        source_mat.append(source_term)

        max_e = max(max_e, np.max(np.sqrt(Ex**2 + Ey**2)))
        max_jx_drift = max(max_jx_drift, np.max(np.abs(jx['drift'])))
        max_jx_diffusion = max(max_jx_diffusion, np.max(np.abs(jx['diffusion'])))
        max_jx_total = max(max_jx_total, np.max(np.abs(jx['drift'] + jx['diffusion'])))

        max_jy_drift = max(max_jy_drift, np.max(np.abs(jy['drift'])))
        max_jy_diffusion = max(max_jy_diffusion, np.max(np.abs(jy['diffusion'])))
        max_jy_total = max(max_jy_total, np.max(np.abs(jy['drift'] + jy['diffusion'])))

        max_n_e = max(max_n_e, np.max(n_e))
        max_n_h = max(max_n_h, np.max(n_h))
        max_source = max(max_source, np.max(source_term))


        print(t , 'ps')




for i in range(len(t_mat)):
        
        params = {
            't': t_mat[i], 'x': x, 'y': y, 'n_e': n_e_mat[i], 'n_h': n_h_mat[i],
            'source_term': source_mat[i], 'Ex': Ex_mat[i], 'Ey': Ey_mat[i], 'X': X, 'Y': Y,
            'nc': nc,
            'jx_drift': jx_drift_mat[i], 'jx_diffusion': jx_diffusion_mat[i],
            'jx_total': jx_total_mat[i], 
            'jy_drift': jy_drift_mat[i], 'jy_diffusion': jy_diffusion_mat[i],
            'jy_total': jy_total_mat[i],
        }

        if i == 0:
            print("init")
            initialize_plots(params)
            buf = BytesIO()
        else:
            plot_results(params)
            buf = BytesIO()

        plt.savefig(buf, format='png', dpi=300)
        buf.seek(0)
        
        # Read the image from the buffer and append it to the frames list
        frames.append(imageio.v2.imread(buf))
        buf.close()
        
        
print(max_e)
imageio.mimsave(os.path.join(output_dir,f'simulation_nc_{nc:.2g}.gif'), frames, fps=2)


