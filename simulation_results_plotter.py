import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def compute_min_max(params):
    """Compute global min and max for each variable over all time frames."""
    min_max_values = {
        'min_n_e': np.min([np.min(frame) for frame in params['saved_results']['n_e']]),
        'max_n_e': np.max([np.max(frame) for frame in params['saved_results']['n_e']]),
        
        'min_n_h': np.min([np.min(frame) for frame in params['saved_results']['n_h']]),
        'max_n_h': np.max([np.max(frame) for frame in params['saved_results']['n_h']]),
        
        'min_source': np.min([np.min(frame) for frame in params['saved_results']['source_term']]),
        'max_source': np.max([np.max(frame) for frame in params['saved_results']['source_term']]),
        
        # Symmetric ranges for currents
        'max_jx_drift': np.max([np.max(np.abs(frame)) for frame in params['saved_results']['j_drift_x']]),
        'max_jx_diff': np.max([np.max(np.abs(frame)) for frame in params['saved_results']['j_diff_x']]),
        'max_jx': np.max([np.max(np.abs(frame)) for frame in params['saved_results']['j_x']]),
        
        'max_jy_drift': np.max([np.max(np.abs(frame)) for frame in params['saved_results']['j_drift_y']]),
        'max_jy_diff': np.max([np.max(np.abs(frame)) for frame in params['saved_results']['j_diff_y']]),
        'max_jy': np.max([np.max(np.abs(frame)) for frame in params['saved_results']['j_y']]),
    }
    return min_max_values

def compute_min_max_diff(params):
    """Compute global min and max for each variable over all time frames."""
    min_max_values = {
        'min_n_e': np.min([np.min(frame) for frame in params['saved_results']['n_e']]),
        'max_n_e': np.max([np.max(frame) for frame in params['saved_results']['n_e']]),
        
        'min_source': np.min([np.min(frame) for frame in params['saved_results']['source_term']]),
        'max_source': np.max([np.max(frame) for frame in params['saved_results']['source_term']]),
        
        'max_jx_diff': np.max([np.max(np.abs(frame)) for frame in params['saved_results']['j_diff_x']]),

        'max_jy_diff': np.max([np.max(np.abs(frame)) for frame in params['saved_results']['j_diff_y']]),
    }
    return min_max_values

def create_gif(params, gif_path='simulation.gif', xmax = None):
    """Create a GIF from simulation data."""

    if xmax is None:
        xmax = params['Lx']
        
    min_max = compute_min_max(params)
    
    fig, ax_list = plt.subplots(3, 3, figsize=(15, 10))
    fig.suptitle(f"Simulation Results: n0={params.get('nc', 'unknown')}")
    
    # Set up initial plots with placeholders and colorbars
    im_n_e = ax_list[0, 0].imshow(params['saved_results']['n_e'][0], origin='lower', aspect=0.5 * xmax/params['Ly'], extent=[params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]], cmap='RdPu',
                                  vmin=min_max['min_n_e'], vmax=min_max['max_n_e'])
    ax_list[0, 0].set_title("Electron Density")
    ax_list[0, 0].set_ylabel('z (µm)')
    ax_list[0, 0].set_xlim(0, xmax)
    fig.colorbar(im_n_e, ax=ax_list[0, 0])

    im_n_h = ax_list[0, 1].imshow(params['saved_results']['n_h'][0], origin='lower', aspect=0.5 * xmax/params['Ly'], extent=[params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]], cmap='RdPu',
                                  vmin=min_max['min_n_h'], vmax=min_max['max_n_h'])
    ax_list[0, 1].set_title("Hole Density")
    ax_list[0, 1].set_xlim(0, xmax)
    fig.colorbar(im_n_h, ax=ax_list[0, 1])

    im_source = ax_list[0, 2].imshow(params['saved_results']['source_term'][0], origin='lower', aspect=0.5 * xmax/params['Ly'], extent=[params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]], cmap='hot',
                                     vmin=min_max['min_source'], vmax=min_max['max_source'])
    ax_list[0, 2].set_title("Source Term")
    ax_list[0, 2].set_xlim(0, xmax)
    fig.colorbar(im_source, ax=ax_list[0, 2])

    # Set symmetric color scales for currents
    im_jx_drift = ax_list[1, 0].imshow(params['saved_results']['j_drift_x'][0], origin='lower', aspect=0.5 * xmax/params['Ly'], extent=[params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]], cmap='Spectral',
                                       vmin=-min_max['max_jx_drift'], vmax=min_max['max_jx_drift'])
    ax_list[1, 0].set_title("Jx Drift")
    ax_list[1, 0].set_ylabel('z (µm)')
    ax_list[1, 0].set_xlim(0, xmax)
    fig.colorbar(im_jx_drift, ax=ax_list[1, 0])

    im_jx_diffusion = ax_list[1, 1].imshow(params['saved_results']['j_diff_x'][0], origin='lower', aspect=0.5 * xmax/params['Ly'], extent=[params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]], cmap='Spectral',
                                           vmin=-min_max['max_jx_diff'], vmax=min_max['max_jx_diff'])
    ax_list[1, 1].set_title("Jx Diffusion")
    ax_list[1, 1].set_xlim(0, xmax)
    fig.colorbar(im_jx_diffusion, ax=ax_list[1, 1])

    im_jx_total = ax_list[1, 2].imshow(params['saved_results']['j_x'][0], origin='lower', aspect=0.5 * xmax/params['Ly'], extent=[params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]], cmap='Spectral',
                                       vmin=-min_max['max_jx'], vmax=min_max['max_jx'])
    ax_list[1, 2].set_title("Jx Total")
    ax_list[1, 2].set_xlim(0, xmax)
    fig.colorbar(im_jx_total, ax=ax_list[1, 2])

    im_jy_drift = ax_list[2, 0].imshow(params['saved_results']['j_drift_y'][0], origin='lower', aspect=0.5 * xmax/params['Ly'], extent=[params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]], cmap='Spectral',
                                       vmin=-min_max['max_jy_drift'], vmax=min_max['max_jy_drift'])
    ax_list[2, 0].set_title("Jy Drift")
    ax_list[2, 0].set_xlabel('x (µm)')
    ax_list[2, 0].set_ylabel('z (µm)')
    ax_list[2, 0].set_xlim(0, xmax)
    fig.colorbar(im_jy_drift, ax=ax_list[2, 0])

    im_jy_diffusion = ax_list[2, 1].imshow(params['saved_results']['j_diff_y'][0], origin='lower', aspect=0.5 * xmax/params['Ly'], extent=[params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]], cmap='Spectral',
                                           vmin=-min_max['max_jy_diff'], vmax=min_max['max_jy_diff'])
    ax_list[2, 1].set_title("Jy Diffusion")
    ax_list[2, 1].set_xlabel('x (µm)')
    ax_list[2, 1].set_xlim(0, xmax)
    fig.colorbar(im_jy_diffusion, ax=ax_list[2, 1])

    im_jy_total = ax_list[2, 2].imshow(params['saved_results']['j_y'][0], origin='lower', aspect=0.5 * xmax/params['Ly'], extent=[params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]], cmap='Spectral',
                                       vmin=-min_max['max_jy'], vmax=min_max['max_jy'])
    ax_list[2, 2].set_title("Jy Total")
    ax_list[2, 2].set_xlabel('x (µm)')
    ax_list[2, 2].set_xlim(0, xmax)
    fig.colorbar(im_jy_total, ax=ax_list[2, 2])

    # Update function for animation
    def update(frame_idx):
        im_n_e.set_data(params['saved_results']['n_e'][frame_idx])
        im_n_h.set_data(params['saved_results']['n_h'][frame_idx])
        im_source.set_data(params['saved_results']['source_term'][frame_idx])
        im_jx_drift.set_data(params['saved_results']['j_drift_x'][frame_idx])
        im_jx_diffusion.set_data(params['saved_results']['j_diff_x'][frame_idx])
        im_jx_total.set_data(params['saved_results']['j_x'][frame_idx])
        im_jy_drift.set_data(params['saved_results']['j_drift_y'][frame_idx])
        im_jy_diffusion.set_data(params['saved_results']['j_diff_y'][frame_idx])
        im_jy_total.set_data(params['saved_results']['j_y'][frame_idx])

        # Update titles to reflect current time
        current_time = params['saved_results']['time'][frame_idx]
        ax_list[0, 0].set_title(f"Electron Density at t = {current_time:.2f} ps")
        ax_list[0, 1].set_title(f"Hole Density at t = {current_time:.2f} ps")
        ax_list[0, 2].set_title(f"Source Term at t = {current_time:.2f} ps")
        ax_list[1, 0].set_title(f"Jx Drift at t = {current_time:.2f} ps")
        ax_list[1, 1].set_title(f"Jx Diffusion at t = {current_time:.2f} ps")
        ax_list[1, 2].set_title(f"Jx Total at t = {current_time:.2f} ps")
        ax_list[2, 0].set_title(f"Jy Drift at t = {current_time:.2f} ps")
        ax_list[2, 1].set_title(f"Jy Diffusion at t = {current_time:.2f} ps")
        ax_list[2, 2].set_title(f"Jy Total at t = {current_time:.2f} ps")

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(params['saved_results']['time']), repeat=False)
    anim.save(gif_path, writer=PillowWriter(fps=1))
    plt.close(fig)

def create_gif_diffusion(params, gif_path, xmax=None):
    """Create a GIF from simulation data."""

    if xmax is None:
        xmax = params['Lx']
        
    min_max = compute_min_max_diff(params)
    
    fig, ax_list = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"Simulation Results: n0={params.get('nc', 'unknown')}")
    
    # Set up initial plots with placeholders and colorbars
    im_n_e = ax_list[0, 0].imshow(
        params['saved_results']['n_e'][0], origin='lower', 
        aspect=0.5 * xmax / params['Ly'], 
        extent=[params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]], 
        cmap='RdPu', vmin=min_max['min_n_e'], vmax=min_max['max_n_e']
    )
    ax_list[0, 0].set_title("Electron Density")
    ax_list[0, 0].set_ylabel('z (µm)')
    ax_list[0, 0].set_xlim(0, xmax)
    fig.colorbar(im_n_e, ax=ax_list[0, 0])

    im_source = ax_list[0, 1].imshow(
        params['saved_results']['source_term'][0], origin='lower', 
        aspect=0.5 * xmax / params['Ly'], 
        extent=[params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]], 
        cmap='hot', vmin=min_max['min_source'], vmax=min_max['max_source']
    )
    ax_list[0, 1].set_title("Source Term")
    ax_list[0, 1].set_xlim(0, xmax)
    fig.colorbar(im_source, ax=ax_list[0, 1])

    im_jx_diffusion = ax_list[1, 0].imshow(
        params['saved_results']['j_diff_x'][0], origin='lower', 
        aspect=0.5 * xmax / params['Ly'], 
        extent=[params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]], 
        cmap='Spectral', vmin=-min_max['max_jx_diff'], vmax=min_max['max_jx_diff']
    )
    ax_list[1, 0].set_title("Jx Diffusion")
    ax_list[1, 0].set_xlabel('x (µm)')
    ax_list[1, 0].set_ylabel('z (µm)')
    ax_list[1, 0].set_xlim(0, xmax)
    fig.colorbar(im_jx_diffusion, ax=ax_list[1, 0])

    im_jy_diffusion = ax_list[1, 1].imshow(
        params['saved_results']['j_diff_y'][0], origin='lower', 
        aspect=0.5 * xmax / params['Ly'], 
        extent=[params['x'][0], params['x'][-1], params['y'][0], params['y'][-1]], 
        cmap='Spectral', vmin=-min_max['max_jy_diff'], vmax=min_max['max_jy_diff']
    )
    ax_list[1, 1].set_title("Jy Diffusion")
    ax_list[1, 1].set_xlabel('x (µm)')
    ax_list[1, 1].set_xlim(0, xmax)
    fig.colorbar(im_jy_diffusion, ax=ax_list[1, 1])

    # Update function for animation
    def update(frame_idx):
        im_n_e.set_data(params['saved_results']['n_e'][frame_idx])
        im_source.set_data(params['saved_results']['source_term'][frame_idx])
        im_jx_diffusion.set_data(params['saved_results']['j_diff_x'][frame_idx])
        im_jy_diffusion.set_data(params['saved_results']['j_diff_y'][frame_idx])

        # Update titles to reflect current time
        current_time = params['saved_results']['time'][frame_idx]
        ax_list[0, 0].set_title(f"Electron Density at t = {current_time:.2f} ps")
        ax_list[0, 1].set_title(f"Source Term at t = {current_time:.2f} ps")
        ax_list[1, 0].set_title(f"Jx Diffusion at t = {current_time:.2f} ps")
        ax_list[1, 1].set_title(f"Jy Diffusion at t = {current_time:.2f} ps")

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(params['saved_results']['time']), repeat=False)
    anim.save(gif_path, writer=PillowWriter(fps=1))
    plt.close(fig)


# Example usage:
# params = {...}
# saved_results = {...}
# create_gif(params, saved_results, gif_path='simulation_results.gif')
