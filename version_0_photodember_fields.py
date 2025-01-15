import sys
import os
import meep as mp
import numpy as np
from matplotlib import pyplot as plt
from itertools import chain
from scipy.interpolate import RegularGridInterpolator , interp1d
from scipy.io import savemat
from eels_utils.ElectronSpectrum import ElectronSpectrum as es
from current_interpolation_new import setup_interpolators, get_jx_at, get_jy_at

class ProblemSetup:

    def __init__(self, sx = None, sy = None, dpml = None, resolution = None):

        if sx is None:
            sx = 400
        if sy is None:
            sy = 50
        if dpml is None:
            dpml = 10
        if resolution is None:
            resolution = 5
            
        self._sx = sx
        self._sy = sy
        self._dpml = dpml
        self._resolution = resolution

def indium_arsenide():
    inas = {
        "n_exc_0" : 3.14e23,
        "n_eq" : 1.0e23,
        "gamma_p_sec_1" : 3.3e12, 
        "alpha" : 7,
        "epsilon_inf" : 12, 
        "epsilon_0" : 15,
        "omega_TO_cm_1" : 218, 
        "omega_LO_cm_1" : 240, 
        "gamma_ph_cm_1" : 3.5,
        "v_t_m_sec_1" : 7.66e5}
    return inas
    
def drude_material_meep(n_eq, gamma_p_sec_1, epsilon_inf):

    f_p = 20.7539 * np.sqrt(n_eq) / (2 * np.pi) / Freq_Hz_To_MEEP
    sigma_p = epsilon_inf
    gamma_p = gamma_p_sec_1 / (2 * np.pi) / Freq_Hz_To_MEEP

    return f_p, gamma_p, sigma_p

def lorentz_material_meep(omega_TO_cm_1, gamma_ph_cm_1, epsilon_0, epsilon_inf):

    f_ph = omega_TO_cm_1 * 3e10 / Freq_Hz_To_MEEP
    gamma_ph = gamma_ph_cm_1 * 3e10 / Freq_Hz_To_MEEP
    sigma_ph = epsilon_0 - epsilon_inf

    return f_ph, gamma_ph, sigma_ph

def current_rectification(sigma_t_sec, t0_sec, weight):
    
    t0 = t0_sec / Time_Sec_To_MEEP
    sigma_t = sigma_t_sec / Time_Sec_To_MEEP

    return lambda t: - weight * (t-t0) * np.exp(- ((t-t0) ** 2) / (2 * sigma_t **2))

def current_photodember_diffusion_x(xprime, yprime):
    """Returns a function to get jx at a specific t for given (xprime, yprime)."""
    def jx_t(t):
        return get_jx_at(xprime, yprime, t / Time_MEEP_To_Sec * 1e12, jx_interpolator)
    return jx_t 

def current_photodember_diffusion_y(xprime, yprime):
    """Returns a function to get jy at a specific t for given (xprime, yprime)."""
    def jy_t(t):
        return get_jy_at(xprime, yprime, t / Time_MEEP_To_Sec * 1e12, jy_interpolator)
    return jy_t 
  

def photodember_source_x(params, xmax, ymax):
    sl_y = [[mp.Source(
        src=mp.CustomSource(src_func=current_photodember_diffusion_x(yi, xi),is_integrated=True),
        center=mp.Vector3(xi,-yi),
        component=mp.Ey) for xi in np.arange(-np.fix(xmax),np.fix(xmax) + 0.5, 0.25)] for yi in np.linspace(0,ymax,5)]
    return list(chain(*sl_y))

def photodember_source_z(params, xmax, ymax):
    sl_x = [[mp.Source(
        src=mp.CustomSource(src_func=current_photodember_diffusion_y(yi, xi),is_integrated=True),
        center=mp.Vector3(xi,-yi),
        component=mp.Ex) for xi in np.arange(-np.fix(xmax),np.fix(xmax) + 0.5, 0.25)] for yi in np.linspace(0,ymax,5)]
    return list(chain(*sl_x))


if __name__ == '__main__':

    
    intensity = 10
    t0_ps = 0.5
    t0_sec = t0_ps * 1e-12
    fwhm_t_fs = 50

    
   # currents_nc_4.00e+10_sigma_14.86.npy
    jx_interpolator, jy_interpolator = setup_interpolators(
        "4.00e+10", 
        "14.86", 
        base_dir="saved_matrices", 
        subdirs=("version_0", "diffusion_currents")
    )

    Freq_Hz_To_MEEP = 3 * 10**14
    Time_Sec_To_MEEP = (1e-6 / 3e8)
    Time_MEEP_To_Sec = 1 / Time_Sec_To_MEEP

    ElectricField_MEEP_TO_SI =  (1e-6 * 8.85e-12 * 3e8)

    sx = 400
    sy = 50
    dpml = 10
    resolution = 5

    inas = indium_arsenide()
    f_p, gamma_p, sigma_p = drude_material_meep(inas['n_eq'], inas['gamma_p_sec_1'], inas['epsilon_inf'])
    f_ph, gamma_ph, sigma_ph = lorentz_material_meep(inas['omega_TO_cm_1'], 
    inas['gamma_ph_cm_1'], inas['epsilon_0'], inas['epsilon_inf'])

    sus_plasma = mp.DrudeSusceptibility(frequency=f_p, sigma=sigma_p, gamma=gamma_p)
    sus_phonon = mp.LorentzianSusceptibility(frequency=f_ph, sigma=sigma_ph, gamma=gamma_ph)
    inas_meep = mp.Medium(epsilon=inas['epsilon_inf'], E_susceptibilities=[sus_phonon, sus_plasma])

    params = {
        't0_sec': t0_sec,
        'weight': 1,
        'alpha': inas['alpha'],
        'sigma_spot': 40 / np.sqrt(8 * np.log(2)),
        'neq': inas['n_eq'],
        'nexc_0': inas['n_exc_0'] * intensity / 10 ,
        'gamma': inas['gamma_p_sec_1'],
        'v_t_m_sec_1' : inas['v_t_m_sec_1'],
        'fwhm_t_fs' : fwhm_t_fs
    }

    xmax = 4 * params['sigma_spot']
    ymax = 3 * (1 / params['alpha'])
    source_pd_x = photodember_source_x(params, xmax, ymax)
    source_pd_z = photodember_source_z(params, xmax, ymax)

    geometry = [
    mp.Block(
        mp.Vector3(mp.inf, sy/2 + dpml, mp.inf),
        center=mp.Vector3(0,-sy/4 - dpml/2),
        material=mp.Medium(epsilon=inas['epsilon_inf'])
    )]


    sim_pd_x = mp.Simulation(
        cell_size=mp.Vector3(sx + 2 * dpml, sy + 2 * dpml),
        boundary_layers=[mp.PML(dpml)],
        geometry=geometry,
        sources=source_pd_x,
        resolution=resolution,
        symmetries=None,
        progress_interval = 15
    )

    vals_pd_x = []
    def get_slice(vals, distance_from_surface):
            return lambda sim : vals.append(sim.get_array(center=mp.Vector3(0,distance_from_surface), size=mp.Vector3(sx,0), component=mp.Ex))

    record_interval = 2
    distance_from_surface = 1
    simulation_end_time_meep = 1000

    sim_pd_x.reset_meep()
    sim_pd_x.run(mp.at_every(record_interval, get_slice(vals_pd_x, distance_from_surface)),
            until=simulation_end_time_meep)

    (x,y,z,w)=sim_pd_x.get_array_metadata(center=mp.Vector3(0,1), size=mp.Vector3(sx,0))
   

    sim_pd_z = mp.Simulation(
        cell_size=mp.Vector3(sx + 2 * dpml, sy + 2 * dpml),
        boundary_layers=[mp.PML(dpml)],
        geometry=geometry,
        sources=source_pd_z,
        resolution=resolution,
        symmetries=None,
        progress_interval = 15
    )

    vals_pd_z = []
    def get_slice(vals, distance_from_surface):
            return lambda sim : vals.append(sim.get_array(center=mp.Vector3(0,distance_from_surface), size=mp.Vector3(sx,0), component=mp.Ex))

    record_interval = 2
    distance_from_surface = 1
    simulation_end_time_meep = 1000

    sim_pd_z.reset_meep()
    sim_pd_z.run(mp.at_every(record_interval, get_slice(vals_pd_z, distance_from_surface)),
            until=simulation_end_time_meep)

    (x,y,z,w)=sim_pd_z.get_array_metadata(center=mp.Vector3(0,1), size=mp.Vector3(sx,0))
   

   
    if mp.am_master():
   
        # Check whether the specified path exists or not

        path = 'saved_matrices/' + 'version_0/photodember/'
        isExist = os.path.exists(path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(path)
            print("The new directory is created!")

        plt.figure()
        time_range = simulation_end_time_meep / Time_MEEP_To_Sec * 1e12
        val_np = np.array(vals_pd_x) / ElectricField_MEEP_TO_SI
        # mm = np.max(val_np)

        plt.imshow(val_np.T, 
                cmap='bwr',
                aspect = time_range / sx,
                extent = [0,time_range,-sx/2,sx/2])

        plt.colorbar()
        plt.savefig(path + '/version_0_photodember_ez_jx.mat.png', dpi=300)
        out_str = path + '/version_0_photodember_ez_jx.mat'
        savemat(out_str, {'e_pd': vals_pd_x, 'zstep': x[2]-x[1], 'tstep' : record_interval / Time_MEEP_To_Sec * 1e12})

        plt.figure()
        time_range = simulation_end_time_meep / Time_MEEP_To_Sec * 1e12
        val_np = np.array(vals_pd_z) / ElectricField_MEEP_TO_SI
        # mm = np.max(val_np)

        plt.imshow(val_np.T, 
                cmap='bwr',
                aspect = time_range / sx,
                extent = [0,time_range,-sx/2,sx/2])

        plt.colorbar()
        plt.savefig(path + '/version_0_photodember_ez_jz.mat.png', dpi=300)
        out_str = path + '/version_0_photodember_ez_jz.mat'
        savemat(out_str, {'e_pd': vals_pd_z, 'zstep': x[2]-x[1], 'tstep' : record_interval / Time_MEEP_To_Sec * 1e12})


