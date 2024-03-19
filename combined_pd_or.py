import sys
import os
import meep as mp
import numpy as np
from matplotlib import pyplot as plt
from itertools import chain
from scipy.interpolate import RegularGridInterpolator
from scipy.io import savemat
from eels_utils.ElectronSpectrum import ElectronSpectrum as es


class ProblemSetup:

    def __init__(self, sx = None, sy = None, dpml = None, resolution = None):

        if sx is None:
            sx = 200
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

def current_photodember(params, xprime, yprime):
    nexc = params['nexc_0'] * np.exp(-yprime * params['alpha']) * np.exp(-xprime**2 / (2 * params['sigma_spot']**2))

    omega_eq = 20.7539 * np.sqrt(params['neq'])
    omega_exc = 44.5865 * np.sqrt(nexc)
    omega_z_sec = np.sqrt(omega_exc**2 + omega_eq**2 - params['gamma']**2 / 4)
    omega_z = omega_z_sec / Freq_Hz_To_MEEP
    
    t0 = params['t0_sec'] / Time_Sec_To_MEEP

    prefactor = (nexc) * 1e-18 * params['alpha'] * (params['v_t_m_sec_1'] / 3e8)**2 
    return lambda t: prefactor * np.sin(omega_z * (t-t0)) * np.exp(-(params['gamma'] / Freq_Hz_To_MEEP) * (t-t0)/2) / omega_z if t>t0 else 0


def photodember_source(params, xmax, ymax):
    sl = [[mp.Source(
        src=mp.CustomSource(src_func=current_photodember(params, xi, yi)),
        center=mp.Vector3(xi,-yi),
        component=mp.Ey) for xi in np.arange(-np.fix(xmax),np.fix(xmax) + 0.5, 0.5)] for yi in np.linspace(0,ymax,5)]
    return list(chain(*sl))



if __name__ == '__main__':

    
    outdir = 'photodember/varying_intensities'
    intensity = float(sys.argv[1])
    t0_sec = float(sys.argv[2]) * 1e-12

    Freq_Hz_To_MEEP = 3 * 10**14
    Time_Sec_To_MEEP = (1e-6 / 3e8)
    Time_MEEP_To_Sec = 1 / Time_Sec_To_MEEP

    ElectricField_MEEP_TO_SI =  (1e-6 * 8.85e-12 * 3e8)

    sx = 200
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
        'sigma_spot': 30 / np.sqrt(8 * np.log(2)),
        'neq': inas['n_eq'],
        'nexc_0': inas['n_exc_0'] * intensity / 10,
        'gamma': inas['gamma_p_sec_1'],
        'v_t_m_sec_1' : inas['v_t_m_sec_1']
    }

    xmax = 4 * params['sigma_spot']
    ymax = 3 * (1 / params['alpha'])
    source_pd = photodember_source(params, xmax, ymax)

    geometry = [
    mp.Block(
        mp.Vector3(mp.inf, sy/2 + dpml, mp.inf),
        center=mp.Vector3(0,-sy/4 - dpml/2),
        material=inas_meep,
    )]

    sim_pd = mp.Simulation(
        cell_size=mp.Vector3(sx + 2 * dpml, sy + 2 * dpml),
        boundary_layers=[mp.PML(dpml)],
        geometry=geometry,
        sources=source_pd,
        resolution=resolution,
        symmetries=None,
        progress_interval = 15
    )

    vals_pd = []
    def get_slice(vals, distance_from_surface):
            return lambda sim : vals.append(sim.get_array(center=mp.Vector3(0,distance_from_surface), size=mp.Vector3(sx,0), component=mp.Ex))

    record_interval = 2
    distance_from_surface = 1
    simulation_end_time_meep = 1000

    sim_pd.reset_meep()
    sim_pd.run(mp.at_every(record_interval, get_slice(vals_pd, distance_from_surface)),
            until=simulation_end_time_meep)

    (x,y,z,w)=sim_pd.get_array_metadata(center=mp.Vector3(0,1), size=mp.Vector3(sx,0))
   
    if mp.am_master():
   
        # Check whether the specified path exists or not
        path = 'saved_matrices/' + outdir
        isExist = os.path.exists(path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(path)
            print("The new directory is created!")

        plt.figure()
        time_range = simulation_end_time_meep / Time_MEEP_To_Sec * 1e12
        val_np = np.array(vals_pd) / ElectricField_MEEP_TO_SI
        mm = np.max(val_np)

        plt.imshow(val_np.T, 
                cmap='bwr',
                aspect = time_range / sx,
                extent = [0,time_range,-sx/2,sx/2])
        # plt.clim(vmin=-mm, vmax=mm)
        plt.colorbar()
        plt.savefig(path + '/pdfield_'+ sys.argv[1] + '.png', dpi=300)

        out_str = path + '/field_ez_pd_intensity_' + sys.argv[1] + 't0_' + sys.argv[2] + '.mat'
        savemat(out_str, {'e_pd': vals_pd, 'zstep': x[2]-x[1], 'tstep' : record_interval / Time_MEEP_To_Sec * 1e12})


