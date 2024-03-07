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
        "alpha" : 1.0/7,
        "epsilon_inf" : 12, 
        "epsilon_0" : 15,
        "omega_TO_cm_1" : 218, 
        "omega_LO_cm_1" : 240, 
        "gamma_ph_cm_1" : 3.5}
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


# def polarized_source(phi, sigma_t_sec, t0_sec, source_x, source_y):
#     return  [
#             mp.Source(
#         src=mp.CustomSource(src_func=current_rectification(sigma_t_sec, t0_sec, np.sqrt(1/3))),
#         center=mp.Vector3(source_x,source_y),
#         component=mp.Ey),
#             mp.Source(
#         src=mp.CustomSource(src_func=current_rectification(sigma_t_sec, t0_sec, - 2 * np.sin(2 * phi) * np.sqrt(1/6))),
#         center=mp.Vector3(source_x,source_y),
#         component=mp.Ez),
#             mp.Source(
#         src=mp.CustomSource(src_func=current_rectification(sigma_t_sec, t0_sec, 2 * np.cos(2 * phi) * np.sqrt(1/6))),
#         center=mp.Vector3(source_x,source_y),
#         component=mp.Ex)
#             ]

def polarized_source_z(sigma_t_sec, t0_sec, source_x, source_y):
    return  [
            mp.Source(
        src=mp.CustomSource(src_func=current_rectification(sigma_t_sec, t0_sec, 1)),
        center=mp.Vector3(source_x,source_y),
        component=mp.Ex)
            ]
def polarized_source_y(sigma_t_sec, t0_sec, source_x, source_y):
    return  [
            mp.Source(
        src=mp.CustomSource(src_func=current_rectification(sigma_t_sec, t0_sec, 1)),
        center=mp.Vector3(source_x,source_y),
        component=mp.Ez)
            ]
def polarized_source_x(sigma_t_sec, t0_sec, source_x, source_y):
    return  [
            mp.Source(
        src=mp.CustomSource(src_func=current_rectification(sigma_t_sec, t0_sec, 1)),
        center=mp.Vector3(source_x,source_y),
        component=mp.Ey)
            ]




if __name__ == '__main__':

    pulse_time_fwhm_fs = float(sys.argv[1])
    outdir = sys.argv[2]

    Freq_Hz_To_MEEP = 3 * 10**14
    Time_Sec_To_MEEP = (1e-6 / 3e8)
    Time_MEEP_To_Sec = 1 / Time_Sec_To_MEEP

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

    pulse_time_fwhm_fs = float(sys.argv[1])
    sigma_t_sec = pulse_time_fwhm_fs * 1e-15 / np.sqrt(8 * np.log(2))
    t0_sec = 0.2e-12
    source_x = 0
    source_y = - (1 / inas["alpha"])
    
    
    source_rectification_x = polarized_source_x(sigma_t_sec, t0_sec, source_x, source_y)
    source_rectification_y = polarized_source_y(sigma_t_sec, t0_sec, source_x, source_y)
    source_rectification_z = polarized_source_z(sigma_t_sec, t0_sec, source_x, source_y)
    
    geometry = [
    mp.Block(
        mp.Vector3(mp.inf, sy/2 + dpml, mp.inf),
        center=mp.Vector3(0,-sy/4 - dpml/2),
        material=inas_meep,
    )]


    sim_or_x = mp.Simulation(
        cell_size=mp.Vector3(sx + 2 * dpml, sy + 2 * dpml),
        boundary_layers=[mp.PML(dpml)],
        geometry=geometry,
        sources=source_rectification_x,
        resolution=resolution,
        symmetries=None,
        progress_interval = 15
    )

    sim_or_y = mp.Simulation(
        cell_size=mp.Vector3(sx + 2 * dpml, sy + 2 * dpml),
        boundary_layers=[mp.PML(dpml)],
        geometry=geometry,
        sources=source_rectification_y,
        resolution=resolution,
        symmetries=None,
        progress_interval = 15
    )

    sim_or_z = mp.Simulation(
        cell_size=mp.Vector3(sx + 2 * dpml, sy + 2 * dpml),
        boundary_layers=[mp.PML(dpml)],
        geometry=geometry,
        sources=source_rectification_z,
        resolution=resolution,
        symmetries=None,
        progress_interval = 15
    )
    vals_x = []
    vals_y = []
    vals_z = []

    def get_slice(vals, distance_from_surface):
        return lambda sim : vals.append(sim.get_array(center=mp.Vector3(0,distance_from_surface), size=mp.Vector3(sx,0), component=mp.Ez))

    record_interval = 2
    distance_from_surface = 1
    simulation_end_time_meep = 500

    sim_or_x.reset_meep()
    sim_or_x.run(mp.at_every(record_interval, get_slice(vals_x, distance_from_surface)),
            until=simulation_end_time_meep)
    
    sim_or_y.reset_meep()
    sim_or_y.run(mp.at_every(record_interval, get_slice(vals_y, distance_from_surface)),
            until=simulation_end_time_meep)
    
    sim_or_z.reset_meep()
    sim_or_z.run(mp.at_every(record_interval, get_slice(vals_z, distance_from_surface)),
            until=simulation_end_time_meep)
    
    
    (x,y,z,w)=sim_or_x.get_array_metadata(center=mp.Vector3(0,1), size=mp.Vector3(sx,0))
    
    if mp.am_master():

        # Check whether the specified path exists or not
        path = 'saved_matrices/' + outdir
        isExist = os.path.exists(path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(path)
            print("The new directory is created!")

        out_str = path + '/field_ez' + str(pulse_time_fwhm_fs) + '_fs' + '.mat'
        savemat(out_str, {'e_or_x': vals_x,
                          'e_or_y': vals_y,
                           'e_or_z': vals_z,
                            'zstep': x[2]-x[1], 'tstep' : record_interval / Time_MEEP_To_Sec * 1e12})