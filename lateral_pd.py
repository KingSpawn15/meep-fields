import sys
import os
import meep as mp
import numpy as np
from matplotlib import pyplot as plt
from itertools import chain
from scipy.interpolate import RegularGridInterpolator , interp1d
from scipy.special import erf
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



# def current_photodember_with_pulse_time_lateral(params, xprime, yprime):
#     # Precompute constants
#     # alpha = params['alpha']
#     sigma_t_fs = params['fwhm_t_fs'] * 1e-15 / Time_Sec_To_MEEP
#     sigma_spot_sq = params['sigma_spot']**2
#     gamma_over_freq = params['gamma'] / Freq_Hz_To_MEEP 

#     v_t_m_sec_1_over_c = params['v_t_m_sec_1'] / 3e8

#     t0 = params['t0_sec'] / Time_Sec_To_MEEP
#     fwhm_t_fs_sqrt_log = params['fwhm_t_fs'] * 1e-15 / np.sqrt(8 * np.log(2))
#     sigma_t_meep = fwhm_t_fs_sqrt_log / Time_Sec_To_MEEP

#     # # Precompute j_pd and pulse_envelope
#     # tt = np.linspace(-500,2000,1500)
#     # # j_pd_l = xprime * np.exp(- 2 * gamma_over_freq * (tt-t0)) * np.exp(-xprime ** 2 / (2 * (v_t_m_sec_1_over_c ** 2 / gamma_over_freq * (tt-t0) + sigma_spot_sq)  )) / (v_t_m_sec_1_over_c ** 2 / gamma_over_freq * (tt-t0) + sigma_spot_sq) ** (3/2)
    
#     # j_pd_l = np.exp(- 2 * gamma_over_freq * (tt-t0)) * np.exp(- xprime ** 2 / (2 * (sigma_spot_sq)  ))


#     # j_pd_l[tt <= t0] = 0
#     # pulse_envelope = (1 / sigma_t_meep) * np.exp(-tt**2 / (2 * sigma_t_meep**2))

#     # # Convolve j_pd and pulse_envelope
#     # padding = 2000  # Increase this value to add more zeros
#     # pulse_envelope_padded = np.pad(pulse_envelope, (0, padding), 'constant')
#     # j_pd_padded = np.pad(j_pd_l, (0, padding), 'constant')
#     # convolved_padded = np.convolve(j_pd_padded, pulse_envelope_padded, mode='full')
#     # convolved = convolved_padded[:len(tt)]
    
#     # tt_max = 2000
#     # tt_min = -500

#     return lambda t: -  np.exp(- ((t-t0) ** 2) / (2 * sigma_t_meep **2)) * np.exp(- xprime ** 2 / (2 * (sigma_spot_sq)  ))
#     # return lambda t: np.interp(t + 500, tt, convolved) if tt_min <= t + 500 <= tt_max else 0

# def current_diffusion_normal(alpha, dd, t, t0, x):
#     # Compute the exponential term
#     exp_term = np.exp(alpha * (-2 * x + 2 * dd * (t-t0) * alpha) / 2)
    
#     # Compute the Gaussian term
#     gaussian_term = np.exp(-((x - 2 * dd * (t-t0) * alpha) ** 2) / (4 * dd * (t-t0)))
#     sqrt_term = np.sqrt( np.pi) * np.sqrt(dd * (t-t0))
#     first_part = gaussian_term / ( sqrt_term)
    
#     # Compute the error function (t-t0)erm
#     erf_term = erf((-x + 2 * dd * (t-t0) * alpha) / (2 * np.sqrt(dd * (t-t0))))
#     second_part = -alpha * (-1 + erf_term)
    
#     # Combine terms
#     result = 0.5 * exp_term * (first_part - second_part)
#     # result = erf_term 
#     return result

def current_diffusion_normal(alpha, dd, t, t0, x):
    # Initialize result array with zeros
    result = np.zeros_like(t)
    
    # Only perform computations where t > t0
    mask = t > t0
    
    if np.any(mask):
        # Compute the exponential term
        exp_term = np.exp(alpha * (-2 * x + 2 * dd * (t[mask]-t0) * alpha) / 2)
        
        # Compute the Gaussian term
        gaussian_term = np.exp(-((x - 2 * dd * (t[mask]-t0) * alpha) ** 2) / (4 * dd * (t[mask]-t0)))
        sqrt_term = np.sqrt(np.pi) * np.sqrt(dd * (t[mask]-t0))
        first_part = gaussian_term / sqrt_term
        
        # Compute the error function term
        erf_term = erf((-x + 2 * dd * (t[mask]-t0) * alpha) / (2 * np.sqrt(dd * (t[mask]-t0))))
        second_part = -alpha * (-1 + erf_term)
        
        # Combine terms
        result[mask] = 0.5 * exp_term * (first_part - second_part)
    
    return result

def current_photodember_with_pulse_time_normal(params, xprime, yprime):
        
        gamma_over_freq = params['gamma'] / Freq_Hz_To_MEEP
        v_t_m_sec_1_over_c = params['v_t_m_sec_1'] / 3e8
        dd = v_t_m_sec_1_over_c ** 2 / gamma_over_freq
        alpha = params['alpha']

        t0 = params['t0_sec'] / Time_Sec_To_MEEP
        fwhm_t_fs_sqrt_log = params['fwhm_t_fs'] * 1e-15 / np.sqrt(8 * np.log(2))

        tt = np.linspace(-500,2000,1500)
        j_pd_n = current_diffusion_normal(alpha, dd, tt, t0, yprime) * np.exp(-xprime ** 2 / (2 * params['sigma_spot']**2))
        # j_pd_n = current_diffusion_normal(alpha, dd, tt, t0, yprime)

        j_pd_n[tt <= t0] = 0
        sigma_t_meep = fwhm_t_fs_sqrt_log / Time_Sec_To_MEEP
        pulse_envelope = (1 / sigma_t_meep) * np.exp(-tt**2 / (2 * sigma_t_meep**2))

        # Convolve j_pd and pulse_envelope
        padding = 2000  # Increase this value to add more zeros
        pulse_envelope_padded = np.pad(pulse_envelope, (0, padding), 'constant')
        j_pd_padded = np.pad(j_pd_n, (0, padding), 'constant')
        convolved_padded = np.convolve(j_pd_padded, pulse_envelope_padded, mode='full')
        convolved = convolved_padded[:len(tt)]
        
        tt_max = 2000
        tt_min = -500
        return lambda t: np.interp(t + 500, tt, convolved) if tt_min <= t + 500 <= tt_max else 0

def current_photodember_with_pulse_time_lateral(params, xprime, yprime):
    # Precompute constants
    # alpha = params['alpha']
    sigma_spot_sq = params['sigma_spot']**2
    gamma_over_freq = params['gamma'] / Freq_Hz_To_MEEP

    v_t_m_sec_1_over_c = params['v_t_m_sec_1'] / 3e8

    t0 = params['t0_sec'] / Time_Sec_To_MEEP
    fwhm_t_fs_sqrt_log = params['fwhm_t_fs'] * 1e-15 / np.sqrt(8 * np.log(2))

    # Precompute j_pd and pulse_envelope
    tt = np.linspace(-500,2000,1500)
    j_pd_l = xprime * np.exp(-  gamma_over_freq * (tt-t0) / 2) * np.exp(-xprime ** 2 / (2 * (v_t_m_sec_1_over_c ** 2 / gamma_over_freq * (tt-t0) + sigma_spot_sq)  )) / (v_t_m_sec_1_over_c ** 2 / gamma_over_freq * (tt-t0) + sigma_spot_sq) ** (3/2)
    
    # j_pd_l = np.exp(- 2 * gamma_over_freq * (tt-t0)) * np.exp(- xprime ** 2 / (2 * (sigma_spot_sq)  ))


    j_pd_l[tt <= t0] = 0
    sigma_t_meep = fwhm_t_fs_sqrt_log / Time_Sec_To_MEEP
    pulse_envelope = (1 / sigma_t_meep) * np.exp(-tt**2 / (2 * sigma_t_meep**2))

    # Convolve j_pd and pulse_envelope
    padding = 2000  # Increase this value to add more zeros
    pulse_envelope_padded = np.pad(pulse_envelope, (0, padding), 'constant')
    j_pd_padded = np.pad(j_pd_l, (0, padding), 'constant')
    convolved_padded = np.convolve(j_pd_padded, pulse_envelope_padded, mode='full')
    convolved = convolved_padded[:len(tt)]
    
    tt_max = 2000
    tt_min = -500
    return lambda t: np.interp(t + 500, tt, convolved) if tt_min <= t + 500 <= tt_max else 0


def photodember_source_lateral(params, xmax, ymax):
    sl = [[mp.Source(
        src=mp.CustomSource(src_func=current_photodember_with_pulse_time_lateral(params, xi, yi),is_integrated=True),
        center=mp.Vector3(xi,-yi),
        component=mp.Ex) for xi in np.arange(-np.fix(xmax),np.fix(xmax) + 0.5, 0.5)] for yi in np.linspace(3,3,1)]
    return list(chain(*sl))

def photodember_source_normal(params, xmax, ymax):
    sl = [[mp.Source(
        src=mp.CustomSource(src_func=current_photodember_with_pulse_time_normal(params, xi, yi),is_integrated=True),
        center=mp.Vector3(xi,-yi),
        component=mp.Ey) for xi in np.arange(-np.fix(xmax),np.fix(xmax) + 0.5, 0.5)] for yi in np.linspace(0.1,0.1,1)]
    return list(chain(*sl))


if __name__ == '__main__':

    
    outdir = 'photodember/lateral'
    intensity = float(sys.argv[1])
    t0_sec = float(sys.argv[2]) * 1e-12
    fwhm_t_fs = float(sys.argv[3])
    # intensity = 10
    # t0_sec = 0.5 * 1e-12
    # fwhm_t_fs = 50
    

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
    non_dispersive_inas = mp.Medium(epsilon=12)

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
    ymax = 4 * (1 / params['alpha'])
    source_pd = photodember_source_lateral(params, xmax, ymax)
    # source_pd = photodember_source_normal(params, xmax, ymax)

    geometry = [
    mp.Block(
        mp.Vector3(mp.inf, sy/2 + dpml, mp.inf),
        center=mp.Vector3(0,-sy/4 - dpml/2),
        material=non_dispersive_inas,
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
        mm = np.max(np.abs(val_np))

        plt.imshow(val_np.T, 
                cmap='bwr',
                aspect = time_range / sx,
                extent = [0,time_range,-sx/2,sx/2])
        plt.clim(vmin=-mm, vmax=mm)
        plt.colorbar()
        plt.savefig(path + '/pdfield_lateral'+ sys.argv[1] + 'fwhm_t_' + sys.argv[3]+'.png', dpi=300)

        # out_str = path + '/field_ez_pd_intensity_' + sys.argv[1] + 't0_' + sys.argv[2] + 'fwhm_t_' + sys.argv[3] + '.mat'
        # savemat(out_str, {'e_pd': vals_pd, 'zstep': x[2]-x[1], 'tstep' : record_interval / Time_MEEP_To_Sec * 1e12})

        t_values = np.linspace(0,1000,1000)
        fs_values = []
        # Loop over t values
        for t in t_values:
            fs_t = current_photodember_with_pulse_time_lateral(params, 5, 0.1)
            # fs_t = current_photodember_with_pulse_time_normal(params, 5, 0.1)
            fs_values.append(fs_t(t))

        # print("")
        plt.figure(figsize=(10, 10))
        plt.rcParams.update({'font.size': 14})
        plt.plot(t_values / Time_MEEP_To_Sec * 1e12, fs_values, label='fs(t)')
        plt.xlabel('t [ps]',fontsize=16)
        plt.ylabel('current',fontsize=16)
        plt.title('Plot of lateral current for t',fontsize=16)
        plt.legend()
        # plt.ylim(0,0.0015)
        
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.grid(True)
        # plt.show()
        plt.savefig(path + '/current_lateral'+ sys.argv[1] + 'fwhm_t_' + sys.argv[3]+'.png', dpi=300)

        # plt.figure(figsize=(10, 6))
        # plt.plot(val_np.T[201][:])
        # plt.plot(val_np.T[501][:])
        # plt.plot(val_np.T[801][:])
        # plt.xlabel('t')
        # plt.ylabel('Ez')
        # plt.title('Plot of lateral current for t')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

        print("")