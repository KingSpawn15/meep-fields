#!/usr/bin/env python
# coding: utf-8

# In[1]:


import meep as mp
import numpy as np
from matplotlib import pyplot as plt
from itertools import chain
from scipy.interpolate import RegularGridInterpolator
from eels_utils.ElectronSpectrum import ElectronSpectrum as es

        

# In[2]:


# Tstep_factor = (1e-6 / 3e8)
# t0_sec = 0.2e-12
# t0 = t0_sec / Tstep_factor
# t0
# 0.2 * 3 * 10 ** (8 + 6 -12)
# 0.5 * 0.1e-6 / (3 * 10**(8))


# In[3]:


resolution = 5
# InAs model


alpha = 7
# depth = 7


csecmusmec = 3 * 10**14
Time_Sec_To_MEEP = (1e-6 / 3e8)
Time_MEEP_To_Sec = 1 / Time_Sec_To_MEEP

# Tstep_ps = (1e-6 / 3e8 * 1e12)
# Tstep_factor = (1e-6 / 3e8) * S

def current_pd(depth, weight):
    n_eq = 1e23
    n_exc = 3.14e23 * np.exp(-abs(depth) * alpha) 
    gamma_sec = 3.3e12 

    omega_eq = 20.7539 * np.sqrt(n_eq)
    omega_exc = 44.5865 * np.sqrt(n_exc)
    omega_z_sec = np.sqrt(omega_exc**2 + omega_eq**2 - gamma_sec**2 / 4)
    
    
    omega_z = omega_z_sec / csecmusmec
    gamma = gamma_sec / csecmusmec

    t0_sec = 0.2e-12
    t0 = t0_sec / Time_Sec_To_MEEP

    return lambda t: weight * n_exc * np.sin(  omega_z *  (t-t0)) * np.exp(-gamma * (t-t0)/2) / omega_z if t>t0 else 0

def current_rectification(sigma, weight, t0):
    return lambda t: - weight * (t-t0) * np.exp(- ((t-t0) ** 2) / (2 * sigma **2))
    


# In[4]:


distance_from_surface = 1
# depth = alpha
t0_sec = 0.2e-12
sigma_t_sec = 50e-15 / (2 * np.sqrt(2*np.log(2)))

sigma_t = sigma_t_sec / Time_Sec_To_MEEP
t0 = t0_sec  / Time_Sec_To_MEEP

# Phi = np.pi * (60) / 180

sx = 200
sy = 50
dpml = 10

cell = mp.Vector3(sx + 2 * dpml, sy + 2 * dpml)
pml_layers = [mp.PML(dpml)]

penetration_d = 1/alpha

def polarized_source(phi):
    return  [
        #         mp.Source(
        # src=mp.CustomSource(src_func=current_rectification(sigma_t, -np.sqrt(1/3),t0)),
        # center=mp.Vector3(0,-0.5 * (1/alpha)),
        # component=mp.Ey),
        #     mp.Source(
        # src=mp.CustomSource(src_func=curent_rectification(sigma_t,  - 2 * np.sin(2 * phi) * np.sqrt(1/6),t0)),
        # center=mp.Vector3(0,-0.5 * alpha),
        # component=mp.Ez),
            mp.Source(
        src=mp.CustomSource(src_func=current_rectification(sigma_t,   2 * np.cos(2 * phi) * np.sqrt(1/6),t0)),
        center=mp.Vector3(0,-0.5 * (1/alpha)),
        component=mp.Ex)
            ]

def pd_source(weight):
    return  [
        #         mp.Source(
        # src=mp.CustomSource(src_func=curent_rectification(sigma_t, -np.sqrt(1/3),t0)),
        # center=mp.Vector3(0,-0.5 * alpha),
        # component=mp.Ey),
        #     mp.Source(
        # src=mp.CustomSource(src_func=curent_rectification(sigma_t,  - 2 * np.sin(2 * phi) * np.sqrt(1/6),t0)),
        # center=mp.Vector3(0,-0.5 * alpha),
        # component=mp.Ez),
            mp.Source(
        src=mp.CustomSource(src_func= current_pd(depth = 0.5 * (1/alpha), weight = weight) ),
        center=mp.Vector3(0,-0.5 * (1/alpha)),
        component=mp.Ey)
            ]

fp_sec = omega_eq = 20.7539 * np.sqrt(1e23) / (2  * np.pi) * np.sqrt(0.05)
fp = fp_sec/csecmusmec
gamma_c = 3.3e12 / (2  * np.pi) / csecmusmec 

omega_TO_cm = 218
omega_LO_cm = 240
gamma_PH_cm = 3.5

freq_TO = omega_TO_cm * 3e10 / csecmusmec
freq_LO = omega_LO_cm * 3e10 / csecmusmec
gamma_ph = gamma_PH_cm * 3e10 / csecmusmec * 5

drude = mp.DrudeSusceptibility(frequency=fp, sigma=12, gamma=gamma_c)
lor = mp.LorentzianSusceptibility(frequency= freq_TO, gamma=gamma_ph, sigma=11)

geometry = [
    mp.Block(
        mp.Vector3(mp.inf, sy/2 + dpml, mp.inf),
        center=mp.Vector3(0,-sy/4 - dpml/2),
        material=mp.Medium(epsilon=12, E_susceptibilities=[lor]),
    )
]

def create_simulation_meep(phi):

    return mp.Simulation(
        cell_size=cell,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=polarized_source(phi),
        resolution=resolution,
        symmetries=None,
        progress_interval = 15
    )

def create_simulation_pd(weight):

    return mp.Simulation(
        cell_size=cell,
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=pd_source(weight),
        resolution=resolution,
        symmetries=None,
        progress_interval = 15
    )







# %matplotlib inline
# plt.figure(dpi=100)
# sim.plot2D()
# plt.show()


# In[5]:


# \mu m
Nsteps = 500

def field_calculate_pd(weight, record_interval):
       sim = create_simulation_pd(weight)
       
       vals = []

       def get_slice(sim):
               vals.append(sim.get_array(center=mp.Vector3(0,distance_from_surface),
                             size=mp.Vector3(sx,0), 
                             component=mp.Ex))

           # %matplotlib inline
       # plt.figure(dpi=100)
       # sim.plot2D()
       # plt.show()
   
       sim.reset_meep()
       sim.run(mp.at_every(record_interval, get_slice),
               until=Nsteps)
       return sim, vals

def field_calculate(phi, record_interval):
       sim = create_simulation_meep(phi)

       # plt.figure(dpi=100)
       # sim.plot2D()
       # plt.show()
   
       vals = []

       def get_slice(sim):
               vals.append(sim.get_array(center=mp.Vector3(0,distance_from_surface),
                             size=mp.Vector3(sx,0), 
                             component=mp.Ex))

       sim.reset_meep()
       sim.run(mp.at_every(record_interval, get_slice),
               until=Nsteps)
       return sim, vals


# In[6]:






field_list = []

# for angle in [0]:
#     sim, vals = field_calculate(np.pi * angle / 180)
#     field_list.append(vals)

record_interval = 2
sim, vals_or_0 = field_calculate(np.pi * 0 / 180, record_interval)
sim, vals_or_90 = field_calculate(np.pi * 90 / 180, record_interval)
sim_pd, vals_pd = field_calculate_pd(1e-6, record_interval)
