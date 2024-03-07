from scipy.interpolate import RegularGridInterpolator
from scipy.signal import convolve2d
import numpy as np

class ElectronSpectrum:
        
        
    @staticmethod 
    def psi_coherent_to_incoherent(psi_coherent, max_e, max_t ):
        
        a = 21.2
        b = 2.2
        c = 3.4

        energy_vec = np.linspace(-5,5,401)
        delta_t = np.linspace(-3,3,201)
                    
        TW, EW = np.meshgrid(delta_t,energy_vec)
        W = np.exp(-( a * np.power(TW,2) + b * EW * TW + c * np.power(EW,2))).T
        
        psi_incoherent = convolve2d(psi_coherent, W, mode='same', boundary='fill', fillvalue=0)

        return psi_incoherent

    @staticmethod
    def field_to_eels(field, t_vec, z_vec, delta_t):

        ve_elec = 0.7 * 3 * 10**(8+6-12) # [\mu m / ps]

        interp = RegularGridInterpolator((t_vec, z_vec), field,
                                bounds_error=False,
                                fill_value=0)
    
        def field_along_electron(deltat):
            ddz = np.arange(-200,200,.5)
            return np.trapz(interp((ddz / ve_elec + deltat,ddz), method='slinear'),dx = 0.5)


        eels = np.array(list(map(field_along_electron,delta_t)))

        return eels
    
    @staticmethod
    def convolve_laser_profile(field, xc, spot_size_fwhm):
        
        xc = np.array(xc)
        sigma = spot_size_fwhm / (np.sqrt(8 * np.log(2)))
        intensity = np.exp(-(xc)**2/(2 * sigma**2))


        def conv_line(field):
            padded_intensity = np.pad(intensity, (len(xc),len(xc)))
            padded_field = np.pad(field, (len(xc),len(xc)))
            conv_fi =  np.convolve(padded_intensity,padded_field,
                                mode='same')
            return conv_fi[len(xc):2 * len(xc) ]
        
        field_laser_profile = np.apply_along_axis(conv_line, 1, field)
        return field_laser_profile
    
    @staticmethod
    def eels_to_psi_coherent(eels, delta_t):
        
        DeltaEmax = 5
        LenEvec = 401

        energy_vec = np.linspace(-DeltaEmax,DeltaEmax,LenEvec)
        psi_coherent = np.zeros([len(delta_t),len(energy_vec)])
        
        AA = (len(energy_vec) - 1) / (2 * DeltaEmax)
        BB = (len(energy_vec) - 1) / 2
        ind_calc = lambda ee : int(AA * ee + BB)

        for ind in range(len(delta_t)):
            ind_to_fill = ind_calc(eels[ind])
            psi_coherent[len(delta_t) - ind - 1][ind_to_fill] = 1

        return psi_coherent


