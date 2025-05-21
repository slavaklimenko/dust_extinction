import numpy as np
from dust_extinction_model.spectrum_model import *


def mag_flux_conv(AB, ref=48.6, offset=0):
    return 10 ** (-(AB + ref + offset) / 2.5) / 1e-23
def mag_flux_err(m,merr):
    return -mag_flux_conv(m + merr) + mag_flux_conv(m)

class sys():
    def __init__(self, z_abs=-1,z_qso=-1, pars_arch = [], AvMW = -1,ref = None):
        self.z_abs = z_abs
        self.z_qso = z_qso
        self.pars_arch = pars_arch
        self.AvMW = AvMW
        self.ref = ref
        self.photo_data = {}
    def get_data(self):
        return (self.z_abs, self.z_qso, self.pars_arch, self.AvMW)
    def add_sdss_photometry(self,band='u',mag=19,mag_err=0.02):
        if band == 'u':
            self.photo_data['SDSSu'] = photometry(l=3562 / 1e4, lmin=3055 / 1e4, lmax=4030 / 1e4, f=mag_flux_conv(mag, offset=-0.04),
                                       err=mag_flux_err(mag-0.04,merr=mag_err), name='SDSSu')
        elif band == 'g':
            self.photo_data['SDSSg'] = photometry(l=4686 / 1e4, lmin=3797 / 1e4, lmax=5553 / 1e4, f=mag_flux_conv(mag),
                       err=mag_flux_err(mag,merr=mag_err), name='SDSSg')
        elif band == 'r':
            self.photo_data['r(sdss)'] = photometry(l=6166 / 1e4, lmin=5418 / 1e4, lmax=6994 / 1e4, f=mag_flux_conv(mag),
                                                      err=mag_flux_err(mag,merr=mag_err), name='SDSSr')
        elif band == 'i':
            self.photo_data['i(sdss)'] = photometry(l=7481 / 1e4, lmin=6692 / 1e4, lmax=8400 / 1e4, f=mag_flux_conv(mag),
                                               err=mag_flux_err(mag,merr=mag_err), name='SDSSi')
        elif band == 'z':
            self.photo_data['z(sdss)'] = photometry(l=8931 / 1e4, lmin=7964 / 1e4, lmax=10873 / 1e4, f=mag_flux_conv(mag),
                                               err=mag_flux_err(mag,merr=mag_err), name='SDSSz')



#add abs sys data
QSO_list = {}
QSO_list['J0745'] = sys(z_abs = 1.8612, z_qso = 2.199,  pars_arch = [1,-0.901, 0.537, 0.358, 4.655, 0.831], AvMW = 0.160,ref = 'Ma2017')
QSO_list['J0850'] = sys(z_abs = 1.3269, z_qso = 1.8939,  pars_arch = [1,-0.984, 0.474, 0.287, 4.497, 0.891], AvMW = 0.073,ref = 'Ma2017')
QSO_list['J1138'] = sys(z_abs = 1.1788, z_qso = 1.6337,  pars_arch = [1,-0.224, 0.298, 0.425, 4.583, 0.908], AvMW = 0.033,ref = 'Ma2017')
#
QSO_list['J0900'] = sys(z_abs = 1.051,  z_qso = 1.993,  pars_arch = [1,0.09, 0.47, 1.46, 4.66, 1.11], AvMW = 0.119, ref = 'Jiang2011')
QSO_list['J0901'] = sys(z_abs = 1.019,   z_qso = 2.093,  pars_arch = [1,-1.07, 0.26, 0.38, 4.46, 0.93], AvMW = 0.076, ref = 'Ma2017')
QSO_list['J1007'] = sys(z_abs = 0.884,  z_qso = 1.047,  pars_arch = [1,-0.55, 0.626, 3.34, 4.65, 1.449], AvMW = 0.063, ref = 'Zhou2010')
QSO_list['J1017'] = sys(z_abs = 1.118,  z_qso = 1.219,  pars_arch = [1,0.16, -0.02, 1.65, 4.55, 1.85],  AvMW = 0.039, ref = 'Jiang2011')

# add photometry
QSO_list['J0900'].add_sdss_photometry('u',20.79,0.074)
QSO_list['J0900'].add_sdss_photometry('g',20.54,0.027)
QSO_list['J0900'].add_sdss_photometry('r',19.69,0.020)
QSO_list['J0900'].add_sdss_photometry('i',19.281,0.024)
QSO_list['J0900'].add_sdss_photometry('z',18.86,0.044)
#
QSO_list['J0901'].add_sdss_photometry('u',18.92,0.020)
QSO_list['J0901'].add_sdss_photometry('g',18.41,0.020)
QSO_list['J0901'].add_sdss_photometry('r',17.76,0.020)
QSO_list['J0901'].add_sdss_photometry('i',17.48,0.02)
QSO_list['J0901'].add_sdss_photometry('z',17.178,0.02)
#
QSO_list['J1007'].add_sdss_photometry('u',21.175,0.030)
QSO_list['J1007'].add_sdss_photometry('g',20.101,0.020)
QSO_list['J1007'].add_sdss_photometry('r',18.766,0.010)
QSO_list['J1007'].add_sdss_photometry('i',18.304,0.01)
QSO_list['J1007'].add_sdss_photometry('z',17.886,0.02)
QSO_list['J1007'].photo_data['NUV(GALEX)'] = photometry(nu=1.29e15, f=6.77e-7, err=3.5e-7)
QSO_list['J1007'].photo_data['J(2MASS)'] = photometry(nu=2.4e14, lmin=10806.47 / 1e4, lmax=14067.97 / 1e4, f=3.74e-4, err=4.8e-5, name='2MASSJ')
QSO_list['J1007'].photo_data['H(2MASS)'] = photometry(nu=1.82e14, lmin=14787.38 / 1e4, lmax=18231.02 / 1e4, f=5.2e-4, err=7.3e-5, name='2MASSH')
QSO_list['J1007'].photo_data['Ks(2MASS)'] = photometry(nu=1.38e14, lmin=19543.69 / 1e4, lmax=23552.40 / 1e4, f=7.93e-4, err=7e-5, name='2MASSK')
#
QSO_list['J0745'].add_sdss_photometry('u',21.46,0.11)
QSO_list['J0745'].add_sdss_photometry('g',20.21,0.02)
QSO_list['J0745'].add_sdss_photometry('r',19.799,0.02)
QSO_list['J0745'].add_sdss_photometry('i',19.013,0.02)
QSO_list['J0745'].add_sdss_photometry('z',18.448,0.032)
#
QSO_list['J0850'].add_sdss_photometry('u',20.02,0.05)
QSO_list['J0850'].add_sdss_photometry('g',19.601,0.03)
QSO_list['J0850'].add_sdss_photometry('r',18.968,0.02)
QSO_list['J0850'].add_sdss_photometry('i',18.275,0.02)
QSO_list['J0850'].add_sdss_photometry('z',18.021,0.026)
#
QSO_list['J1138'].add_sdss_photometry('u',20.11,0.04)
QSO_list['J1138'].add_sdss_photometry('g',19.581,0.06)
QSO_list['J1138'].add_sdss_photometry('r',18.872,0.03)
QSO_list['J1138'].add_sdss_photometry('i',18.475,0.03)
QSO_list['J1138'].add_sdss_photometry('z',18.425,0.03)
#
#spectral data
folder = '/home/slava/science/codes/python/dust_extinction_model/'
QSO_list['J0745'].path_sdss_spec = folder+'spectrum/spec-J0745-3673-55178-0352.fits'
QSO_list['J0850'].path_sdss_spec = folder+'spectrum/spec-J0850-0447-51877-0105.fits'
QSO_list['J0901'].path_sdss_spec = folder+'spectrum/spec-2282-53683-9.dat'
QSO_list['J1138'].path_sdss_spec = folder+'spectrum/spec-J1138-0880-52367-0435.fits'


# set masked regions for sdss spectra
QSO_list['J0745'].mask_bad_pixels_sdss_spec = [3818, 4366,4430,4532.4601,4740,4780,5577,5893,6704,6793,6817,
                                               7401,7439,7999,8019,8160,8186,9314,9378,9441,9480,9504,9522,9791]
QSO_list['J0850'].mask_bad_pixels_sdss_spec = [3888,3915,4206,4314,4714,4797,5259,5995,6017,6048,
                                               6303,6506,6522,6637,8828,8910,8960]
QSO_list['J1138'].mask_bad_pixels_sdss_spec = [3847,3864,5171,5191, 5578,5666,5998,6091,6107,6303,7855,8764]





if __name__ == '__main__':
    Q = QSO_list['J0745']
    print()