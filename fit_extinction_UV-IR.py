#!/usr/bin/env python


from bisect import bisect_left
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
import numpy as np
from pathlib import Path
from scipy import interpolate
import sys
sys.path.append('/home/slava/science/codes/python/spectro/')
sys.path.append('/home/slava/science/codes/python/dust_extinction_model/')
sys.path.append('/home/slava/anaconda3')
sys.path.append('/home/slava/science/codes/python/spectro/sviewer/')
from astropy import constants as const
import os
import pickle
from scipy.interpolate import interp1d
from matplotlib import rcParams
from astropy.io import ascii, fits
rcParams['font.family'] = 'serif'
import emcee
from chainconsumer import ChainConsumer

from astropy.io import fits
from scipy import signal
import scipy.signal
import time, glob
from numpy.polynomial.polynomial import polyval
import csv
from scipy.signal import savgol_filter

from dust_extinction_model.spectrum_model import *
from dust_extinction_model.gordon_extinction import *



###################################################################################################
#SET path to folder containing spectra and dust templates
if 1:
    folder = '/home/slava/science/codes/python/dust_extinction_model/'
    path_to_sdss_spectrum = folder+'spectrum/sdss.fits'
    path_to_composite_spectra = '/home/slava/science/article/absorption_spectra/Composite_spectra/'
    path_to_ir_continuum = folder+'IR_cont.dat'

#read ned filters
if 1:
    #J0900
    if 0:
        z_abs = 1.051
        z_qso = 1.993
        q_name = 'J0900'
        pars_arch = [0.09, 0.47, 1.46, 4.66, 1.11]
        AvMW = 0.119
    #J0901
    if 1:
        z_abs = 1.02
        z_qso = 2.0934
        q_name = 'J0901'
        pars_arch = [-1.07, 0.26, 0.38, 4.46, 0.93]
        AvMW = 0.076
    #J1007
    if 0:
        z_abs = 0.8839
        z_qso = 1.047
        q_name = 'J1007'
        #-0.55 + c1_corr,  c2=0.626, c3=3.34, g=1.449, x0=4.65, c4=0
        pars_arch = [-0.55, 0.626, 3.34, 4.65, 1.449]
        AvMW = 0.063
        continuum_model_name = 'VG'
    #J1017
    if 0:
        z_abs = 1.118
        z_qso = 1.219
        q_name = 'J1017'
        pars_arch = [0.16, -0.02, 1.65, 4.55, 1.85]
        AvMW = 0.039


    def mag(AB,ref=48.6,offset=0):
        return 10 ** (-(AB + ref+offset) / 2.5) / 1e-23
    ned_fluxes = {}
    if q_name == 'J0900':
        ned_fluxes['u(sdss)'] = photometry(l=3562 / 1e4, lmin=3055 / 1e4, lmax=4030 / 1e4, f=mag(20.79, offset=-0.04),
                                           err=-mag(20.79 - 0.04 + 0.074) + mag(20.79 - 0.04), name='SDSSu')  # pm0.09
        ned_fluxes['g(sdss)'] = photometry(l=4686 / 1e4, lmin=3797 / 1e4, lmax=5553 / 1e4, f=mag(20.544),
                                           err=-mag(20.54 + 0.027) + mag(20.54), name='SDSSg')  # pm0.02
        ned_fluxes['r(sdss)'] = photometry(l=6166 / 1e4, lmin=5418 / 1e4, lmax=6994 / 1e4, f=mag(19.69),
                                           err=-mag(19.69 + 0.020) + mag(19.69), name='SDSSr')  # pm0.01
        ned_fluxes['i(sdss)'] = photometry(l=7481 / 1e4, lmin=6692 / 1e4, lmax=8400 / 1e4, f=mag(19.281),
                                           err=-mag(19.281 + 0.024) + mag(19.281), name='SDSSi')  # pm0.01
        ned_fluxes['z(sdss)'] = photometry(l=8931 / 1e4, lmin=7964 / 1e4, lmax=10873 / 1e4, f=mag(18.96),
                                           err=-mag(19.00 + 0.044) + mag(19.00), name='SDSSz')  # pm0.02
    elif q_name == 'J0901':
        ned_fluxes['u(sdss)'] = photometry(l=3562 / 1e4, lmin=3055 / 1e4, lmax=4030 / 1e4, f=mag(18.92 - 0.04),
                                           err=-mag(18.932 + 0.020) + mag(18.932), name='SDSSu')  # pm0.09
        ned_fluxes['g(sdss)'] = photometry(l=4686 / 1e4, lmin=3797 / 1e4, lmax=5553 / 1e4, f=mag(18.41),
                                           err=-mag(18.423 + 0.020) + mag(18.423), name='SDSSg')  # pm0.02
        ned_fluxes['r(sdss)'] = photometry(l=6166 / 1e4, lmin=5418 / 1e4, lmax=6994 / 1e4, f=mag(17.759),
                                           err=-mag(17.759 + 0.020) + mag(17.759), name='SDSSr')  # pm0.01
        ned_fluxes['i(sdss)'] = photometry(l=7481 / 1e4, lmin=6692 / 1e4, lmax=8400 / 1e4, f=mag(17.48),
                                           err=-mag(17.518 + 0.02) + mag(17.518), name='SDSSi')  # pm0.01
        ned_fluxes['z(sdss)'] = photometry(l=8931 / 1e4, lmin=7964 / 1e4, lmax=10873 / 1e4, f=mag(17.178),
                                           err=-mag(17.178 + 0.02) + mag(17.178), name='SDSSz')  # pm0.02
    elif q_name == 'J1007':
        ned_fluxes['NUV(GALEX)'] = photometry(nu=1.29e15, f=6.77e-7, err=3.5e-7)  # 3.4e-7
        ned_fluxes['u(sdss)'] = photometry(l=3562 / 1e4, lmin=3055 / 1e4, lmax=4030 / 1e4,
                                           f=mag(21.175, offset=-0.04), err=-mag(21.175 + 0.030) + mag(21.175),
                                           name='SDSSu')  # pm0.09
        ned_fluxes['g(sdss)'] = photometry(l=4686 / 1e4, lmin=3797 / 1e4, lmax=5553 / 1e4, f=mag(20.101),
                                           err=-mag(20.101 + 0.02) + mag(20.101), name='SDSSg')  # pm0.02
        ned_fluxes['r(sdss)'] = photometry(l=6166 / 1e4, lmin=5418 / 1e4, lmax=6994 / 1e4, f=mag(18.766),
                                           err=-mag(18.766 + 0.01) + mag(18.766), name='SDSSr')  # pm0.01
        ned_fluxes['i(sdss)'] = photometry(l=7481 / 1e4, lmin=6692 / 1e4, lmax=8400 / 1e4, f=mag(18.304),
                                           err=-mag(18.304 + 0.01) + mag(18.304), name='SDSSi')  # pm0.01
        ned_fluxes['z(sdss)'] = photometry(l=8931 / 1e4, lmin=7964 / 1e4, lmax=10873 / 1e4, f=mag(17.886),
                                           err=-mag(17.886 + 0.02) + mag(17.886), name='SDSSz')  # pm0.02
        ned_fluxes['J(2MASS)'] = photometry(nu=2.4e14, lmin=10806.47 / 1e4, lmax=14067.97 / 1e4, f=3.74e-4, err=4.8e-5,
                                            name='2MASSJ')
        ned_fluxes['H(2MASS)'] = photometry(nu=1.82e14, lmin=14787.38 / 1e4, lmax=18231.02 / 1e4, f=5.2e-4, err=7.3e-5,
                                            name='2MASSH')
        ned_fluxes['Ks(2MASS)'] = photometry(nu=1.38e14, lmin=19543.69 / 1e4, lmax=23552.40 / 1e4, f=7.93e-4, err=7e-5,
                                             name='2MASSK')


    # set normalization point for continuum
    w_norm = 0.7481
    w_norm_disp = 0.1
    f_units = 'F_lam'

    #convert flux units to Jy or Flam
    if f_units == 'F_lam':
        for el in ned_fluxes.values():
            fnorm = (el.l * 1e4) ** 2 / 3e18 * 1e-17 / 1e-23  # in 1e-17 erg/s/cm2/A
            el.f /= fnorm
            el.err /= fnorm

    # correct photometry fluxes for galactic extinction
    if AvMW > 0:
        for el in ned_fluxes.values():
            el.f *= np.exp(+AvMW * extinction_gordon23(l=np.array(el.l), Rv=3.1) / 1.086)

    # derive function for calculation the integrated flux within the band
    def calc_band_flux(f=ned_fluxes['i(sdss)'], redenned_model=interp1d([1,2],[1,2])):
        xx = np.linspace(f.lmin, f.lmax, 100)
        band_flux = np.trapz(redenned_model(xx / (1 + z_abs)), xx) / (f.lmax - f.lmin)
        return band_flux


# read spectral data
if 1:
    # read sdss spectrum
    if 1:
        # change rebin to True, if you would like to plot rebinned spectrum
        rebin =False
        qname = path_to_sdss_spectrum
        f_units = 'F_lam'
        if qname.endswith('fits'):
            hdulist = fits.open(qname)
            data = hdulist[1].data
            l = 10 ** data.field('loglam')  # in Angstrom
            fl = data.field('flux')  # in 1e-17 erg/cm2/s/Ang
            sig = (data.field('ivar')) ** (-0.5)


        elif qname.endswith('.dat'):
            data = np.loadtxt(qname)
            l = data[:,0]
            fl = data[:,1]
            dig =  data[:,2]

        fnorm = 1
        if f_units == 'Jy':
            fnorm = l ** 2 / 3e18 * 1e-17 / 1e-23  # in Jy
        col1 = l / 1e4 # convert to microns
        col2 = fl * fnorm
        col3 = sig * fnorm

        #rebin the data
        if rebin:
            n = 4
            x = rebin_arr(col1, n)
            y, err = rebin_weight_mean(col2, col3, n)
        else:
            x, y, err = col1, col2, col3

        # save spectral data as the "spectrum" object
        mask = y > 0
        sp_sdss = spectrum(x=x[mask], y=y[mask], err=err[mask], name=qname)
        sp_sdss.z_abs = z_abs
        sp_sdss.z_qso = z_qso
        sp_sdss.case = qname

        # correction for galactic extinction
        if AvMW > 0:
            sp_sdss.y *= np.exp(+AvMW * extinction_gordon23(l=np.array(sp_sdss.x), Rv=3.1) / 1.086)
            sp_sdss.err *= np.exp(+AvMW * extinction_gordon23(l=np.array(sp_sdss.x), Rv=3.1) / 1.086)


        # correct spectral data for the offset with sdss photometry in i filter
        sp_sdss.y *= ned_fluxes['i(sdss)'].f / np.mean(sp_sdss.y[np.abs(sp_sdss.x - 7481 / 1e4) < 300 / 1e4])
        sp_sdss.err *= ned_fluxes['i(sdss)'].f / np.mean(sp_sdss.y[np.abs(sp_sdss.x - 7481 / 1e4) < 300 / 1e4])


# create the quasar continuum
if 1:

    from dust_extinction_model.quasar_composite import qso_composite

    # function for qso composite construction
    def calc_composite_cont(s=qso_composite(units='micron',mode='JG',flux_units='Jy'), debug=False,  f_units=f_units):
        '''
        mode - set the continuum model:
        JG  = Jiang+Glikman
        VSG = Vanderberk+Selsing+Glikman
        VG =   Vanderberk+Glikman
        '''
        xx = np.array(s.x)
        fnorm = 1
        if f_units == 'F_lam':
            fnorm = (xx * 1e4) ** 2 / 3e18 * 1e-17 / 1e-23  # in 1e-17 erg/s/cm2/A
        return spectrum(xx, np.array(s.y) / fnorm)

    # create the composite
    s_composite = calc_composite_cont()


# define data for fitting (data_x,data_y)
if 1:
    #create a mask for sdss spectrum
    sp_sdss.mask_fit = sp_sdss.x > 0

    # exclude regions around emission lines
    lines = {}
    lines['Lya'] = em_line('Lya', 1215.67, 19.4)
    lines['SiIV'] = em_line('SiIV', 1398.3, 12.5)
    lines['CIV'] = em_line('CIV', 1546.15, 17)
    lines['CIII]'] = em_line('CIII]', 1905.9, 23)
    lines['MgII'] = em_line('MgII', 2800, 34)
    lines['[OII]'] = em_line('[OII]', 3729, 6)
    lines['[NeIII]'] = em_line('[NeIII]', 3869, 4)
    lines['Hg'] = em_line('Hg', 4346, 40)
    lines['FeII'] = em_line('FeII', 4564, 61.9)
    lines['Hb'] = em_line('Hb', 4853, 40.4)
    lines['[OIII]5008'] = em_line('[OIII]5008', 5008, 12)
    lines['[OIII]4964'] = em_line('[OIII]4964', 4964.3, 6)
    lines['Ha'] = em_line('Ha', 6564.9, 47.4)
    for l in lines.values():
        sp_sdss.mask_fit[np.abs(sp_sdss.x * 1e4 / (1 + sp_sdss.z_qso) - l.l) < l.w] = False
    sp_sdss.mask_fit[sp_sdss.x > 0.98] = False

    # excluded absorption lines
    if q_name == 'J0900':
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 8960 / 1e4) < 2 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 8920 / 1e4) < 2 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 8830 / 1e4) < 10 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 7380 / 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 7278 / 1e4) < 3 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 5888 / 1e4) < 2 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 5852 / 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 5582 / 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 5550 / 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 5740 / 1e4) < 10 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 5330 / 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 4965 / 1e4) < 4 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 4890 / 1e4) < 4 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 4870 / 1e4) < 4 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 4845 / 1e4) < 4 / 1e4] = False

        sp_sdss.mask_fit[np.abs(sp_sdss.x - 4800 / 1e4) < 10 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 4360 / 1e4) < 10 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 4155 / 1e4) < 10 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 4050 / 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 4020 / 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[sp_sdss.y < 1] = False
    elif q_name == 'J1017':
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 8886 / 1e4) < 4 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 6043 / 1e4) < 10 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 5930 / 1e4) < 15 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 5578 / 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 5508 / 1e4) < 4 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 5478 / 1e4) < 4 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 5048 / 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 5029 / 1e4) < 4 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 4965 / 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[sp_sdss.y < 1] = False
    elif q_name == 'J0901':
        # remove absorptions
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 8222 / 1e4) < 40 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 7600 / 1e4) < 60 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 7000 / 1e4) < 30 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 6897 / 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 6865 / 1e4) < 6 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 6475 / 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 5760 / 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 5653 / 1e4) < 30 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 5237 / 1e4) < 30 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 4906 / 1e4) < 30 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 4733 / 1e4) < 20 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 4595 / 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 4541 / 1e4) < 20 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 4491 / 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 4541 / 1e4) < 20 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 4471 / 1e4) < 20 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 4080 / 1e4) < 20 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 3917 / 1e4) < 20 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 3875 / 1e4) < 10 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 3834 / 1e4) < 10 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 3708 / 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 3689 / 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 3662 / 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 3634 / 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 3620 / 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 3605 / 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 3592 / 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[sp_sdss.x < 3596 / 1e4] = False
    elif q_name == 'J1007':
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 9948 / 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 7478 / 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 7413 / 1e4) < 10 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 5685 / 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 5671 / 1e4) < 6 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 5374 / 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 5275 / 1e4) < 15 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 4880 / 1e4) < 40 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 4489 / 1e4) < 7 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 4473 / 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 4415 / 1e4) < 7 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 4181 / 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 4088 / 1e4) < 8 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 4002 / 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 3787 / 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 3748 / 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 3716 / 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 3617/ 1e4) < 5 / 1e4] = False
        sp_sdss.mask_fit[np.abs(sp_sdss.x - 3594/ 1e4) < 5 / 1e4] = False

    # derive data: dat_x,data_y and data_err - lambda, flux, flux uncertainty
    data_x = np.array(sp_sdss.x[sp_sdss.mask_fit])
    data_y = np.array(sp_sdss.y[sp_sdss.mask_fit])
    data_err = np.array(sp_sdss.err[sp_sdss.mask_fit])

    # normalize to flux at wavelength = w_norm
    data_norm = np.nanmean(sp_sdss.y[np.abs(sp_sdss.x - w_norm) < w_norm_disp])
    data_y /= data_norm
    data_err /= data_norm

    # shift data_x to the absorption restframe
    data_x /= (1 + sp_sdss.z_abs)

    # check data for absorption/emission details
    if 1:
        plt.subplots()
        # plot unmasked sdss spectra
        plt.plot(sp_sdss.x, sp_sdss.y / data_norm, color='red', zorder=-10)
        plt.errorbar(x=sp_sdss.x, y=sp_sdss.y / data_norm, yerr=sp_sdss.err / data_norm, ecolor='red', fmt='none', zorder=-10)
        # plot masked sdss spectra
        plt.plot(data_x * (1 + sp_sdss.z_abs), data_y, 'o')
        # plot ned fluxes
        for el in ned_fluxes.values():
            if 'SDSS' in el.name:
                plt.plot(el.l, el.f / data_norm, 's', color='blue')
            elif '2MASS' in el.name:
                plt.plot(el.l, el.f / data_norm, 's', markerfacecolor='orange', markeredgecolor='black')
            elif 'WISE' in el.name:
                plt.plot(el.l, el.f / data_norm, 's', color='red')
            else:
                plt.plot(el.l, el.f / data_norm, 's', color='magenta')
        #plot composite
        if 1:
            mask_norm = np.abs(s_composite.x * (1 + sp_sdss.z_qso) - w_norm) < w_norm_disp
            y_cont_norm = np.nanmean(s_composite.y[mask_norm])
            plt.plot(s_composite.x * (1 + z_qso), s_composite.y / y_cont_norm, color='black')
        #plot normalization wavelength
        plt.axvline(w_norm, ls='--')
        plt.xlabel('Wavelength (Observed)')
        plt.ylabel('Flux')
        # set scale of y-axis
        #plt.yscale('log')
        plt.show()

# define the fitting model
if 1:
    (f0, c1, c2, c3, x0, gamma) = [1, 0.03, 0.01, 0.4, 4.58, 1.46]
    parameters_reduced = [f0, c1, c2, c3, x0, gamma]
    # a function for calculation the extinction curve
    def extinction_quasar_reduced(l=[1], c1=c1, c2=c2, c3=c3, x0=x0, gamma=gamma, debug=False):
        l = np.array(l)
        f = FM_ext(xx=l, c1=c1, c2=c2, c3=c3, x0=x0, g=gamma)
        if debug == True:
            plt.subplots()
            plt.plot(l, f)
            plt.xscale('log')
            plt.yscale('log')
            plt.show()
        return f

    # a function for comparison data with the redenned model
    def fit_qso_reduced(pars=parameters_reduced, debug=False, photometry=False, return_data = False,show_archival=True):
        (f0, c1, c2, c3, x0, gamma) = pars
        # create the composite spectrum in quasar restframe
        s_composite = calc_composite_cont()
        # create a mask for composite array covering spectral region near w_norm wavelength
        mask_norm = np.abs(s_composite.x * (1 + sp_sdss.z_qso) - w_norm) < w_norm_disp


        # normalize composite at flux at the w_norm wavelength
        s_composite.y /= np.nanmean(s_composite.y[np.abs(s_composite.x*(1+z_qso) - w_norm) < 0.05])

        # create the extinction_model
        ext_model = extinction_quasar_reduced(l=s_composite.x * (1 + sp_sdss.z_qso) / (1 + sp_sdss.z_abs), c1=c1,
                                                  c2=c2, c3=c3, x0=x0, gamma=gamma)

        # create a model for scaled redenned composite
        fit_model = f0*np.array(s_composite.y)*np.exp(-ext_model/ 1.086)
        # interpolation the model in absorption restframe
        fit_model_interp = interp1d(s_composite.x * (1 + sp_sdss.z_qso) / (1 + sp_sdss.z_abs), fit_model,
                                    fill_value='extrapolate')



        # define weight function for spectral data
        weight = np.ones_like(data_x)
        weight[np.abs(data_x-2175)<400] = 10

        # calculate chi2 fro spectral data
        chiq = np.sum(weight*np.power((data_y - fit_model_interp(data_x)) / (data_err), 2))

        # calculate additional chi2 for photometric data
        if photometry:
            chiq_phot = 0
            for f in ned_fluxes.values():
                weight = 0
                if '2MASS' in f.name:
                    phot_y = np.array(f.f / data_norm)
                    phot_err = np.array(f.err / data_norm)
                    model_flux = calc_band_flux(f=f, redenned_model=fit_model_interp)
                    weight = 200
                elif f.name in ['u(sdss)','g(sdss)','r(sdss)','i(sdss)','z(sdss)']:
                    phot_y = np.array(f.f / data_norm)
                    phot_err = np.array(f.err / data_norm)
                    model_flux = calc_band_flux(f=f, redenned_model=fit_model_interp)
                    weight = 100
                # calculate chi2 for bands with weight>0
                if weight > 0:
                    chiq_phot += weight * np.power((phot_y - model_flux) / phot_err, 2)
            # add  photometric chi2 to spectral data chi2
            chiq += chiq_phot

        # plot model and data
        if debug:
            print(pars)
            fit_composite = interp1d(s_composite.x * (1 + sp_sdss.z_qso) / (1 + sp_sdss.z_abs), f0*s_composite.y,
                                        fill_value='extrapolate')

            fig, ax = plt.subplots(2, 1, sharex=True)

            ax[0].plot(s_composite.x * (1 + z_qso) / (1 + z_abs), ext_model, label='ext_model',color='red')
            #ax[0].plot(s_composite.x * (1 + z_qso) / (1 + z_abs), np.exp(-ext_model), label='ext_model')
            ax[0].plot(data_x, -1.086 * np.log(data_y/ fit_composite(data_x)), label='Data',color='black', zorder=-100)

            ax[1].plot(s_composite.x * (1 + z_qso) / (1 + z_abs),
                       s_composite.y / np.nanmean(s_composite.y[mask_norm]), label='Cont')
            ax[1].plot(s_composite.x * (1 + z_qso) / (1 + z_abs), fit_model, label='Cont+Ext', color='red')

            ax[1].plot(data_x, data_y, label='Data', color='black', lw=4, zorder=-100)
            ax[1].errorbar(x=data_x, y=data_y, yerr=data_err, fmt='none', color='black', zorder=-100)
            if photometry:
                for f in ned_fluxes.values():

                    phot_y = np.array(f.f / data_norm)
                    phot_err = np.array(f.err / data_norm)
                    ax[1].errorbar(x=f.l / (1 + z_abs), y=phot_y, yerr=phot_err, fmt='s')
                    ax[1].errorbar(x=f.l / (1 + z_abs), y=f.f / data_norm, yerr=phot_err, fmt='v')


            ax[1].set_title('chi2=' + str(chiq))
            ax[1].legend()
            ax[1].set_xscale('log')
            # ax[1].set_yscale('log')
            ax[0].set_ylabel('Ext curve')
            ax[1].set_ylabel('Flux')
            ax[0].axvline(w_norm / (1 + z_abs), ls='--')
            ax[1].axvline(w_norm / (1 + z_abs), ls='--')
            if show_archival and 1:
                ext_model_archival = extinction_quasar_reduced(l=s_composite.x * (1 + sp_sdss.z_qso) / (1 + sp_sdss.z_abs),
                                                      c1=pars_arch[0], c2=pars_arch[1], c3=pars_arch[2], x0=pars_arch[3], gamma=pars_arch[4])
                fnorm =  np.mean((s_composite.y * np.exp(-ext_model_archival / 1.086))[mask_norm])

                if 1:

                    f_arch = interp1d(s_composite.x * (1 + z_qso) / (1 + z_abs), s_composite.y * np.exp(-ext_model_archival / 1.086)/fnorm)
                    f0_arc = np.linspace(0.5,3,100)
                    weight = np.ones_like(data_x)
                    weight[np.abs(data_x - 2175) < 400] = 10
                    chiq_x =  [np.sum(weight*np.power((data_y - f_arch(data_x)*el) / (data_err), 2)) for el in f0_arc]
                    f_arch_x = interp1d(chiq_x-np.nanmin(chiq_x),f0_arc)
                    #plt.subplots()
                    #plt.plot(x0,chiq_x)
                    print('arch theta:', f_arch_x(0),pars_arch)
                ax[0].plot(s_composite.x * (1 + z_qso) / (1 + z_abs), ext_model_archival, color='green')
                ax[1].plot(s_composite.x * (1 + z_qso) / (1 + z_abs), f_arch_x(0)*s_composite.y * np.exp(-ext_model_archival / 1.086)/fnorm,
                           color='green')
                if 1:
                    ax[1].plot(s_composite.x * (1 + z_qso) / (1 + z_abs),
                               f_arch_x(0) * s_composite.y * np.exp(-extinction_quasar_reduced(l=s_composite.x * (1 + sp_sdss.z_qso) / (1 + sp_sdss.z_abs),
                                                      c1=pars_arch[0], c2=pars_arch[1], c3=0, x0=pars_arch[3], gamma=pars_arch[4]) / 1.086) / fnorm,
                               color='green',ls= '--')
                    ax[1].plot(s_composite.x * (1 + z_qso) / (1 + z_abs), f0*np.array(s_composite.y)*
                               np.exp(-extinction_quasar_reduced(l=s_composite.x * (1 + sp_sdss.z_qso) / (1 + sp_sdss.z_abs), c1=c1,
                                                  c2=c2, c3=0, x0=x0, gamma=gamma)/1.086), label='Cont+Ext', color='red',ls='--')

            plt.show()


        ln = -chiq

        # return ln (by default) or model details
        if return_data:
            print('chi2=', chiq, ' chi2/red', chiq / np.size(data_y))
            return (s_composite.x, s_composite.y, ext_model, fit_model)
        else:
            return ln

    # test procedure
    fit_qso_reduced(debug=True)


# define likelihood and mcmc sampler
if 1:

    # define likelihood function for data
    def log_likelihood(theta, x, y):
        ln = fit_qso_reduced(pars=theta)
        return ln

    # define prior function for paramters
    def log_prior(theta):
        (f0,c1, c2, c3, x0, gamma) = theta
        if (0<f0 and -4 < c1 <10  and -1 < c2 <10 and 0 < c3 <10 and 4.4 < x0 < 4.8 and 0.5< gamma<2.7):
            return 0.0
        return -np.inf



    x, y = data_x, data_y

    # define combined likelihood
    def log_probability(theta, x, y):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, x, y)


    # defin fitting parmeters
    if 1:
        ndim = 6  #number of parameters
        nwalkers = 300
        nsteps = 300
        par_names = ['f0', 'c1', 'c2', 'c3', 'x0', 'gamma']

    # set initial values and run mcmc and save calcualtion to mcmc.pkl
    if 1:
        f0 = 1
        c1 = 0.5
        c2 = 0.2
        c3 = 0.3
        x0 = 4.6
        gamma = 1
        init = [f0, c1, c2, c3, x0, gamma]
        init_range = [1, 0.5, 0.2, 0.2, 0.5, 0.5]

        pos = []
        for i in range(nwalkers):
            prob = -np.inf
            while np.isinf(prob):
                rndm = np.random.randn(ndim)
                wal_pos = init + init_range * rndm
                prob = log_probability(theta=wal_pos, x=x, y=y)
            pos.append(wal_pos)

        from multiprocessing import Pool

        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool, args=(x, y))
            start = time.time()
            sampler.run_mcmc(pos, nsteps, progress=True)
            end = time.time()
            multi_time = end - start
            print("Multiprocessing took {0:.1f} seconds".format(multi_time))

        samples = sampler.chain[:, :, :]
        if 1:
            with open('mcmc.pkl', 'wb') as f:
                pickle.dump(samples, f)

    #read  mcmc.pkl and analyse the chain
    if 1:
        with open('mcmc.pkl', 'rb') as f:
            samples = pickle.load(f)

        # make a plot for chain statistic - mean, dispersion, and random walker path
        if 1:
            means = np.zeros((ndim, nsteps))
            vars = np.zeros((ndim, nsteps))
            single = np.zeros((ndim, nsteps))
            chi2 = np.zeros((nsteps))
            chi2mean = np.zeros((nsteps))
            chi2max = np.zeros((nsteps))
            for i in range(nsteps):
                for j in range(ndim):
                    means[j, i] = np.mean(samples[:, i, j])
                    vars[j, i] = np.std(samples[:, i, j])
                    single[j, i] = samples[5, i, j]
            for i in range(int(1),nsteps,100):
                print('i=',i)
                chi2ij = []
                for j in range(0,nwalkers):
                    theta = [samples[j, i, k] for k in range(ndim)]
                    chi2ij.append(-log_probability(theta, x, y))
                chi2ij = np.array(chi2ij)
                chi2[i] = np.nanmin(chi2ij)
                chi2mean[i] = np.mean(chi2ij[np.isfinite(chi2ij)])
                chi2max[i] = np.max(chi2ij[np.isfinite(chi2ij)])

                # chain stats
                print('chain stats')
                fig, ax = plt.subplots(nrows=1, ncols=ndim+1,figsize=(3*(ndim+1),3))
                if ndim > 1:
                    i = 0
                    for col in ax:
                        print(i)
                        if i < ndim:
                            col.errorbar(np.arange(nsteps), means[i, :], yerr=vars[i, :],
                                         fmt='-', color='black',
                                         markeredgecolor='black', markeredgewidth=2, capsize=2,
                                         ecolor='royalblue', alpha=0.7)
                            col.plot(np.arange(nsteps), single[i, :], color='red')
                            col.set_title(par_names[i])
                        else:
                            mask = chi2>0
                            col.plot(np.arange(nsteps)[mask], chi2[mask], color='red')
                            col.plot(np.arange(nsteps)[mask], chi2mean[mask], color='black')
                            col.plot(np.arange(nsteps)[mask], chi2max[mask], color='blue')

                            col.set_yscale('log')
                        i += 1

                else:
                    i = 0
                    ax[0].errorbar(np.arange(nsteps), means[i, :], yerr=vars[i, :],
                                   fmt='-', color='black',
                                   markeredgecolor='black', markeredgewidth=2, capsize=2,
                                   ecolor='royalblue', alpha=0.7)
                    ax[0].plot(np.arange(nsteps), single[i, :], color='red')
                plt.show()

        #cut the chain at 0.8 of total lenght
        burnin = int(nsteps * 0.8)
        chain = samples[:, burnin:, :].reshape((-1, ndim))

        # analyse the chain using chainconsumer

        #calculte chi2 ditribution and lowest chi2 solution - theta_best
        if 1:
            from tqdm import tqdm
            fig,ax = plt.subplots(1,ndim,sharey=True)
            chain_last = samples[:, int(0.95*nsteps):, :].reshape((-1, ndim))
            chi2 = np.zeros(chain_last.shape[0])
            for i in tqdm(range(chain_last.shape[0])):

                chi2[i] = -log_probability(chain_last[i,:], x, y)
            for i in range(ndim):
                ax[i].plot(chain_last[:,i],chi2[:],'.')
                ax[i].set_xlabel(par_names[i])

            #define theta best
            m = np.where(chi2 == np.nanmin(chi2))[0][0]
            theta_best =  chain_last[m,:]

            for i in range(ndim):
                ax[i].plot(chain_last[m, i], chi2[m], 'o',c='red')

            print('theta_best',theta_best)
            plt.show()


        # calculate and plot contours for parameters
        from chainconsumer import ChainConsumer
        c = ChainConsumer()
        if 1:
            c.add_chain(chain, parameters=par_names)
            c.plotter.plot(filename="example.png", figsize="column")
            res = c.analysis.get_summary(parameters=par_names)
            print('res', res)

        if 0:
            theta_for_plot = []
            for p in par_names:
                theta_for_plot.append(res[p][1])
        else:
            theta_for_plot = theta_best

        # plot the best fit solution
        fit_qso_reduced(pars=ttheta_for_plot, debug=True)
        plt.show()



    #calculate paramters distribution in Av-Abump parameter space
    if 1:
        chain_tmp = np.zeros((chain.shape[0],2))
        chain_tmp[:,0] = chain[:,1]+chain[:,2]/0.55
        chain_tmp[:,1] = chain[:,3]*np.pi/2/chain[:,5]
        c2 = ChainConsumer()
        c2.add_chain(chain_tmp, parameters=['Av','A2175'])
        c2.plotter.plot(filename="example.png", figsize="column")
        res = c2.analysis.get_summary(parameters=['Av','A2175'])
        print('res', res)
        plt.show()