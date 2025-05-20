#!/usr/bin/env python


from bisect import bisect_left
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
import numpy as np
from pathlib import Path
from scipy import interpolate
import sys

from ydata_profiling.report.structure.variables import render_real

#from MaNGA.sdss_image import f_name

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
from scipy.signal import savgol_filter
import csv
from scipy.signal import savgol_filter
from spectrum_model import * #spectrum, rebin_arr,rebin_weight_mean

path_to_qso_normalized_spectra = '/home/slava/science/research/kulkarni/JWST-DLAs/ID2155/results/normalized_spectra/'
path_to_dust_templates_labs = '/home/slava/science/research/kulkarni/JWST-DLAs/ID2155/results/figures/fig-fit-profiles/dust_templates/Templates/profiles/Lab-Templates/'
path_to_dust_templates_astro = '/home/slava/science/research/kulkarni/JWST-DLAs/ID2155/results/figures/fig-fit-profiles/dust_templates/Templates/profiles/Obs-Template/'

path_to_sdss_spectrum = '/home/slava/science/research/kulkarni/JWST-DLAs/ID2155/sdss/spec-10453-58136-0558-j1007.fits'
path_to_jwst_spectrum = '/home/slava/science/research/kulkarni/JWST-DLAs/ID2155/results/J1007/J1007_scailing_2s-rebinned24.fits'
path_to_composite_spectra = '/home/slava/science/article/absorption_spectra/Composite_spectra/'
path_to_mir_spectra = '/home/slava/science/codes/python/dust_extinction_model/'



qso_composite_vanderberk = path_to_composite_spectra + 'Vanderberk/sdss.txt'
uv_qso_composite = np.loadtxt(qso_composite_vanderberk)
col1 = uv_qso_composite[:, 0]
col2 = uv_qso_composite[:, 1]

s_uv_qso = spectrum(x=col1,y=col2*col1**2*1e-7)
s_uv_qso.interp = interp1d(s_uv_qso.x, s_uv_qso.y,fill_value='extrapolate')

qso_composite_glikman = path_to_composite_spectra + 'Glikman/J_ApJ_640_579/table7.dat'
nir_qso_composite = np.loadtxt(qso_composite_glikman)
col1 = nir_qso_composite[:, 0]
col2 = nir_qso_composite[:, 1]
n = 4
col1 = rebin_arr(col1, n)
col2 = rebin_arr(col2, n)
s_nir_qso = spectrum(x=col1,y=col2*col1**2*1e-7)



qso_composite_selsing_path = path_to_composite_spectra + 'Selsing/spectrum.dat'
f = np.loadtxt(qso_composite_selsing_path)
col1 = f[:, 0]
col2 = f[:, 1]
mask_absorption = col2 == 0
s_selsing_qso = spectrum(x=col1[~mask_absorption], y=col2[~mask_absorption] * col1[~mask_absorption] ** 2 * 1e-7)
s_selsing_qso.ynu = np.array(col2[~mask_absorption])
#s_selsing_qso.interp = interp1d(s_selsing_qso.x,s_selsing_qso.y)
#fnorm = np.mean(s_nir_qso.y[(s_nir_qso.x > 2800) * (s_nir_qso.x < 3200)]) / np.mean(
#    s_selsing_qso.y[(s_selsing_qso.x > 2800) * (s_selsing_qso.x < 3200)])
#s_selsing_qso.y *= fnorm


qso_composite_jiang = path_to_composite_spectra + 'Jiang/Jiang2011.dat'
jiang_qso_composite = np.loadtxt(qso_composite_jiang)
col1 = jiang_qso_composite[:, 0]
col2 = jiang_qso_composite[:, 1]

s_jiang_qso = spectrum(x=np.array(col1),y=np.array(col2*col1**2*1e-7))
s_jiang_qso.interp = interp1d(s_jiang_qso.x, s_jiang_qso.y,fill_value='extrapolate')


#renormaliztion
def renorm_qso(x,y,lc=8000,dl=200): #3250
    mask = np.abs(x-lc)<dl
    ynorm = np.nanmean(y[mask])
    return y/ynorm

#fnorm = np.mean(s_uv_qso.y[np.abs(s_uv_qso.x - 3250)<200]) / np.mean(
#    s_selsing_qso.y[np.abs(s_selsing_qso.x - 3250)<200])
s_uv_qso.y = renorm_qso(s_uv_qso.x,s_uv_qso.y)
s_uv_qso.interp = interp1d(s_uv_qso.x, s_uv_qso.y, fill_value='extrapolate')

s_jiang_qso.y = renorm_qso(s_jiang_qso.x,s_jiang_qso.y)
s_jiang_qso.interp = interp1d(s_jiang_qso.x, s_jiang_qso.y, fill_value='extrapolate')

s_selsing_qso.y = renorm_qso(s_selsing_qso.x,s_selsing_qso.y)
s_selsing_qso.interp = interp1d(s_selsing_qso.x, s_selsing_qso.y, fill_value='extrapolate')

s_nir_qso.y = renorm_qso(s_nir_qso.x,s_nir_qso.y)
s_nir_qso.interp = interp1d(s_nir_qso.x, s_nir_qso.y, fill_value='extrapolate')


#MIR models
mir_hatzi_qso_composite = np.loadtxt(path_to_mir_spectra+'/data/MIR/J_AJ_129_1198/table2.dat')
#noramlized at 0.3 micron
col1 = mir_hatzi_qso_composite[:, 0]
col2 = mir_hatzi_qso_composite[:, 1]
s_hatzi_qso = spectrum(x=np.array(col1),y=np.array(col2*col1**2))
s_hatzi_qso.interp = interp1d(s_hatzi_qso.x, s_hatzi_qso.y,fill_value='extrapolate')

mir_hernan_qso_composite = np.loadtxt(path_to_mir_spectra+'/data/MIR/Hernan-Caballero-16/table1.dat',skiprows=17)
col1 = mir_hernan_qso_composite[:, 0]
col2 = mir_hernan_qso_composite[:, 1]
s_hernan_qso = spectrum(x=np.array(col1)*1e4,y=np.array(col2))
s_hernan_qso.interp = interp1d(s_hernan_qso.x, s_hernan_qso.y,fill_value='extrapolate')



plt.subplots()
plt.plot(col1,col2,ls='-')
plt.xscale('log')
plt.yscale('log')
plt.show()

#fnorm = np.mean(s_selsing_qso.y[np.abs(s_selsing_qso.x - 11000)<200]) / np.mean(s_nir_qso.y[np.abs(s_nir_qso.x - 11000)<200])
#s_nir_qso.y *= fnorm
#s_nir_qso.interp = interp1d(s_nir_qso.x, s_nir_qso.y, fill_value='extrapolate')



mask_uv = (s_uv_qso.x<=1100)
mask_opt = (s_selsing_qso.x > 1100) * (s_selsing_qso.x <= 11000)
mask_nir = (s_nir_qso.x > 11000)

x = np.append(s_uv_qso.x[mask_uv],s_selsing_qso.x[mask_opt])
x = np.append(x,s_nir_qso.x[mask_nir])
y = np.append(s_uv_qso.y[mask_uv],s_selsing_qso.y[mask_opt])
y = np.append(y,s_nir_qso.y[mask_nir])

s_combined_composite = spectrum(x,y)


#s_combined_composite.x /= 1e4 #convert to microns
class qso_composite():
    def __init__(self, units='angstrom',smooth_ir=False,mir_interp=True,mode='VSG',flux_units='Jy',debug=False,wave_mode = 'normal'):
        # F in Jy (F_nu)
        if mode == 'VSG':
            mask_uv = (s_uv_qso.x <= 1100)
            mask_opt = (s_selsing_qso.x > 1100) * (s_selsing_qso.x <= 10400)
            mask_nir = (s_nir_qso.x > 10400)

            x,y = [],[]
            x = np.append(x,s_uv_qso.x[mask_uv])
            y = np.append(y,s_uv_qso.y[mask_uv])

            f_norm_opt = np.nanmean(s_uv_qso.y[np.abs(s_uv_qso.x-1100)<100])/np.nanmean(s_selsing_qso.y[np.abs(s_selsing_qso.x-1100)<100])
            x = np.append(x,s_selsing_qso.x[mask_opt])
            y = np.append(y,s_selsing_qso.y[mask_opt]*f_norm_opt)

            f_norm_nir =   np.nanmean(y[np.abs(x - 10300) < 100])/ np.nanmean(s_nir_qso.y[np.abs(s_nir_qso.x - 10300) < 100])
            x = np.append(x, s_nir_qso.x[mask_nir])
            y = np.append(y, s_nir_qso.y[mask_nir] * f_norm_nir)

            if wave_mode == 'normal':
                self.x = x
                self.y = y
            elif wave_mode == 'extended':
                self.x = np.append(x,np.linspace(x[-1],30*1e4,50))
                self.y = np.append(y,np.zeros(50))
            self.y = renorm_qso(self.x, self.y)

            if mir_interp:
                self.y[self.x > 2e4] = 3.08672 * (self.x[self.x > 2e4] / 19470.561) ** (1.17085)

        if mode == 'VG':
            mask_uv = (s_uv_qso.x <= 3000)
            mask_nir = (s_nir_qso.x > 3000)

            x, y = [], []
            x = np.append(x, s_uv_qso.x[mask_uv])
            y = np.append(y, s_uv_qso.y[mask_uv])


            f_norm_nir = np.nanmean(y[np.abs(x - 3000) < 100]) / np.nanmean(s_nir_qso.y[np.abs(s_nir_qso.x - 3000) < 100])
            x = np.append(x, s_nir_qso.x[mask_nir])
            y = np.append(y, s_nir_qso.y[mask_nir] * f_norm_nir)


            self.x = x
            self.y = y
            self.y = renorm_qso(self.x, self.y)

            if mir_interp:
                self.y[self.x > 2e4] = 3.08672 * (self.x[self.x > 2e4] / 19470.561) ** (1.17085)

        #Vandenberg-Selsing-Glickman-Hernan
        if mode == 'VSGH':
            mask_uv = (s_uv_qso.x <= 1100)
            mask_opt = (s_selsing_qso.x > 1100) * (s_selsing_qso.x <= 10400)
            mask_nir = (s_nir_qso.x > 10400)*(s_nir_qso.x < 21000)
            mask_mir = (s_hernan_qso.x > 21000)

            x, y = [], []
            x = np.append(x, s_uv_qso.x[mask_uv])
            y = np.append(y, s_uv_qso.y[mask_uv])

            f_norm_opt = np.nanmean(s_uv_qso.y[np.abs(s_uv_qso.x - 1100) < 100]) / np.nanmean(
                s_selsing_qso.y[np.abs(s_selsing_qso.x - 1100) < 100])
            x = np.append(x, s_selsing_qso.x[mask_opt])
            y = np.append(y, s_selsing_qso.y[mask_opt] * f_norm_opt)

            f_norm_nir = np.nanmean(y[np.abs(x - 10300) < 100]) / np.nanmean(
                s_nir_qso.y[np.abs(s_nir_qso.x - 10300) < 100])
            x = np.append(x, s_nir_qso.x[mask_nir])
            y = np.append(y, s_nir_qso.y[mask_nir] * f_norm_nir)

            f_norm_mir = np.nanmean(y[np.abs(x - 20900) < 100]) / np.nanmean(s_hernan_qso.y[np.abs(s_hernan_qso.x - 20900) < 100])
            x = np.append(x, s_hernan_qso.x[mask_mir])
            y = np.append(y, s_hernan_qso.y[mask_mir] * f_norm_mir)

            if wave_mode == 'normal':
                self.x = x
                self.y = y
            elif wave_mode == 'extended':
                self.x = np.append(x, np.linspace(x[-1], 30 * 1e4, 50))
                self.y = np.append(y, np.zeros(50))
            self.y = renorm_qso(self.x, self.y)

            #if mir_interp:
            #    self.y[self.x > 2e4] = 3.08672 * (self.x[self.x > 2e4] / 19470.561) ** (1.17085)

        elif mode == 'JG':
            mask_opt = (s_jiang_qso.x <= 8000)
            mask_nir = (s_nir_qso.x > 8000)

            x, y = [], []
            x = np.append(x, s_jiang_qso.x[mask_opt])
            y = np.append(y, s_jiang_qso.y[mask_opt])

            f_norm_nir = np.nanmean(y[np.abs(x - 7900) < 100]) / np.nanmean(
                s_nir_qso.y[np.abs(s_nir_qso.x - 7900) < 100])
            x = np.append(x, s_nir_qso.x[mask_nir])
            y = np.append(y, s_nir_qso.y[mask_nir] * f_norm_nir)

            self.x = x
            self.y = y

            self.y = renorm_qso(self.x, self.y)

            if mir_interp:
                self.y[self.x > 2e4] = 4.852695841834076 * (self.x[self.x > 2e4] / 20051.60) ** (1.170858849426)


        if flux_units == 'Jy':
            print()
        elif flux_units == 'F_lambda':
            self.y = renorm_qso(self.x, self.y/self.x**2)

        if smooth_ir:
            self.y[self.x <  11000] = savgol_filter(self.y[self.x <  11000], 50, 3)

        self.units = units
        if units == 'micron':
            self.x /= 1e4
        self.interp()

        if debug:
            self.cont_plot()

    def interp(self):
        self.interp = interp1d(self.x,self.y)

    def cont_plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.x,self.y,'o',label='composite',zorder=10)
        ax.plot(s_uv_qso.x ,s_uv_qso.y,label='UV')
        ax.plot( s_nir_qso.x, s_nir_qso.y,label='Glikman')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    fig, ax = plt.subplots()
    ax.plot(s_selsing_qso.x, s_selsing_qso.y, label='Selsing')
    ax.plot(s_nir_qso.x, s_nir_qso.y, label='Glikman')
    ax.plot(s_uv_qso.x, s_uv_qso.y, label='Vanderberk',ls='--')
    ax.plot(s_jiang_qso.x, s_jiang_qso.y, label='Jiang')

    q_comp = qso_composite(mir_interp=False,mode='VG',flux_units = 'Jy', debug=False)
    ax.plot(q_comp.x, q_comp.y, 'o', label='Composite', markersize=4, zorder=-10)

    plt.legend()
    plt.show()

    q_comp = qso_composite(mir_interp=False, mode='VSGH', flux_units='Jy')
    a = np.zeros((q_comp.x.shape[0],2))
    a[:,0] = q_comp.x
    a[:, 1] = q_comp.y
    #np.savetxt('quasar_composite.txt',a)
    #plt.subplots()
    #plt.plot(q_comp.x,q_comp.y)
    #plt.show()

    l_ref = 0.3
    #ax.plot(s_hatzi_qso.x*1e4, s_hatzi_qso.y/s_hatzi_qso.interp(l_ref)*s_nir_qso.interp(l_ref*1e4), label='Hatzi')
    #ax.plot(s_hernan_qso.x, s_hernan_qso.y / s_hernan_qso.interp(l_ref) * s_nir_qso.interp(l_ref*1e4), label='Hernan')

    if 0:
        ax.plot(s_uv_qso.x,renorm_qso(s_uv_qso.x,s_uv_qso.x**0.28),label='a=0.3')
        ax.plot(s_uv_qso.x,renorm_qso(s_uv_qso.x,s_uv_qso.x**0.46),label='a=0.5')
        ax.plot(s_uv_qso.x,renorm_qso(s_uv_qso.x,s_uv_qso.x**0.28)-renorm_qso(s_uv_qso.x,s_uv_qso.x**0.46),ls='--')
        xx = np.linspace(0,12e3,100)
        ax.plot(xx, renorm_qso(xx, xx ** (-0.1))-1, ls=':')
    ax.set_xlabel('Wavelengh (AGN), A')
    ax.set_ylabel('F$_{\\nu}$, a.u.')
    ax.legend()
    plt.show()





    from astropy import modeling


    fitter = modeling.fitting.LevMarLSQFitter()
    #fitter = modeling.fitting.PowerLaw1D()
    #astropy.modeling.powerlaws.PowerLaw1D
    model = modeling.powerlaws.PowerLaw1D(x_0=20000)
    mask = (q_comp.x>16500)*(np.abs(q_comp.x-18700)>300)*(np.abs(q_comp.x-23000)>2500)
    y =  q_comp.y[mask]
    x=q_comp.x[mask]
    fitted_model = fitter(model, x, y)
    amp = fitted_model.amplitude.value
    x_0 = fitted_model.x_0.value
    alpha = fitted_model.alpha.value
    print(amp,x_0,alpha)

    plt.subplots()
    plt.plot(q_comp.x,q_comp.y,label='data')
    plt.plot(q_comp.x[mask],fitted_model(q_comp.x[mask]),label='fit')
    plt.legend()
    plt.show()


    q_comp = qso_composite(mir_interp=True,flux_units='F_lambda')
    plt.subplots()
    plt.plot(q_comp.x,q_comp.y)

    d = np.zeros((np.size(q_comp.x),2))
    d[:,0] = q_comp.x
    d[:, 1] = q_comp.y
    np.savetxt('composite_qso.dat',d)
    plt.show()

