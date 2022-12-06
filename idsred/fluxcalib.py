# ING-IDS spectroscopic standards:
# https://www.ing.iac.es//Astronomy/observing/manuals/html_manuals/tech_notes/tn065-100/workflux.html

import warnings
import pandas as pd
import matplotlib.pyplot as plt

import astropy.units as u
from specutils.spectra import Spectrum1D
from specutils.fitting.continuum import fit_continuum
from scipy.interpolate import UnivariateSpline

from wavecalib import wavelength_function

def find_skiprows(filename):
    skiprows = 0
    with open(filename) as file:
        for i, line in enumerate(file.readlines()):
            if '*' in line:
                skiprows = i
    skiprows += 1

    return skiprows


def convert_flux(calspec):
    wave = calspec['wave'].values
    if 'mag' in calspec.columns:
        mag = calspec['mag'].values  # assumed to be in AB
        flux_nu = 10 ** (-0.4 * (mag + 48.60))
    else:
        # flux is assumed to be in mJy
        flux_Jy = calspec['flux_mJy'].values * 1e-3  # mJy to Jy
        flux_nu = flux_Jy * 1e-23

    flux_lam = flux_nu * (3e18) / (wave ** 2)
    calspec['flux'] = flux_lam


def get_calspec(filename):
    skiprows = find_skiprows(filename)
    if filename.endswith('a.sto') or filename.endswith('a.og'):
        columns = ['wave', 'mag']
    else:
        columns = ['wave', 'flux_mJy']

    calspec = pd.read_csv(filename, delim_whitespace=True, skiprows=skiprows, names=columns)
    convert_flux(calspec)

    return calspec


def plot_calspec(calspec, units='flux'):
    fig, ax = plt.subplots(figsize=(8, 6))

    if units == 'mag':
        ax.plot(calspec['wave'], calspec['mag'], lw=2)
        ax.set_ylabel('Magnitude', fontsize=16)
    elif units == 'flux':
        ax.plot(calspec['wave'], calspec['flux'], lw=2)
        ax.set_ylabel(r'$F_{\lambda}$', fontsize=16)
    else:
        raise ValueError('Not a valid unit ("flux" or "mag" only)')
    ax.set_xlabel('Wavelength ($\AA$)', fontsize=16)

    plt.show()


def fit_calspec_continuum(calspec, window=None, plot=False):
    spectrum = Spectrum1D(flux=calspec['flux'].values * u.erg,
                          spectral_axis=calspec['wave'].values * u.angstrom)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cs_fit = fit_continuum(spectrum, window=window)

    # for plotting purposes only
    continuum_fit = cs_fit(calspec['wave'].values * u.angstrom)

    if plot:
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(calspec['wave'], calspec['flux'], lw=2)
        ax.plot(calspec['wave'], continuum_fit, lw=2, label='Continuum fit')
        ax.set_xlabel('Wavelength ($\AA$)', fontsize=16)
        ax.set_ylabel(r'$F_{\lambda}$', fontsize=16)
        ax.legend(fontsize=14)

        plt.show()

    return cs_fit


def fit_sensfunc(raw_spectrum, params, cs_fit, plot=False):
    # ratio between observed standard and the continuum of the "archive" standard
    raw_wave = np.arange(len(raw_spectrum))
    cal_wave = wavelength_function(params, raw_wave)
    ratio = raw_spectrum / cs_fit(cal_wave * u.angstrom)
    log_ratio = np.log10(np.abs(ratio.value))

    # fit with spline
    mask = (3800 < cal_wave) & (cal_wave < 9000)
    sensfunc = UnivariateSpline(cal_wave[mask], log_ratio[mask], k=4)

    if plot:
        fig, ax = plt.subplots(figsize=(8, 6))

        plt.plot(cal_wave, ratio, lw=2, label='Ratio')
        plt.plot(cal_wave, 10 ** sensfunc(cal_wave), lw=2, label='Fit')
        ax.set_ylabel('Sensitivity function', fontsize=16)
        ax.set_xlabel(r'Wavelength ($\AA$)', fontsize=16)
        ax.legend(fontsize=14)

        plt.show()

    return sensfunc