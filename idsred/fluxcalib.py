# ING-IDS spectroscopic standards:
# https://www.ing.iac.es//Astronomy/observing/manuals/html_manuals/tech_notes/tn065-100/workflux.html
import os
import glob
import json
import requests

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import dotenv_values

import astropy.units as u
from astropy.io import fits
from specutils.spectra import Spectrum1D
from specutils.fitting.continuum import fit_continuum
from scipy.interpolate import UnivariateSpline

import idsred
idsred_path = idsred.__path__[0]
from .wavecalib import apply_wavesol

# List of extension for the standard stars from the ING catalog.
# Those with "a." are in AB magnitudes, while those without are in flux (mJy).
std_extensions = ['.oke', '.sto', '.og', 'a.oke', 'a.sto', 'a.og']
std_json_file = os.path.join(idsred_path, 'standards', 'standards.json')

def download_std(std_name):
    ing_cat_url = 'https://www.ing.iac.es//Astronomy/observing/manuals/html_manuals/tech_notes/tn065-100/workflux.html'
    response = requests.get(ing_cat_url)

    # Search for the URL of the standard star
    if response.status_code == 200:
        for line in response.text.split():
            if std_name in line:
                std_line = line
                html = std_line.split('"')[1]

    # Standard star name as in the ING catalog
    cat_name = os.path.basename(html).split('.')[0]

    # Download any of the available files (in flux or mag)
    global std_extensions
    for ext in std_extensions:
        url = os.path.join(os.path.dirname(html), f'{cat_name}{ext}')
        response = requests.get(url)
        if response.status_code == 200:
            target_file = os.path.basename(url)
            global idsred_path
            outfile = os.path.join(idsred_path, 'standards', target_file)
            with open(outfile, "w") as f_out:
                f_out.write(response.text)

                # update standard stars dictionary
                global std_json_file
                with open(std_json_file, 'r') as file:
                    std_dict = json.load(file)

                std_dict[std_name] = cat_name
                with open(std_json_file, 'w') as file:
                    file.write(json.dumps(std_dict))

                return outfile

def get_std_file(std_name):
    global idsred_path, std_json_file
    with open(std_json_file, 'r') as file:
        std_dict = json.load(file)

    if std_name in std_dict.keys():
        # check if the star already exists locally
        cat_name = std_dict[std_name]
        global std_extensions
        for ext in std_extensions:
            target_file = f'{cat_name}{ext}'
            outfile = os.path.join(idsred_path, 'standards', target_file)
            if os.path.isfile(outfile):
                return outfile
    else:
        # download the star file if not found
        outfile = download_std(std_name)
        return outfile


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


def get_calspec(std_name):
    filename = get_std_file(std_name)
    skiprows = find_skiprows(filename)
    if filename.endswith('a.sto') or filename.endswith('a.og') or filename.endswith('a.oke'):
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


def get_standard(std_name, fmask=True):

    config = dotenv_values(".env")
    PROCESSING = config['PROCESSING']
    obs_std_file = os.path.join(PROCESSING, f'{std_name}_1d.fits')

    hdu = fits.open(obs_std_file)
    raw_flux = hdu[0].data
    cols = np.arange(len(raw_flux))
    cal_wave = apply_wavesol(cols)

    if fmask:
        mask = raw_flux >= 0.0
        cal_wave = cal_wave[mask]
        raw_flux = raw_flux[mask]

    return cal_wave, raw_flux

def fit_sensfunc(std_name=None, fmask=True, degree=5, xmin=None, xmax=None, plot_diag=False):

    if std_name is None:
        config = dotenv_values(".env")
        PROCESSING = config['PROCESSING']
        std_files = os.path.join(PROCESSING, '*_1d.fits')
        basename = os.path.basename(glob.glob(std_files)[0])
        std_name = basename.split('_')[0]

    cal_wave, raw_flux = get_standard(std_name, fmask)  # observed standard
    calspec = get_calspec(std_name)  # catalog/calibrated standard
    interp_calflux = np.interp(cal_wave, calspec.wave.values, calspec.flux.values)

    # mask wavelength range
    wave_min = calspec.wave.values.min()
    wave_max = calspec.wave.values.max()
    wave_mask = (wave_min <= cal_wave) & (cal_wave <= wave_max)

    if xmin is not None:
        wave_mask = wave_mask & (xmin <= cal_wave)
    if xmax is not None:
        wave_mask = wave_mask & (cal_wave <= xmax)

    cal_wave = cal_wave[wave_mask]
    raw_flux = raw_flux[wave_mask]
    interp_calflux = interp_calflux[wave_mask]

    if plot_diag:
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax[0].plot(cal_wave, raw_flux / raw_flux.max(), label='Obs. Spec.')
        ax[0].plot(cal_wave, interp_calflux / interp_calflux.max(), label='Cal. Spec.')
        ax[0].legend(fontsize=14)

    # sensitivity function calculation starts here
    flux_ratio = interp_calflux / raw_flux
    log_ratio = np.log10(flux_ratio)

    coefs = np.polyfit(cal_wave, log_ratio, degree)
    log_sensfunc = np.polyval(coefs, cal_wave)
    sensfunc = 10 ** log_sensfunc

    if plot_diag:
        ax[1].plot(cal_wave, sensfunc, label=f'Sens. Func. (deg. {degree})', color='g')
        ax[1].plot(cal_wave, flux_ratio, label='ratio (Cal./Obs.)', color='k')
        ax[1].set_xlabel(r'Wavelength ($\AA$)', fontsize=16)
        ax[1].set_ylabel('Inverse Sensitivity', fontsize=16)
        ax[1].set_yscale('log')
        ax[1].legend(fontsize=14)
        plt.show()

    min_wave, max_wave = cal_wave.min(), cal_wave.max()
    save_sensfunc(min_wave, max_wave, coefs)


def save_sensfunc(xmin, xmax, coefs):
    """Saves the sensitivity function coefficients.

    Parameters
    ----------
    coefs

    Returns
    -------

    """
    config = dotenv_values(".env")
    PROCESSING = config['PROCESSING']
    sensfunc_file = os.path.join(PROCESSING, 'sensfunc.txt')

    with open(sensfunc_file, 'w') as file:
        file.write(f'xmin xmax: {xmin} {xmax}\n')
        coefs_str = ' '.join(str(coef) for coef in coefs)
        file.write(f'coefficients: {coefs_str}\n')


def load_sensfunc():
    """Loads the sensitivity function coefficients.

    Returns
    -------

    """
    config = dotenv_values(".env")
    PROCESSING = config['PROCESSING']
    sensfunc_file = os.path.join(PROCESSING, 'sensfunc.txt')

    with open(sensfunc_file, 'r') as file:
        lines = file.read().splitlines()

    xmin_xmax = lines[0].split(' ')
    xmin, xmax = float(xmin_xmax[-2]), float(xmin_xmax[-1])
    coefs_line = lines[1].split(' ')
    coefs = [float(coef) for coef in coefs_line[1:]]

    return xmin, xmax, coefs

def apply_sensfunc(wavelength, raw_flux):
    """Applies the sensitivity function to an uncalibrated spectrum.

    Parameters
    ----------
    xdata
    raw_flux

    Returns
    -------

    """
    min_wave, max_wave, coefs = load_sensfunc()
    log_sensfunc = np.polyval(coefs, wavelength)
    sensfunc = 10 ** log_sensfunc
    flux = raw_flux * sensfunc

    mask = (min_wave <= wavelength) & (wavelength <= max_wave)
    wavelength = wavelength[mask]
    flux = flux[mask]

    return wavelength, flux





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


def fit_sensfunc_OLD(raw_spectrum, params, cs_fit, plot=False):
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