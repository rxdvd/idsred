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
from specutils.manipulation import box_smooth

from .wavecalib import apply_wavesol

import idsred

idsred_path = idsred.__path__[0]

# List of extension for the standard stars from the ING catalog.
# Those with "a." are in AB magnitudes, while those without are in flux (mJy).
std_extensions = [".oke", ".sto", ".og", "a.oke", "a.sto", "a.og"]
std_json_file = os.path.join(idsred_path, "standards", "standards.json")


def download_std(std_name):
    """Downloads an ING standard star.

    URL used:
    https://www.ing.iac.es//Astronomy/observing/manuals/html_manuals/tech_notes/tn065-100/workflux.html

    Parameters
    ----------
    std_name: str
        Standard star name. E.g. `SP0105+625`.
    """
    ing_cat_url = "https://www.ing.iac.es//Astronomy/observing/manuals/html_manuals/tech_notes/tn065-100/workflux.html"
    response = requests.get(ing_cat_url)

    # Search for the URL of the standard star
    html = None
    if response.status_code == 200:
        print(response.text.split())
        for line in response.text.split():
            if std_name in line:
                std_line = line
                html = std_line.split('"')[1]

    assert html is not None, f"No URL found to download the SED of {std_name}"

    # Standard star name as in the ING catalog
    cat_name = os.path.basename(html).split(".")[0]

    # Download any of the available files (in flux or mag)
    global std_extensions
    for ext in std_extensions:
        url = os.path.join(os.path.dirname(html), f"{cat_name}{ext}")
        response = requests.get(url)
        if response.status_code == 200:
            target_file = os.path.basename(url)
            global idsred_path
            outfile = os.path.join(idsred_path, "standards", target_file)
            with open(outfile, "w") as f_out:
                f_out.write(response.text)

                # update standard stars dictionary
                global std_json_file
                with open(std_json_file, "r") as file:
                    std_dict = json.load(file)

                std_dict[std_name] = cat_name
                with open(std_json_file, "w") as file:
                    file.write(json.dumps(std_dict))

                return outfile


def get_std_file(std_name):
    """Gets the standard star file.

    The file is downloaded if it is not found in the
    local installation.

    Parameters
    ----------
    std_name: str
        Standard star name. E.g. `SP0105+625`.

    Returns
    -------
    outfile: str
        Full path of the standard star file.
    """
    global idsred_path, std_json_file
    with open(std_json_file, "r") as file:
        std_dict = json.load(file)

    if not std_name.startswith("SP"):
        # this is for non-ING standard stars (e.g. ESO STDs)
        # name format should be fGD71.dat, where 'f' stands for flux
        outfile = os.path.join(idsred_path, "standards", f"f{std_name}.dat")
        return outfile
    elif std_name in std_dict.keys():
        # check if the star already exists locally
        cat_name = std_dict[std_name]
        global std_extensions
        for ext in std_extensions:
            target_file = f"{cat_name}{ext}"
            outfile = os.path.join(idsred_path, "standards", target_file)
            if os.path.isfile(outfile):
                return outfile
    else:
        # download the star file if not found
        outfile = download_std(std_name)
        return outfile


def _find_skiprows(filename):
    """Finds out how many rows to skip when reading the
    standard star files.

    Parameters
    ----------
    filename: str
        Standard star file.
    """
    skiprows = 0
    with open(filename) as file:
        for i, line in enumerate(file.readlines()):
            if "*" in line:
                skiprows = i + 1

    return skiprows


def _convert_flux(calspec):
    """Convert the flux of a calibrated ING standard star.

    Conversion from AB magnitudes or flux in mJy to flux
    in erg/cm**2/s/AA

    Parameters
    ----------
    calspec: `pandas.DataFrame`
        Calibrated standard star SED.
    """
    wave = calspec["wave"].values
    if "mag" in calspec.columns:
        mag = calspec["mag"].values  # assumed to be in AB
        flux_nu = 10 ** (-0.4 * (mag + 48.60))
    else:
        # flux is assumed to be in mJy
        flux_Jy = calspec["flux_mJy"].values * 1e-3  # mJy to Jy
        flux_nu = flux_Jy * 1e-23

    flux_lam = flux_nu * 3e18 / (wave**2)
    calspec["flux"] = flux_lam


def _get_calspec(std_name):
    """Retrieves the calibrated standard star SED.

    Parameters
    ----------
    std_name: str
        Standard star name. E.g. `SP0105+625`.

    Returns
    -------
    calspec: `pandas.DataFrame`
        Calibrated standard star SED.
    """
    filename = get_std_file(std_name)
    skiprows = _find_skiprows(filename)

    if filename.endswith(".dat"):
        # non-ING standards
        columns = ["wave", "flux"]
        needs_conversion = False
    elif (
        filename.endswith("a.sto")
        or filename.endswith("a.og")
        or filename.endswith("a.oke")
    ):
        columns = ["wave", "mag"]
        needs_conversion = True

    else:
        columns = ["wave", "flux_mJy"]
        needs_conversion = True

    calspec = pd.read_csv(
        filename, delim_whitespace=True, skiprows=skiprows, names=columns
    )
    if needs_conversion is True:
        _convert_flux(calspec)

    return calspec


def plot_calspec(calspec, units="flux"):
    """Plots the calibrated standard star SED.

    Parameters
    -------
    calspec: str or `pandas.DataFrame`
        Standard star name or calibrated SED.
    units: str, default ``flux``
        Either ``flux`` or ``mag``.
    """
    if type(calspec) == str:
        calspec = _get_calspec(calspec)

    fig, ax = plt.subplots(figsize=(12, 6))

    if units == "mag":
        ax.plot(calspec["wave"], calspec["mag"], lw=2)
        ax.set_ylabel("Magnitude", fontsize=16)
    elif units == "flux":
        ax.plot(calspec["wave"], calspec["flux"], lw=2)
        ax.set_ylabel(r"$F_{\lambda}$", fontsize=16)
    else:
        raise ValueError('Not a valid unit ("flux" or "mag" only)')
    ax.set_xlabel("Wavelength ($\AA$)", fontsize=16)

    plt.show()


def get_standard(std_name, fmask=True, use_master_arc=False):
    """Gets the observed standard star SED.

    The SED is wavelength calibrated.

    Parameters
    ----------
    std_name: str
        Standard star name. E.g. `SP0105+625`.
    fmask: bool, default ``True``
        If ``True``, negative fluxes are masked out.
    use_master_arc: bool, default ``False``
        If ``True``, the wavelength solution from the master ARC
        is used instead of the target's specific solution.


    Returns
    -------
    cal_wave: array
        Wavelength of the standard star SED.
    raw_flux: array
        Uncalibrated flux of the standard star SED.
    hdu: `~fits.hdu`
        Header Data Unit of the standard star file.
    """
    config = dotenv_values(".env")
    PROCESSING = config["PROCESSING"]
    obs_std_file = os.path.join(PROCESSING, f"{std_name}_1d.fits")

    hdu = fits.open(obs_std_file)
    raw_flux = hdu[0].data
    cols = np.arange(len(raw_flux))
    if use_master_arc is True:
        wavesol_file = "wavesol.txt"
    else:
        wavesol_file = f"wavesol_{std_name}.txt"
    cal_wave = apply_wavesol(cols, wavesol_file)

    if fmask:
        mask = raw_flux >= 0.0
        cal_wave = cal_wave[mask]
        raw_flux = raw_flux[mask]

    return cal_wave, raw_flux, hdu


def _calc_airmass(hdu):
    """Calculates the airmass for a given target.

    The zenith es calculated as the average of
    ZDSTART and ZDEND. The airmass is the secant
    of zenit: X = sec(z).

    Parameters
    ----------
    hdu: ~fits.hdu
        Header Data Unit.

    Returns
    -------
    airmass: float
        Target's airmass.
    """
    zenith = (hdu[0].header["ZDSTART"] + hdu[0].header["ZDEND"]) / 2
    zenith_rad = zenith * np.pi / 180
    airmass = 1 / np.cos(zenith_rad)

    return airmass


def _correct_extinction(wave, flux, airmass):
    """Corrects a spectrum for atmospheric extinction.

    Parameters
    ----------
    wave: array
        Target's wavelength.
    flux: array
        Target's flux.
    airmass: float
        Target's airmass.

    Returns
    -------
    corr_flux: array
        Extinction-corrected flux.
    """
    ext_path = os.path.join(idsred.__path__[0], "extinction/lapalmaext.txt")
    _wave, _ext_mag = np.loadtxt(
        ext_path
    ).T  # _ext_mag in units of mag/airmass

    _ext = 10 ** (_ext_mag * airmass)
    _ext = np.interp(wave, _wave, _ext)

    corr_flux = flux * _ext

    return corr_flux


def fit_sensfunc(
    std_name=None,
    fmask=True,
    degree=5,
    wmin=3600,
    wmax=None,
    plot_diag=False,
    use_master_arc=False,
):
    """Fits the sensitivity function using a standard star.

    A simple polynomial is used.

    Parameters
    ----------
    std_name: str, default ``None``
        Standard star name. E.g. `SP0105+625`. If ``None``,
        use the first one.
    fmask: bool, default ``True``
        If ``True``, negative fluxes are masked out.
    degree: float, default ``5``
        Degree of the polynomial
    wmin: float, default ``3600``
        Minimum wavelength to use.
    wmax: float, default ``None``
        Maximum wavelength to use.
    plot_diag: bool, default ``False``
        If ``True``, diagnostic plots are shown with the solution.
    use_master_arc: bool, default ``False``
        If ``True``, the wavelength solution from the master ARC
        is used instead of the target's specific solution.
    """
    config = dotenv_values(".env")
    PROCESSING = config["PROCESSING"]

    if std_name is None:
        # choose first standard found
        std_files = os.path.join(PROCESSING, "SP*_1d.fits")
        basename = os.path.basename(glob.glob(std_files)[0])
        std_name = basename.split("_")[0]

    # observed standard
    cal_wave, raw_flux, std_hdu = get_standard(std_name, fmask, use_master_arc)
    # correct for atmospheric extinction at Observatorio Roque de Los Muchachos
    airmass = _calc_airmass(std_hdu)
    raw_flux = _correct_extinction(cal_wave, raw_flux, airmass)

    # catalog/calibrated standard
    calspec = _get_calspec(std_name)
    interp_calflux = np.interp(
        cal_wave, calspec.wave.values, calspec.flux.values
    )

    # mask wavelength range
    wave_min = calspec.wave.values.min()
    wave_max = calspec.wave.values.max()
    wave_mask = (wave_min <= cal_wave) & (cal_wave <= wave_max)

    if wmin is not None:
        wave_mask = wave_mask & (wmin <= cal_wave)
    if wmax is not None:
        wave_mask = wave_mask & (cal_wave <= wmax)

    cal_wave = cal_wave[wave_mask]
    raw_flux = raw_flux[wave_mask]
    interp_calflux = interp_calflux[wave_mask]

    # sensitivity function calculation starts here
    flux_ratio = interp_calflux / raw_flux
    log_ratio = np.log10(flux_ratio)

    coefs = np.polyfit(cal_wave, log_ratio, degree)
    log_sensfunc = np.polyval(coefs, cal_wave)
    sensfunc = 10**log_sensfunc

    # calculate telluric correction
    tellurics = sensfunc / flux_ratio
    mask = ((cal_wave > 6860) & (cal_wave < 6910)) | (
        (cal_wave > 7570) & (cal_wave < 7680)
    )
    tellurics[~mask] = 1
    tellurics[tellurics > 1] = 1

    tellurics_file = os.path.join(PROCESSING, "telluric_correction.txt")
    np.savetxt(tellurics_file, np.array([cal_wave, tellurics]).T, fmt="%.2f")

    if plot_diag:
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        ax[0].plot(cal_wave, raw_flux / raw_flux.max(), label="Obs. Spec.")
        ax[0].plot(
            cal_wave, interp_calflux / interp_calflux.max(), label="Cal. Spec."
        )
        ax[0].set_title(std_name, fontsize=16)
        ax[0].legend(fontsize=14)

        ax[1].plot(
            cal_wave, sensfunc, label=f"Sens. Func. (deg. {degree})", color="g"
        )
        ax[1].plot(cal_wave, flux_ratio, label="ratio (Cal./Obs.)", color="k")
        ax[1].set_xlabel(r"Wavelength ($\AA$)", fontsize=16)
        ax[1].set_ylabel("Inverse Sensitivity", fontsize=16)
        ax[1].set_yscale("log")
        ax[1].legend(fontsize=14)
        plt.show()

    min_wave, max_wave = cal_wave.min(), cal_wave.max()
    save_sensfunc(min_wave, max_wave, coefs)


def save_sensfunc(xmin, xmax, coefs):
    """Saves the sensitivity function coefficients.

    Parameters
    ----------
    xmin: float
        Minimum value of the data.
    xmax: float
        Maximum value of the data.
    coefs: array-like
        Coefficients for the sensitivity function.
    """
    config = dotenv_values(".env")
    PROCESSING = config["PROCESSING"]
    sensfunc_file = os.path.join(PROCESSING, "sensfunc.txt")

    with open(sensfunc_file, "w") as file:
        file.write(f"xmin xmax: {xmin} {xmax}\n")
        coefs_str = " ".join(str(coef) for coef in coefs)
        file.write(f"coefficients: {coefs_str}\n")


def load_sensfunc():
    """Loads the sensitivity function parameters.

    Returns
    -------
    xmin: float
        Minimum value of the data.
    xmax: float
        Maximum value of the data.
    coefs: array-like
        Coefficients for the sensitivity function.
    """
    config = dotenv_values(".env")
    PROCESSING = config["PROCESSING"]
    sensfunc_file = os.path.join(PROCESSING, "sensfunc.txt")

    with open(sensfunc_file, "r") as file:
        lines = file.read().splitlines()

    xmin_xmax = lines[0].split(" ")
    xmin, xmax = float(xmin_xmax[-2]), float(xmin_xmax[-1])
    coefs_line = lines[1].split(" ")
    coefs = [float(coef) for coef in coefs_line[1:]]

    return xmin, xmax, coefs


def apply_sensfunc(wavelength, raw_flux):
    """Applies the sensitivity function to an uncalibrated spectrum.

    Parameters
    ----------
    wavelength: array
        Calibrated wavelengths.
    raw_flux: array
        Uncalibrated fluxes.

    Returns
    -------
    wavelength: array
        Calibrated wavelengths (same as input).
    flux: array
        Calibrated fluxes.
    """
    min_wave, max_wave, coefs = load_sensfunc()
    log_sensfunc = np.polyval(coefs, wavelength)
    sensfunc = 10**log_sensfunc
    flux = raw_flux * sensfunc

    mask = (min_wave <= wavelength) & (wavelength <= max_wave)
    wavelength = wavelength[mask]
    flux = flux[mask]

    return wavelength, flux


def correct_tellurics(wavelength, flux):
    """Corrects for telluric absorptions.

    Parameters
    ----------
    wavelength: array
        Calibrated wavelengths.
    flux: array
        Calibrated fluxes.

    Returns
    -------
    corr_flux: array
        Corrected fluxes.
    """
    config = dotenv_values(".env")
    PROCESSING = config["PROCESSING"]
    tellurics_file = os.path.join(PROCESSING, "telluric_correction.txt")
    tell_wave, tellurics = np.loadtxt(tellurics_file).T

    tellurics = np.interp(wavelength, tell_wave, tellurics)
    corr_flux = flux / tellurics

    return corr_flux


def calibrate_spectra(use_master_arc=False, smoothing=False):
    """Calibrates all the spectra in the working directory.

    The spectra are wavelength- and flux-calibrated, including
    atmospheric and telluric corrections.

    Parameters
    ----------
    use_master_arc: bool, default ``False``
        If ``True``, the wavelength solution from the master ARC
        is used instead of the target's specific solution.
    smoothing: bool, default ``False``
        If ``True``, the spectra is smoothed with a window
        of 5 angstrom.
    """
    config = dotenv_values(".env")
    PROCESSING = config["PROCESSING"]
    files_path = os.path.join(PROCESSING, "*_1d.fits")
    science_files = glob.glob(files_path)

    for file in science_files:
        # extract raw spectrum
        hdu = fits.open(file)
        header = hdu[0].header
        raw_flux = hdu[0].data
        raw_wave = np.arange(len(raw_flux))

        target_name = hdu[0].header["OBJECT"]

        # apply wavelength and flux calibration
        if use_master_arc is True:
            wavesol_file = "wavesol.txt"
        else:
            wavesol_file = f"wavesol_{target_name}.txt"
        cal_wave = apply_wavesol(raw_wave, wavesol_file)
        cal_wave, cal_flux = apply_sensfunc(cal_wave, raw_flux)

        # telluric correction
        cal_flux = correct_tellurics(cal_wave, cal_flux)

        # apply smoothing
        if smoothing is True:
            spec = Spectrum1D(
                spectral_axis=cal_wave * u.angstrom, flux=cal_flux * u.Jy
            )  # flux units don't matter
            spec_bsmooth = box_smooth(spec, width=5)
            cal_flux = spec_bsmooth.flux.value

        # and flux to HDU data and wavelength calibration to header
        hdu[0].data = cal_flux
        header["CRVAL1"] = cal_wave.min()  # initial wavelength
        wave_diff = np.diff(cal_wave)
        header["CD1_1"] = np.mean(wave_diff)  # mean increment per pixel

        # save calibrated spectrum to fits file...
        outfile = file.replace("_1d", "_wf")
        hdu.writeto(outfile, overwrite=True)

        # ...and save it in an asci file
        outfile_asci = file.replace("_1d.fits", "_wf.txt")
        data = np.array([cal_wave, cal_flux]).T
        np.savetxt(outfile_asci, data)
