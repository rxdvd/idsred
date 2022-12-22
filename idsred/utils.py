import numpy as np
import matplotlib.pyplot as plt
from dotenv import dotenv_values

from ccdproc import CCDData
from ccdproc import ImageFileCollection

import warnings
from astropy.utils.exceptions import AstropyWarning


def collect_data(path=None):

    if path is None:
        config = dotenv_values(".env")
        path = config['WORKDIR']

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        observations = ImageFileCollection(path)

    return observations


def update_header(hdu, new_header):
    header = hdu[0].header
    skip_keywords = ['SIMPLE', 'BITPIX', 'NAXIS',
                     'NAXIS1', 'NAXIS2', 'EXTEND']

    for content in new_header.cards:
        keyword = content[0]
        if keyword not in skip_keywords:
            header.append(content)


def plot_image(hdu):
    """Plots a 2D image.
    
    Parameters
    ----------
    Parameters
    ----------
    hdu: ~fits.hdu
        Header Data Unit.
        
    Returns
    -------
    ax: `~.axes.Axes`
        Plot axis.
    """
    data = hdu[0].data
    header = hdu[0].header
    if data is None:
        data = hdu[1].data

    m, s = np.nanmean(data), np.nanstd(data)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(data, interpolation='nearest',
               cmap='gray',
               vmin=m-s, vmax=m+s,
               origin='lower')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(header['OBJECT'], fontsize=16)
    return ax

def obs_plots(observations, obstype):
    """Plots all images of a given ``obstype``.
    
    Parameters
    ----------
    observations: `~ImageFileCollection`
        Table-like object with images information.
    obstype: str
        Type of Image. E.g. ``BIAS``, ``FLAT``.
    """    
    for filename in observations.files_filtered(include_path=True, obstype=obstype):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyWarning)
            ccd = CCDData.read(filename, hdu=1, unit=u.electron)
            plot_image(ccd.data.T)


def fits2array(hdu):
    """Extracts a FITS file data into arrays.

    This function works for spectra only, not imaging, and
    returns the wavelength calibrated spectrum.

    Parameters
    ----------
    hdu: ~fits.hdu
        Header Data Unit.

    Returns
    -------
    wave: array
        Wavelengths of a spectrum.
    flux: array
        Flux density of a spectrum.
    bkg: array
        Background of a spectrum.
    err: array
        Flux density error of a spectrum.
    """
    flux = hdu[0].data
    header = hdu[0].header

    # wavelength array calculation
    start_wave = header['CRVAL1']  # initial wavelength
    step = header['CD1_1']  # increment per pixel

    w0, n = start_wave, len(flux)
    w = start_wave + step * n
    wave = np.linspace(w0, w, n, endpoint=False)

    return wave, flux