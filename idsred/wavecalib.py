# ING-IDS lamps: https://www.ing.iac.es/astronomy/instruments/ids/wavelength_calibration.html

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from dotenv import dotenv_values

from scipy.signal import find_peaks
from astropy.stats import sigma_clip
from scipy.optimize import minimize, curve_fit
from ccdproc import CCDData

from lmfit import Minimizer, Parameters

import warnings
from astropy.utils.exceptions import AstropyWarning

import idsred

idsred_path = idsred.__path__[0]
XMIN, XMAX = 1172.67, 3597.24  # ad-hoc values used for the initial solution
################
# ARC spectrum #
################


def _gaussian(x, *params):
    """Simple Gaussian function for fitting.

    Parameters
    ----------
    x: array
        Measured values.
    params: list or array-like
        Amplitude, center, standard deviation and y-axis offset
        of the Gaussian.
    """
    amp, x0, sigma, offset = params
    return amp * np.exp(-((x - x0) ** 2) / 2 / sigma**2) + offset


def _fit_gauss2peaks(arc_disp, arc_profile, peak_ids, plot_diag=False):
    """Fits Gaussian functions to the lines of an arc lamp.

    Parameters
    ----------
    arc_disp: array
        Dispersion axis of the lamp (e.g. columns in an image).
    arc_profile: array
        Profile of the lamp (e.g. intensity/amplitude in an image).
    peak_ids: array-like
        Indeces of the peaks in the lamp.
    plot_diag: bool, default ``False``
        If ``True``, a set of diagnostic plots are shown for each step
        and the final solution as well.

    Returns
    -------
    amplitudes: array
        Amplitudes of the peaks/spectral lines.
    centers: array
        Centers of the peaks/spectral lines.
    sigmas: array
        Standard deviations of the peaks/spectral lines.
    offsets: array
        Y-axis offsets.
    """
    amplitudes, centers, sigmas, offsets = [], [], [], []
    sigma = 1.0  # initial guess
    offset = 0.0
    width = 4  # width of the lines to fit
    for i in peak_ids:
        center0 = arc_disp[i]
        amplitude0 = arc_profile[i]

        guess = (amplitude0, center0, sigma, offset)
        bounds = (
            (0, center0 - 5, 0, -np.inf),
            (np.inf, center0 + 5, np.inf, np.inf),
        )

        # indices to bound the profile of each line
        i_min = int(center0 - width)
        i_max = int(center0 + width)

        try:
            popt, pcov = curve_fit(
                _gaussian,
                arc_disp[i_min:i_max],
                arc_profile[i_min:i_max],
                p0=guess,
                bounds=bounds,
            )
            amp, center, sigma, offset = popt

            # chi square
            mask = (arc_disp >= center - 3 * sigma) & (
                arc_disp <= center + 3 * sigma
            )
            y_mod = _gaussian(arc_disp[mask], *popt)
            residual = y_mod - arc_profile[mask]
            chi2 = np.sum(residual**2 / sigma**2)
            chi2_red = chi2 / (len(y_mod) - len(popt))

            if np.abs(center - center0) > 4:
                center = np.inf

            if chi2_red < 0.7:
                center = np.inf

            if plot_diag and np.isfinite(center):
                # diagnostic plot with the fit result
                x_mod = np.linspace(
                    center - 3 * sigma, center + 3 * sigma, 1000
                )
                y_mod = _gaussian(x_mod, *popt)
                mask = (arc_disp >= x_mod.min()) & (arc_disp <= x_mod.max())

                fig, ax = plt.subplots(figsize=(8, 6))

                ax.plot(x_mod, y_mod, color="r", lw=2, label="Gaussian fit")
                ax.scatter(
                    center, amp + offset, color="r", marker="*", s=60
                )  # peak
                ax.axvline(x_mod[np.argmax(y_mod)], color="r", ls="--")

                ax.plot(arc_disp[mask], arc_profile[mask], color="k", lw=2)
                ax.scatter(
                    center0, amplitude0, color="k", marker="*", s=60
                )  # peak
                ax.axvline(center0, color="k", ls="--")

                ax.set_ylabel("Intensity", fontsize=16)
                ax.set_xlabel("Dispersion axis (pixels)", fontsize=16)
                ax.legend()
                plt.show()
        except RuntimeError:
            # curve_fit failed to converge...skip
            continue

        amplitudes.append(amp)
        centers.append(center)
        sigmas.append(sigma)
        offsets.append(offset)

    amplitudes = np.array(amplitudes)
    centers = np.array(centers)
    sigmas = np.array(sigmas)
    offsets = np.array(offsets)

    # filter lamp_lines to keep only lines that were fit
    fit_mask = np.isfinite(centers)
    amplitudes = amplitudes[fit_mask]
    centers = centers[fit_mask]
    sigmas = sigmas[fit_mask]
    offsets = offsets[fit_mask]

    return amplitudes, centers, sigmas, offsets


def find_arc_peaks(data, plot_solution=False, plot_diag=False):
    """Finds the centers of the emission lines of an arc lamp.

    Gaussian functions are fit to the arc lamp lines.

    Parameters
    ----------
    data: `~astropy.nddata.CCDData`-like, array-like
        Image data.
    plot_solution: bool, default ``False``
        If ``True``, the lamp with the solution is plotted.
    plot_diag: bool, default ``False``
        If ``True``, a set of diagnostic plots are shown for each step
        and the final solution as well.

    Returns
    -------
    arc_pixels: array
        Centers of the peaks in pixels.
    arc_peaks: array
        Peaks intensity:
    arc_sigmas: array
        Standard deviations of the Gaussian fits.
    """
    ny, nx = data.shape
    cy, cx = ny // 2, nx // 2

    arc_disp = np.arange(nx)
    arc_profile = data[cy][::-1]  # the axis is inverted
    arc_profile -= arc_profile.min()

    # initial peak estimation
    prominence = 100  # minimum line intensity
    peak_ids = find_peaks(arc_profile, prominence=prominence)[0]

    # peak estimation with gaussian fitting
    arc_peaks, arc_pixels, arc_sigmas, offsets = _fit_gauss2peaks(
        arc_disp, arc_profile, peak_ids, plot_diag
    )

    # saturation mask / maximum line intensity
    sat_mask = arc_peaks < 64000
    arc_pixels = arc_pixels[sat_mask]
    arc_peaks = arc_peaks[sat_mask]
    offsets = offsets[sat_mask]

    if plot_solution:
        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(arc_profile)
        ax.scatter(
            arc_disp[peak_ids],
            arc_profile[peak_ids],
            marker="*",
            color="g",
            label="Initial Peaks",
        )
        ax.scatter(
            arc_pixels,
            arc_peaks + offsets,
            marker="*",
            color="r",
            label="Optimised Peaks (Gaussian Fit)",
        )

        ax.set_ylabel("Intensity", fontsize=16)
        ax.set_xlabel("Dispersion axis (pixels)", fontsize=16)
        ax.legend()
        plt.show()

        zoom_in_plots = False
        if zoom_in_plots is True:
            cut = 1840  # this value can change quite a bit
            xmin = 1000
            xmax = 3750

            blue_profile = arc_profile[xmin:cut]
            blue_columns = np.arange(xmin, cut)
            red_profile = arc_profile[cut:xmax]
            red_columns = np.arange(cut, xmax)

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(blue_columns, blue_profile)
            ax.set_ylabel("Intensity", fontsize=16)
            ax.set_xlabel("Dispersion axis (pixels)", fontsize=16)
            plt.show()

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(red_columns, red_profile)
            ax.set_ylabel("Intensity", fontsize=16)
            ax.set_xlabel("Dispersion axis (pixels)", fontsize=16)
            plt.show()

    return arc_pixels, arc_peaks, arc_sigmas


#######################
# Wavelength Solution #
#######################


def _find_nearest(array1, array2):
    """Finds the nearest values between two arrays.

    Finds the values in ``array2`` nearest to ``array1``,
    without repetition.
    **NOTE:* The arrays are assumed to be sorted. In addition,
    each input array is assumed to not have repeated values.

    Parameters
    ----------
    array1: array
        First array.
    array2: array
        Second array.

    Returns
    -------
    ids1: list
        Indices of the values in ``array1``.
    ids2: list
        Indices of the values in ``array2``.
    """
    ids1, ids2 = [], []
    if len(array1) < len(array2):
        short_array = array1
        long_array = array2
    else:
        long_array = array1
        short_array = array2

    i_min = 0
    for i, val in enumerate(short_array):
        imin_temp = np.argmin(np.abs(val - long_array[i_min:]))
        i_min = np.where(long_array == long_array[i_min:][imin_temp])[0][0]
        if len(array1) < len(array2):
            ids1.append(i)
            ids2.append(i_min)
        else:
            ids1.append(i_min)
            ids2.append(i)

    # TEMP: any order
    # """
    ids1, ids2 = [], []
    for i, val in enumerate(short_array):
        imin = np.argmin(np.abs(val - long_array))
        if len(array1) < len(array2):
            ids1.append(i)
            ids2.append(imin)
        else:
            ids1.append(imin)
            ids2.append(i)
    # """

    return ids1, ids2


def _norm_xaxis(xdata, xmin=None, xmax=None):
    """Normalises data to be between -1 and 1.

    Parameters
    ----------
    xdata: array
        Data to normalise
    xmin: float, default ``None``
        Minimum value of the data.
    xmax: float, default ``None``
        Maximum value of the data.

    Returns
    -------
    xnorm: array
        Normalised values.
    """
    if (xmin is None) or (xmax is None):
        xmin, xmax = xdata.min(), xdata.max()
    xnorm = (2 * xdata - (xmax + xmin)) / (xmax - xmin)

    return xnorm


def wavelength_function(params, x, func="legendre", xmin=None, xmax=None):
    """Function to fit for the wavelength solution.

    Parameters
    ----------
    params: array-like
        Whatever parameters the function accepts.
    x: array
        x-coordinate values.
    func: str, default ``legendre``
        Function to use: ``legendre`` or ``chebyshev``
    xmin: float, default ``None``
        Minimum value in the x-coordinate for normalising purposes.
    xmax: float, default ``None``
        Maximum value in the x-coordinate for normalising purposes.

    Returns
    -------
    y_model: array
        Model evaluated at ``x``.
    """
    xnorm = _norm_xaxis(np.copy(x), xmin, xmax)
    if func == "legendre":
        y_model = np.polynomial.legendre.legval(xnorm, params)
    elif func == "chebyshev":
        y_model = np.polynomial.chebyshev.chebval(xnorm, params)
    else:
        raise ValueError("Not a valid function.")

    return y_model


# Quick solution


def _chi_sq(
    params,
    arc_pixels,
    lamp_wave,
    func="legendre",
):
    """Chi squared for the wavelength solution.

    Parameters
    ----------
    params: array-like
        Parameters for the wavelength-solution function.
    arc_pixels: array
        Dispersion axis in pixels.
    lamp_wave: array
        Wavelengths of a lamp.
    func: str, default ``legendre``
        Function to use: ``legendre`` or ``chebyshev``

    Returns
    -------
    chi: float
        Chi squared value.
    """
    global XMIN, XMAX
    xmin, xmax = XMIN, XMAX
    model_wave = wavelength_function(params, arc_pixels, func, xmin, xmax)
    ids_lamp, ids_model = _find_nearest(lamp_wave, model_wave)
    residual = model_wave[ids_model] - lamp_wave[ids_lamp]

    chi = np.sum(residual**2)

    # check if the function monotonically increases
    edges_pixels = np.array([0, 1, 4999, 5000])
    check_wave = wavelength_function(params, edges_pixels, func, xmin, xmax)
    if (check_wave[0] > check_wave[1]) or (check_wave[-2] > check_wave[-1]):
        # blow up the residual
        return 1e6

    return chi


def _prepare_params(params):
    """Prepares the parameters to be used with the fitter.

    Parameters
    ----------
    params: array-like
        Parameters for the wavelength solution function.

    Returns
    -------
    parameters: ``~lmfit.Parameters``
        Parameters with bounds.
    """
    parameters = Parameters()
    for i, value in enumerate(params):
        if i == 0:
            min_val, max_val = 6000, 7500
        elif i == 1:
            min_val, max_val = 1800, 2600
        else:
            min_val, max_val = -50, 50
        parameters.add(f"c{i}", value=value, min=min_val, max=max_val)

    return parameters


def quick_wavelength_solution(
    arc_pixels,
    lamp_wave,
    func="legendre",
    k=3,
    params=None,
    niter=3,
    sigclip=3,
    plot_solution=False,
    data=None,
    sol_pixels=None,
    sol_waves=None,
    outfile="wavesol.txt",
):
    """Finds a wavelength solution with a simple fit.

    Parameters
    ----------
    arc_pixels: array
        Dispersion axis in pixels.
    lamp_wave: array
        Wavelengths of a lamp.
    func: str, default ``legendre``
        Function used to fit the wavelength solution.
        Either ``chebyshev`` or ``legendre``.
    k: int, default ``3``
        Degree of the polynomial.
    params: array-like, default ``None``
        Initial guess for the parameters for the
        wavelength-solution function.
    niter: int, default ``3``
        Number of iteration for the fit. Values are sigma clipped.
    sigclip: float, default ``3``
        Threshold for the sigma clipping.
    plot_solution: bool, default ``False``
        If ``True``, the solution is plotted.
    data: `~astropy.nddata.CCDData`-like, array-like
        Image data for plotting purposes only.
    sol_pixels: array, default ``None``
        Center of the emission lines in pixel units for an initial
        wavelength solution. If ``None``, a precomputed solution is used.
    sol_waves: array, default ``None``
        Center of the emission lines in wavelength units for an initial
        wavelength solution. If ``None``, a precomputed solution is used.
    outfile: str, default ``wavesol.txt``
        Name of the file where the wavelength solution is saved.
    """
    if params is None:
        params = [6848.4117, 2267.754] + (k - 1) * [0]
    method = "nelder"  # for minimisation with lmfit

    arc_pixels0 = np.copy(arc_pixels)
    # xmin, xmax = arc_pixels0.min(), arc_pixels0.max()
    global XMIN, XMAX
    xmin, xmax = XMIN, XMAX

    if sol_pixels is None or sol_waves is None:
        # use initial solution provided by the pipeline
        wavesol_file = os.path.join(idsred_path, "lamps", "init_wavesol.txt")
        sol_pixels, sol_waves = np.loadtxt(wavesol_file).T
        # else use manual identification provided by the user

    init_pixels = np.zeros_like(sol_pixels)
    for i, pix in enumerate(sol_pixels):
        ids_lamp, _ = _find_nearest(arc_pixels, np.array([pix]))
        init_pixels[i] = arc_pixels[ids_lamp]

    parameters = _prepare_params(params)
    fitter = Minimizer(
        _chi_sq,
        parameters,
        fcn_args=(init_pixels, sol_waves, func),
    )
    result = fitter.minimize(method=method)
    params = [result.params[key].value for key in result.params]

    # iterate fit with sigma clipping
    if niter > 0:
        for _ in range(niter):
            parameters = _prepare_params(params)
            fitter = Minimizer(
                _chi_sq,
                parameters,
                fcn_args=(arc_pixels0, lamp_wave, func),
            )
            result = fitter.minimize(method=method)
            params = [result.params[key].value for key in result.params]

            calibrated_wave = wavelength_function(
                params, arc_pixels0, func, xmin, xmax
            )
            ids_lamp, ids_calwave = _find_nearest(lamp_wave, calibrated_wave)
            residuals = calibrated_wave[ids_calwave] - lamp_wave[ids_lamp]

            # outliers removal
            mask = ~sigma_clip(residuals, sigma=sigclip).mask
            arc_pixels0 = arc_pixels0[ids_calwave][mask]

    parameters = _prepare_params(params)
    fitter = Minimizer(
        _chi_sq,
        parameters,
        fcn_args=(arc_pixels0, lamp_wave, func),
    )
    result = fitter.minimize(method=method)
    params = [result.params[key].value for key in result.params]

    outliers_mask = np.array(
        [False if pixel in arc_pixels0 else True for pixel in arc_pixels]
    )
    if plot_solution:
        check_solution(
            params, arc_pixels, lamp_wave, outliers_mask, data, func
        )

    save_wavesol(func, xmin, xmax, params, outfile)


# Checking output


def check_solution(
    params, arc_pixels, lamp_wave, mask=None, data=None, func="legendre"
):
    """Shows the residuals of the wavelength solution.

    Parameters
    ----------
    params: array-like
        Parameters for the wavelength-solution function.
    arc_pixels: array
        Dispersion axis in pixels.
    lamp_wave: array
        Wavelengths of a lamp.
    mask: bool array, default ``None``
        Mask of outliers.
    data: `~astropy.nddata.CCDData`-like, array-like, default ``None``.
        Image data for plotting purposes only.
    func: str, default ``legendre``
        Function used to fit the wavelength solution.
        Either ``chebyshev`` or ``legendre``.
    """
    global XMIN, XMAX
    xmin, xmax = XMIN, XMAX

    calibrated_wave = wavelength_function(params, arc_pixels, func, xmin, xmax)
    ids_lamp, ids_calwave = _find_nearest(lamp_wave, calibrated_wave)

    residuals = calibrated_wave[ids_calwave] - lamp_wave[ids_lamp]
    n_all = len(calibrated_wave[ids_calwave])
    mean, std = residuals.mean(), residuals.std()

    if data is not None:
        ny, nx = data.shape
        cy, cx = ny // 2, nx // 2
        arc_disp = np.arange(nx)
        arc_profile = data[cy][::-1]
        arc_wave = wavelength_function(params, arc_disp, func, xmin, xmax)

        # plot ARC lines
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(16, 8),
            sharex=True,
            height_ratios=[3, 4, 2],
            gridspec_kw=dict(hspace=0),
        )

        axes[0].plot(arc_wave, arc_profile)
        axes[0].set_ylabel("Intensity", fontsize=16)
        if mask is not None:
            # mark outliers with dotted lines
            waves = calibrated_wave[~mask]
            for wave in calibrated_wave[mask]:
                axes[0].axvline(wave, ls="dotted", alpha=0.2, color="k")
        else:
            waves = calibrated_wave
        for wave in waves:
            axes[0].axvline(wave, ls="--", alpha=0.2, color="k")
        i = 1
    else:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(12, 8),
            sharex=True,
            height_ratios=[4, 2],
            gridspec_kw=dict(hspace=0),
        )
        i = 0
    # plot pixels vs wavelengths
    axes[i].scatter(
        calibrated_wave[ids_calwave],
        arc_pixels[ids_calwave],
        marker="*",
        c="r",
    )

    # fit
    arc_model = np.arange(
        arc_pixels[ids_calwave].min(), arc_pixels[ids_calwave].max(), 10
    )
    wave_model = wavelength_function(params, arc_model, func, xmin, xmax)
    axes[i].plot(
        wave_model,
        arc_model,
        lw=2,
        c="g",
        label=f"fit ({func} k={len(params) - 1})",
        zorder=0,
    )

    # plot wavelength solution residuals
    axes[i + 1].scatter(
        calibrated_wave[ids_calwave], residuals, marker="*", c="r"
    )

    if mask is not None:
        masked_calibrated_wave = wavelength_function(
            params, arc_pixels[~mask], func, xmin, xmax
        )
        ids_lamp, ids_calwave = _find_nearest(
            lamp_wave, masked_calibrated_wave
        )
        masked_res = masked_calibrated_wave[ids_calwave] - lamp_wave[ids_lamp]
        mean, std = masked_res.mean(), masked_res.std()

        masked_calibrated_wave = wavelength_function(
            params, arc_pixels[mask], func, xmin, xmax
        )
        ids_lamp, ids_calwave = _find_nearest(
            lamp_wave, masked_calibrated_wave
        )
        masked_res = masked_calibrated_wave[ids_calwave] - lamp_wave[ids_lamp]
        n_out = len(calibrated_wave[ids_calwave])

        # plot outliers
        axes[i].scatter(
            masked_calibrated_wave[ids_calwave],
            arc_pixels[mask][ids_calwave],
            marker="x",
            c="b",
            label=f"outliers ({n_out}/{n_all})",
        )
        axes[i].legend(fontsize=14)
        axes[i + 1].scatter(
            masked_calibrated_wave[ids_calwave], masked_res, marker="x", c="b"
        )

    axes[i + 1].axhline(mean, c="k")
    axes[i + 1].axhline(mean + std, c="k", ls="--")
    axes[i + 1].axhline(mean - std, c="k", ls="--")
    axes[i].set_ylabel(r"Dispersion axis (pixels)", fontsize=16)
    axes[i + 1].set_ylabel(r"Residual ($\AA$)", fontsize=16)
    axes[i + 1].set_xlabel(r"Wavelength ($\AA$)", fontsize=16)
    axes[i + 1].set_ylim(mean - 3 * std, mean + 3 * std)

    axes[0].set_title(
        f"Residual: {mean:.2f} $+/-$ {std:.2f}" + r" $\AA$", fontsize=16
    )
    plt.show()


def find_wavesol(
    func="legendre",
    coefs=None,
    k=5,
    niter=5,
    sigclip=2.5,
    plot_solution=False,
    sol_pixels=None,
    sol_waves=None,
    extract_individual_solutions=False
):
    """Finds the wavelength solution.

    The master ARC file is used for this.

    Parameters
    ----------
    func: str, default ``legendre``
        Function used to fit the wavelength solution.
        Either ``chebyshev`` or ``legendre``.
    coefs: array-like, default ``None``
        Initial guess for the parameters for the
        wavelength-solution function. Tuple, e.g. (4500, 0.5, 0, 0).
        If this is given, it overwrites the degrees of the polynomial.
    k: int, default ``5``
        Degree of the polynomial.
    niter: int, default ``5``
        Number of iteration for the fit. Values are sigma clipped.
    sigclip: float, default ``2.5``
        Threshold for the sigma clipping.
    plot_solution: bool, default ``False``
        If ``True``, the solution is plotted.
    sol_pixels: array-like, default ``None``
        Center of the emission lines in pixel units for an initial
        wavelength solution. If ``None``, a precomputed solution is used.
    sol_waves: array-like, default ``None``
        Center of the emission lines in wavelength units for an initial
        wavelength solution. If ``None``, a precomputed solution is used.
    extract_individual_solutions: bool, default ``False``
        If ``True``, the solutions using the arcs for each target are extracted.
    """
    # load master ARC file
    config = dotenv_values(".env")
    PROCESSING = config["PROCESSING"]
    arc_file = os.path.join(PROCESSING, "master_arc.fits")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        master_arc = CCDData.read(arc_file)

    # fit and extract peak of the arc lamps
    data = master_arc.data.T
    arc_pixels, arc_peaks, arc_sigmas = find_arc_peaks(
        data, plot_solution=True, plot_diag=False
    )
    start, step = 0, 1
    arc_pixels, arc_peaks, arc_sigmas = (
        arc_pixels[start::step],
        arc_peaks[start::step],
        arc_sigmas[start::step],
    )

    global idsred_path
    lamp_file = os.path.join(idsred_path, "lamps/CuArNe_low.dat")
    lamp_wave = np.loadtxt(lamp_file).T
    print("Finding the wavelength solution for the master ARC...")
    quick_wavelength_solution(
        arc_pixels,
        lamp_wave,
        func=func,
        k=k,
        params=coefs,
        niter=niter,
        sigclip=sigclip,
        plot_solution=plot_solution,
        data=data,
        sol_pixels=sol_pixels,
        sol_waves=sol_waves,
        outfile="wavesol.txt",
    )

    if extract_individual_solutions is True:
        # repeat above steps for each target's arc
        arc_files = glob.glob(os.path.join(PROCESSING, "arc_*.fits"))
        for arc_file in arc_files:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", AstropyWarning)
                arc_ccd = CCDData.read(arc_file)

            arc_name = arc_ccd.header["OBJECT"]
            target_name = arc_name.split("ARC_")[1]

            # fit and extract peak of the arc lamps
            data = arc_ccd.data.T
            arc_pixels, arc_peaks, arc_sigmas = find_arc_peaks(
                data, plot_solution=True, plot_diag=False
            )
            arc_pixels, arc_peaks, arc_sigmas = (
                arc_pixels[start::step],
                arc_peaks[start::step],
                arc_sigmas[start::step],
            )

            print(f"Finding the wavelength solution for {target_name}'s ARC...")
            quick_wavelength_solution(
                arc_pixels,
                lamp_wave,
                func=func,
                k=k,
                params=coefs,
                niter=niter,
                sigclip=sigclip,
                plot_solution=plot_solution,
                data=data,
                sol_pixels=sol_pixels,
                sol_waves=sol_waves,
                outfile=f"wavesol_{target_name}.txt",
            )


def save_wavesol(func, xmin, xmax, coefs, outfile):
    """Saves the output of the wavelength solution.

    Parameters
    ----------
    func: str
        Function used to fit the wavelength solution.
        Either ``chebyshev`` or ``legendre``.
    xmin: float
        Minimum value of the data.
    xmax: float
        Maximum value of the data.
    coefs: array-like
        Initial guess for the parameters for the
        wavelength-solution function. Tuple, e.g. (4500, 0.5, 0, 0).
    outfile: str
        Name of the output file.
    """
    config = dotenv_values(".env")
    PROCESSING = config["PROCESSING"]
    wavesol_file = os.path.join(PROCESSING, outfile)

    with open(wavesol_file, "w") as file:
        file.write(f"function: {func}\n")
        file.write(f"xmin xmax: {xmin} {xmax}\n")
        coefs_str = " ".join(str(coef) for coef in coefs)
        file.write(f"coefficients: {coefs_str}\n")


def load_wavesol(inputfile):
    """Loads the wavelength solution.

    The solution has to be computed first.

    Parameters
    ----------
    inputfile: str
        Name of the file with the wavelength solution
    """
    config = dotenv_values(".env")
    PROCESSING = config["PROCESSING"]
    wavesol_file = os.path.join(PROCESSING, inputfile)

    with open(wavesol_file, "r") as file:
        lines = file.read().splitlines()

    func = lines[0].split(" ")[-1]
    xmin_xmax = lines[1].split(" ")
    xmin, xmax = float(xmin_xmax[-2]), float(xmin_xmax[-1])
    coefs_line = lines[2].split(" ")
    coefs = [float(coef) for coef in coefs_line[1:]]

    return func, xmin, xmax, coefs


def apply_wavesol(xdata, wavesol_file):
    """Applies the wavelength solution to an array.

    Parameters
    ----------
    xdata: array
        Values in pixel units.
    wavesol_file: str
        File with the wavelength solution

    Returns
    -------
    wavelengths: array
        Calibrated wavelengths.
    """
    func, xmin, xmax, coefs = load_wavesol(wavesol_file)
    wavelengths = wavelength_function(coefs, xdata, func, xmin, xmax)

    return wavelengths
