import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from dotenv import dotenv_values

from ccdproc import CCDData
from astropy.io import fits
from scipy.optimize import minimize
from scipy.signal import find_peaks
from astropy.stats import sigma_clip
from astropy.convolution import (
    Box2DKernel,
    Gaussian2DKernel,
    convolve_fft,
    interpolate_replace_nans,
)

from .utils import plot_image

import warnings
from astropy.utils.exceptions import AstropyWarning


def quick_trace(
    hdu, center=None, width=50, gap=20, sky_width=40, plot_trace=False
):
    """Extracts a "raw" spectrum in a quick way.

    The trace is background subtracted.

    Parameters
    ----------
    hdu: Header Data Unit
        HDU 2D image.
    center: float or None, optional
        Center of the trace. If not give, one is obtained with ``find_peaks``,
        using the peak with the largest amplitud.
    width: float, default ``50``
        Width of the trace in pixels.
    gap: float, default ``20``
        Separation between the trace and sky in pixels.
    sky_width: float, default ``40``
        Width of the sky in pixels. Used for background subtraction.
    plot_trace: bool, default ``False``
        If ``True``, the image is plotted with the trace. The raw spectrum is also plotted.

    Returns
    -------
    raw_spectrum: array
        Raw spectrum of the image.
    """
    data = hdu[0].data
    header = hdu[0].header

    if center is None:
        ny, nx = data.shape
        center0 = ny // 2
        peaks = find_peaks(
            data[:, nx // 2].data,
            height=np.nanmedian(data[:, nx // 2]),
            width=3,
        )[0]
        if len(peaks) == 0:
            print("Peak not found to guess the trace centre")
            center = center0
        else:
            peak_id = np.argmax(data[:, nx // 2][peaks])
            center = peaks[peak_id]

    # sky on one side
    imin_sky1 = int(center - (width // 2 + gap + sky_width))
    imax_sky1 = int(center - (width // 2 + gap))
    sky1 = np.nanmean(data[imin_sky1:imax_sky1], axis=0)

    # sky on the other side
    imin_sky2 = int(center + (width // 2 + gap))
    imax_sky2 = int(center + (width // 2 + gap + sky_width))
    sky2 = np.nanmean(data[imin_sky2:imax_sky2], axis=0)

    # pseudo-interpolation of the sky by taking the mean
    sky = (sky1 + sky2) / 2

    # sum flux in aperture and subtract sky
    imin = int(center - width // 2)
    imax = int(center + width // 2)
    raw_spectrum = np.nansum(data[imin:imax] - sky, axis=0)

    # invert axis and convert masked array into array
    raw_spectrum = raw_spectrum[::-1]

    if plot_trace:
        for i in range(2):
            if i == 1:
                data = data[:, 1700:2300]

            ax = plot_image(data)
            ax.set_title(header["OBJECT"], fontsize=16)
            ax.axhline(imin, c="r", lw=2, label="aperture")
            ax.axhline(imax, c="r", lw=2)
            ax.axhspan(imin_sky1, imax_sky1, color="g", alpha=0.4, label="sky")
            ax.axhspan(imin_sky2, imax_sky2, color="g", alpha=0.4)
        ax.legend(fontsize=16)
        plt.show()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(raw_spectrum)
        ax.set_xlabel("Dispersion axis (pixels)", fontsize=16)
        ax.set_ylabel("Raw Flux", fontsize=16)
        plt.show()

    return raw_spectrum


def _get_profile_model(params, ys):
    """Gaussian function with offset for fitting.

    Parameters
    ----------
    params: list or array-like
        Amplitude, center, standard deviation and y-axis offset
        of the Gaussian.
    ys: array
        Measured values.
    """
    amplitude, center, sigma, yoffset = params

    profile = np.exp(-((ys - center) ** 2) / 2 / sigma**2)
    profile /= np.nanmax(profile)
    profile *= amplitude
    profile += yoffset

    return profile


def _get_profile_chisq(params, ys, profile):
    """Reduced chi-squared for fitting.

    Parameters
    ----------
    params: list or array-like
        Amplitude, center, standard deviation and y-axis offset
        of the Gaussian.
    ys: array
        Measured values.
    profile: array
        Profile of the observed Gaussian.
    """
    model = _get_profile_model(params, ys)

    return np.sum((profile - model) ** 2 / (profile.size - len(params)))


def optimised_trace(
    hdu,
    center=None,
    amp=None,
    hwidth=50,
    t_order=3,
    plot_diag=False,
    plot_trace=False,
):
    """Extracts a "raw" spectrum in an optimised way.

    The trace is background subtracted. Sigma clipping is used for removing "untrusted" fits.
    The sky width is fixed.

    Parameters
    ----------
    hdu: Header Data Unit
        HDU 2D image.
    center: float or None, optional
        Initial guess of the trace center. If not give, one is obtained with ``find_peaks``,
        using the peak with the largest amplitud.
    amp: float or None, optional
        Initial guess of the trace amplitude. If not give, one is obtained with ``find_peaks``.
    hwidth: float, default ``50``
        Number of pixels to used for each bin in the dispersion axis.
    t_order: int, default ``3``
        Order of the polynomial used for fitting the trace.
    plot_diag: bool, default ``False``
        If ``True``, a set of diagnostic plots are shown for each step and the final solution as well.
    plot_trace: bool, default ``False``
        If ``True``, the image is plotted with the trace. The raw spectrum is also plotted.

    Returns
    -------
    raw_spectrum: array
        Raw spectrum of the image.
    """
    data = hdu[0].data
    header = hdu[0].header

    ny, nx = data.shape
    xs = np.arange(nx)
    ys = np.arange(ny)

    cols = np.arange(hwidth, nx + 1, 2 * hwidth)
    ycenter = np.zeros(len(cols))
    ywidth = np.zeros(len(cols))
    bkg_width = np.zeros(len(cols))

    for icol, col in enumerate(cols):
        if col < 500 or col > 3500:
            # avoid edges as there is no signal
            ycenter[icol] = np.inf
            ywidth[icol] = np.inf
            bkg_width[icol] = np.inf
            continue

        stamp = data[:, col - hwidth : col + hwidth]
        profile = np.nanmean(stamp, axis=1)

        if center is None or amp is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                peaks = find_peaks(
                    profile, height=np.nanmedian(profile), prominence=10
                )[0]
            if len(peaks) > 0:
                if amp is None:
                    amp = np.max(profile[peaks])
                if center is None:
                    peak_id = np.argmax(profile[peaks])
                    center = peaks[peak_id]
            else:
                if amp is None:
                    amp = 10
                if center is None:
                    center = len(profile) // 2

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            guess = (amp, center, 5, np.nanmedian(profile))

        results = minimize(_get_profile_chisq, guess, args=(ys, profile))
        params = results.x
        if params[2] < 20:
            ycenter[icol] = params[1]
            ywidth[icol] = 4 * params[2]  # aperture width in sigmas units
            bkg_width[icol] = 8 * params[2]  # beginning of bkg in sigma units
            model = _get_profile_model(params, ys)

            # diagnostic plots for each step
            if plot_diag:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(ys, profile, label="data")
                ax.plot(ys, model, label="model")
                ax.axvline(
                    ycenter[icol] + ywidth[icol],
                    c="r",
                    ls="dotted",
                    label="aperture",
                )
                ax.axvline(ycenter[icol] - ywidth[icol], c="r", ls="dotted")
                ax.axvline(
                    ycenter[icol] + bkg_width[icol],
                    c="g",
                    ls="dotted",
                    label="background",
                )
                ax.axvline(ycenter[icol] - bkg_width[icol], c="g", ls="dotted")
                ax.set_xlabel("Dispersion axis (pixels)", fontsize=16)
                ax.set_ylabel("Median Counts", fontsize=16)
                ax.legend()
                plt.grid()
                plt.show()
        else:
            ycenter[icol] = np.inf
            ywidth[icol] = np.inf

    # remove bad fits
    mask = np.isfinite(ycenter)
    ycenter = ycenter[mask]
    ywidth = ywidth[mask]
    bkg_width = bkg_width[mask]
    cols = cols[mask]

    # remove untrusted fits with sigma clipping
    mask = ~sigma_clip(ycenter, sigma=2.5, maxiters=10).mask
    ycenter = ycenter[mask]
    ywidth = ywidth[mask]
    bkg_width = bkg_width[mask]
    cols = cols[mask]

    trace_coef = np.polyfit(cols, ycenter, t_order)
    trace = np.polyval(trace_coef, xs)

    # trace aperture + background
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        trace_top = trace + np.median(ywidth)
        trace_bottom = trace - np.median(ywidth)
        bkg_top = trace + np.median(bkg_width)
        bkg_bottom = trace - np.median(bkg_width)
        sky_width = 25

    # final diagnostic plots
    if plot_diag:
        # spline fit
        fig, ax = plt.subplots(2, figsize=(14, 4), sharex=True)
        ax[0].plot(cols, ycenter, "ro", label="data")
        ax[0].plot(xs, trace, "r", label="spline")
        ax[0].plot(xs, trace_top, "r", ls="--", label="aperture")
        ax[0].plot(xs, trace_bottom, "r", ls="--")
        ax[0].plot(xs, bkg_top, "g", ls="--", label="background")
        ax[0].plot(xs, bkg_bottom, "g", ls="--")
        ax[0].set_title("Trace", fontsize=16)
        ax[0].axes.set_ylabel("y-coordinate", fontsize=16)
        ax[0].legend()
        ax[0].grid()

        # residuals
        trace_col = np.polyval(trace_coef, cols)
        ax[1].plot(cols, ycenter - trace_col, "ro")
        ax[1].axhline(0.0, c="k")
        ax[1].axes.set_ylabel("Fit Residual (pixels)", fontsize=16)
        ax[1].set_xlabel("Dispersion axis", fontsize=16)
        ax[1].grid()
        plt.show()

    if plot_trace:
        ymax, xmax = data.shape
        ymin, xmin = 0, 0

        # plot full trace
        ax = plot_image(hdu)
        ax.plot(xs, trace_top, c="r", lw=1)
        ax.plot(xs, trace_bottom, c="r")
        ax.fill_between(xs, bkg_top, bkg_top + sky_width, color="g", alpha=0.5)
        ax.fill_between(
            xs, bkg_bottom, bkg_bottom - sky_width, color="g", alpha=0.5
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        plt.show()

        # zoom-in trace + slice
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        # zoom-in first
        m, s = np.nanmean(data), np.nanstd(data)
        xmin, xmax = xmax // 2 - 300, xmax // 2 + 300

        im = axes[0].imshow(
            data,
            interpolation="nearest",
            cmap="gray",
            vmin=m - s,
            vmax=m + s,
            origin="lower",
        )
        fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
        axes[0].set_title(header["OBJECT"], fontsize=16)
        axes[0].plot(xs, trace_top, c="r", lw=1, label="aperture")
        axes[0].plot(xs, trace_bottom, c="r")
        axes[0].fill_between(
            xs,
            bkg_top,
            bkg_top + sky_width,
            color="g",
            alpha=0.5,
            label="background",
        )
        axes[0].fill_between(
            xs, bkg_bottom, bkg_bottom - sky_width, color="g", alpha=0.5
        )
        axes[0].set_xlim(xmin, xmax)
        axes[0].set_ylim(ymin, ymax)
        axes[0].legend(fontsize=16)

        # slice plot
        index = (xmin + xmax) // 2
        slice = data[:, index]
        cols = np.arange(ymax)
        axes[1].plot(cols, slice, c="k", lw=2)  # trace profile
        axes[1].axvline(trace_top[index], c="r", lw=1)
        axes[1].axvline(trace_bottom[index], c="r", lw=1)

        y_bkg = np.arange(slice.min() * 0.9, slice.max() * 1.1)
        axes[1].fill_betweenx(
            y_bkg,
            bkg_top[index],
            bkg_top[index] + sky_width,
            color="g",
            alpha=0.5,
        )
        axes[1].fill_betweenx(
            y_bkg,
            bkg_bottom[index],
            bkg_bottom[index] - sky_width,
            color="g",
            alpha=0.5,
        )
        axes[1].set_xlim(
            bkg_bottom[index] - sky_width - 30, bkg_top[index] + sky_width + 30
        )
        axes[1].set_ylim(None, slice.max() * 1.1)

        # bkg = (np.nanmedian(slice[int(bkg_top[index]):int(bkg_top[index]) + sky_width]) +
        #       np.nanmedian(slice[int(bkg_bottom[index]) - sky_width:int(bkg_top[index])] ) )/2
        # axes[2].plot(cols, slice-bkg, c="k", lw=2)

        plt.show()

    # flux in trace aperture
    raw_spectrum = np.zeros_like(trace)
    for i in xs:
        imin_ap = int(trace_bottom[i])
        imax_ap = int(trace_top[i])

        # estimate background:
        # take average of the sky at both sides of the aperture
        imin = int(bkg_bottom[i])
        sky_bottom = np.nanmedian(data[imin - sky_width : imin, i])
        imax = int(bkg_top[i])
        sky_top = np.nanmedian(data[imax : imax + sky_width, i])
        bkg_sky = (sky_bottom + sky_top) / 2
        # background subtraction
        slice_data = data[imin_ap:imax_ap, i] - bkg_sky

        # sum the counts inside the trace aperture
        raw_spectrum[i] = np.nansum(slice_data)

    # the axis is inverted
    raw_spectrum = raw_spectrum[::-1]

    if plot_trace:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(raw_spectrum)
        ax.set_xlabel("Dispersion axis (pixels)", fontsize=16)
        ax.set_ylabel("Raw Flux", fontsize=16)
        ax.set_title(header["OBJECT"], fontsize=16)
        plt.show()

    return raw_spectrum


def quick_1Dreduction(center=None, plot_diag=False, plot_trace=False, order=3):
    """Performs a "quick" 2D image reduction.

    Mostly default parameters are used, but should work in most cases.

    Parameters
    ----------
    plot_diag: bool, default ``False``
        If ``True``, a set of diagnostic plots are shown for each step and the final solution as well.
    plot_trace: bool, default ``False``
        If ``True``, the image is plotted with the trace. The raw spectrum is also plotted.
    """
    config = dotenv_values(".env")
    PROCESSING = config["PROCESSING"]

    # get 2D reduced data
    files = glob.glob(os.path.join(PROCESSING, "*_2d.fits"))
    for file in files:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyWarning)
            hdu = fits.open(file)
            header = hdu[0].header

        # extract trace
        try:
            raw_spectrum = optimised_trace(
                hdu, center=center, plot_diag=plot_diag, plot_trace=plot_trace, t_order=order
            )
            hdu[0].data = raw_spectrum
            # update header
            header["NAXIS"] = 1
            header["NAXIS2"] = len(raw_spectrum)
            del header["NAXIS2"]

            object_name = os.path.basename(file).split("_")[0]
            outfile = os.path.join(PROCESSING, f"{object_name}_1d.fits")
            hdu.writeto(outfile, overwrite=True)
        except Exception as exc:
            print(f'{exc}: {file}')
