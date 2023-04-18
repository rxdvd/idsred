import os
import numpy as np
from dotenv import dotenv_values

import astropy.units as u
from astropy.io import fits
from astropy.time import Time
from astropy.stats import mad_std

import ccdproc
from ccdproc import CCDData

from .utils import collect_data, update_header

import warnings
from astropy.utils.exceptions import AstropyWarning

import logging

logging.getLogger().setLevel(
    logging.ERROR
)  # not ideal, but ignores annoying warnings


def _validate_method(method):
    """Checks the validity of a method for combining images.

    Parameters
    ----------
    method: str
        Method for combining images: ``median`` or ``average``.
    """
    valid_methods = ["median", "average"]
    assert (
        method in valid_methods
    ), f"the method used in not valid, choose from {valid_methods}"


def create_images_list(
    observations,
    obstype,
    subtract_overscan=False,
    trim_image=True,
    master_bias=None,
):
    """Creates a list of images.

    The images can be overscan subtracted, trimmed and bias subtracted.

    Parameters
    ----------
    observations: `~ImageFileCollection`
        Table-like object with images information.
    obstype: str
        Type of Image. E.g. ``BIAS``, ``FLAT``.
    subtract_overscan: bool, default ``False``.
        If ``True``, the image gets overscan subtract.
    trim_image: bool, default ``True``.
        If ``True``, the image gets trimmed.
    master_bias: `~astropy.nddata.CCDData`-like, array-like or None, optional
        Master bias image. If given, images are bias subtracted.

    Returns
    -------
    images_list: list
        List of images.
    """
    images_list = []

    for filename in observations.files_filtered(
        include_path=True, obstype=obstype
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyWarning)
            # initial data on first extension
            ccd = CCDData.read(filename, hdu=1, unit=u.electron)

        if subtract_overscan is True:
            ccd = ccdproc.subtract_overscan(
                ccd,
                median=True,
                overscan_axis=0,
                fits_section=ccd.header["BIASSEC"],
            )
        if trim_image is True:
            ccd = ccdproc.trim_image(ccd, ccd.header["TRIMSEC"])
            # ccd = ccdproc.trim_image(ccd[360:3601, :])
        if master_bias is not None:
            ccd = ccdproc.subtract_bias(ccd, master_bias)
        images_list.append(ccd)

    return images_list


def _inv_median(array):
    """Inverse median function."""
    return 1 / np.nanmedian(array)


def combine_images(images_list, method="average", scale=None):
    """Combines a list of images.

    Parameters
    ----------
    images_list: list
        List of images.
    method: str, default ``average``
        Method for combining images: ``median`` or ``average``.
    scale: function or `numpy.ndarray`-like or None, optional

    Returns
    -------
    master_image: `~astropy.nddata.CCDData`
        Combined image.
    """
    _validate_method(method)
    if method == "median":
        master_image = ccdproc.combine(images_list, method=method, scale=scale)

    elif method == "average":
        master_image = ccdproc.combine(
            images_list,
            method=method,
            scale=scale,
            sigma_clip=True,
            sigma_clip_low_thresh=5,
            sigma_clip_high_thresh=5,
            sigma_clip_func=np.ma.median,
            signma_clip_dev_func=mad_std,
            mem_limit=350e6,
        )
    else:
        raise ValueError("Not a valid method:", method)

    return master_image


def create_master_bias(
    observations,
    subtract_overscan=False,
    trim_image=True,
    method="average",
    save_output=True,
):
    """Creates a master bias image.

    Parameters
    ----------
    observations: `~ImageFileCollection`
        Table-like object with images information.
    subtract_overscan: bool, default ``False``.
        If ``True``, the image gets overscan subtract.
    trim_image: bool, default ``True``.
        If ``True``, the image gets trimmed.
    method: str, default ``average``
        Method for combining images: ``median`` or ``average``.
    save_output: bool, default ``True``
        If ``True``, the master flat image is saved in the processing directory.

    Returns
    -------
    master_bias: `~astropy.nddata.CCDData`
        Master bias image.
    """
    obstype = "BIAS"
    bias_list = create_images_list(
        observations, obstype, subtract_overscan, trim_image
    )
    master_bias = combine_images(bias_list, method)
    master_bias.header["OBJECT"] = "MASTER_BIAS"
    print(f"{len(bias_list)} images combined for the master BIAS")

    if save_output is True:
        config = dotenv_values(".env")
        PROCESSING = config["PROCESSING"]
        outfile = os.path.join(PROCESSING, "master_bias.fits")
        if not os.path.isdir(PROCESSING):
            os.mkdir(PROCESSING)
        master_bias.write(outfile, overwrite=True)

    return master_bias


def correct_flat(flat):
    """Filters out the strong and general wavelength dependence of
    CCD’s sensitivity and spectrograph+telescope+response.

    Parameters
    ----------
    flat: array
        Combined flats.

    Returns
    -------
    master_flat: array
        Corrected master flat.
    """
    avcol_in = np.average(
        flat[:, 30:306].copy(), axis=1
    )  # average column value, avoiding edges
    avcol_out = np.concatenate([[avcol_in]] * flat.shape[1], axis=0).T
    master_flat = np.copy(flat / avcol_out)

    return master_flat


def create_master_flat(
    observations,
    master_bias=None,
    subtract_overscan=False,
    trim_image=True,
    method="average",
    scale_flats=True,
    save_output=True,
    corr_flat=True,
):
    """Creates a master flat image.

    Parameters
    ----------
    observations: `~ImageFileCollection`
        Table-like object with images information.
    master_bias: `~astropy.nddata.CCDData`-like, array-like or None, optional
        Master bias image. If given, images are bias subtracted.
    subtract_overscan: bool, default ``True``.
        If ``True``, the image gets overscan subtract.
    trim_image: bool, default ``True``.
        If ``True``, the image gets trimmed.
    method: str, default ``average``
        Method for combining images: ``median`` or ``average``.
    scale_flats: bool, default ``True``
        If ``True``, the flats are scaled by the inverse median before being combined.
    save_output: bool, default ``True``
        If ``True``, the master flat image is saved in the processing directory.

    Returns
    -------
    master_flat: `~astropy.nddata.CCDData`
        Master flat image.
    """
    obstype = "FLAT"
    if scale_flats:
        scale = _inv_median
    else:
        scale = None

    flat_list = create_images_list(
        observations, obstype, subtract_overscan, trim_image, master_bias
    )
    for flat in flat_list:
        flat.data[flat.data <= 0] = 100
    master_flat = combine_images(flat_list, method, scale=scale)
    master_flat.header["OBJECT"] = "MASTER_FLAT"
    print(f"{len(flat_list)} images combined for the master FLAT")

    if corr_flat is True:
        master_flat.data = correct_flat(master_flat)

    if save_output is True:
        config = dotenv_values(".env")
        PROCESSING = config["PROCESSING"]
        outfile = os.path.join(PROCESSING, "master_flat.fits")
        if not os.path.isdir(PROCESSING):
            os.mkdir(PROCESSING)
        master_flat.write(outfile, overwrite=True)

    return master_flat


def extract_arcs(observations, method="average", trim_image=True):
    """Obtains the ARC files closest in time to each of the targets.

    Parameters
    ----------
    observations: `~ImageFileCollection`
        Table-like object with images information.
    method: str, default ``average``
        Method for combining images: ``median`` or ``average``.
    trim_image: bool, default ``True``.
        If ``True``, the image gets trimmed.
    """
    config = dotenv_values(".env")
    PROCESSING = config["PROCESSING"]

    obs_df = observations.summary.to_pandas()

    # get the mid MJD of each file
    times = obs_df["date-obs"].values + "T" + obs_df["ut-mid"].values
    times = Time(list(times), format="isot", scale="utc")
    obs_df["mid_mjd"] = times.mjd

    targets_df = obs_df[obs_df.obstype == "TARGET"]
    arc_df = obs_df[obs_df.obstype == "ARC"]

    # get the closest arc in time for each target
    for row in targets_df.iterrows():
        dt = np.abs(row[1].mid_mjd - arc_df.mid_mjd.values)
        arc_id = np.argmin(dt)
        arc_file = os.path.join(PROCESSING, "..", arc_df.file.values[arc_id])

        # update header
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyWarning)
            target_name = row[1].object
            arc_ccd = CCDData.read(arc_file, hdu=1, unit=u.electron)
            arc_ccd.header["OBJECT"] = f"ARC_{target_name}"
            arc_ccd = combine_images([arc_ccd], method=method)

            if trim_image is True:
                arc_ccd = ccdproc.trim_image(
                    arc_ccd, arc_ccd.header["TRIMSEC"]
                )

        # save output
        outfile = os.path.join(PROCESSING, f"arc_{target_name}.fits")
        arc_ccd.write(outfile, overwrite=True)


def create_master_arc(
    observations,
    beginning=True,
    method="average",
    trim_image=True,
    save_output=True,
):
    """Creates a master bias image.

    ARC files closest in time to each of the targets are also extracted.

    Parameters
    ----------
    observations: `~ImageFileCollection`
        Table-like object with images information.
    beginning: bool, default ``True``
        If ``True``, the arcs from the beginning of the night are used.
    method: str, default ``average``
        Method for combining images: ``median`` or ``average``.
    trim_image: bool, default ``True``.
        If ``True``, the image gets trimmed.
    save_output: bool, default ``True``
        If ``True``, the master flat image is saved in the processing directory.

    Returns
    -------
    master_arc: `~astropy.nddata.CCDData`
        Master arc image.
    """
    config = dotenv_values(".env")
    PROCESSING = config["PROCESSING"]
    if not os.path.isdir(PROCESSING):
        os.mkdir(PROCESSING)

    extract_arcs(observations, method=method, trim_image=trim_image)
    print("extraction of ARC files for each of the targets successful")

    obstype = "ARC"
    obs_files = observations.filter(obstype=obstype).files
    if not beginning:
        # use the ARCs from the end of the night
        obs_files = obs_files[::-1]

    numbers = []
    iraf_names = []
    for file in obs_files:
        basename = os.path.basename(file)
        irafname = basename.split(".")[0]
        number = float(irafname[1:])

        if len(iraf_names) == 0:
            iraf_names.append(irafname)
        elif any(np.abs(number - np.array(numbers)) < 2):
            iraf_names.append(irafname)
        else:
            break
        numbers.append(number)

    irafname_mask = "|".join(irafname for irafname in iraf_names)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        arc_observations = observations.filter(
            regex_match=True, irafname=irafname_mask
        )

    arc_list = create_images_list(
        arc_observations,
        obstype,
        subtract_overscan=False,
        trim_image=trim_image,
    )
    master_arc = combine_images(arc_list, method)
    master_arc.header["OBJECT"] = "MASTER_ARC"
    print(f"{len(arc_list)} images combined for the master ARC")

    if save_output is True:
        outfile = os.path.join(PROCESSING, "master_arc.fits")
        master_arc.write(outfile, overwrite=True)

    return master_arc


def reduce_images(
    observations,
    master_bias=None,
    master_flat=None,
    subtract_overscan=False,
    trim_image=True,
    method="average",
    save_output=True,
):
    """Reduces science images.

    If more than one image of the same target is given, these are combined.

    Parameters
    ----------
    observations: `~ImageFileCollection`
        Table-like object with images information.
    master_bias: `~astropy.nddata.CCDData`-like, array-like or None, optional
        Master bias image. If given, images are bias subtracted.
    master_flat: `~astropy.nddata.CCDData`-like, array-like or None, optional
        Master flat image. If given, images are flat-fielded.
    subtract_overscan: bool, default ``True``.
        If ``True``, the image gets overscan subtract.
    trim_image: bool, default ``True``.
        If ``True``, the image gets trimmed.
    method: str, default ``average``
        Method for combining images: ``median`` or ``average``.
    save_output: bool, default ``True``
        If ``True``, the science images are saved in the processing directory.

    Returns
    -------
    red_images: list
        List of reduced images.
    """
    obs_df = observations.summary.to_pandas()
    object_names = obs_df[obs_df.obstype == "TARGET"].object.unique()

    red_images = []
    for object_name in object_names:
        if "focus" in object_name:
            continue  # skips this
        print("Reducing:", object_name)
        target_list = []

        for i, filename in enumerate(
            observations.files_filtered(include_path=True, object=object_name)
        ):
            hdu = fits.open(filename)
            print(filename)
            header = hdu[0].header + hdu[1].header
            if i == 0:
                # for the output
                master_header = header
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", AstropyWarning)
                ccd = CCDData(hdu[1].data, header=header, unit=u.electron)

            try:
                ccd = ccdproc.cosmicray_lacosmic(
                    ccd, niter=10, sigfrac=0.3, psffwhm=2.5
                )
                if subtract_overscan is True:
                    ccd = ccdproc.subtract_overscan(
                        ccd,
                        median=True,
                        overscan_axis=0,
                        fits_section=ccd.header["BIASSEC"],
                    )
                if trim_image is True:
                    ccd = ccdproc.trim_image(ccd, ccd.header["TRIMSEC"])
                    # ccd = ccdproc.trim_image(ccd[360:3601, :])
                if master_bias is not None:
                    ccd = ccdproc.subtract_bias(ccd, master_bias)
                if master_flat is not None:
                    ccd = ccdproc.flat_correct(
                        ccd, master_flat, min_value=0.01
                    )

                # Rotate Frame
                ccd.data = ccd.data.T
                ccd.mask = ccd.mask.T
                target_list.append(ccd)
            except Exception as error:
                print(error)

        if len(target_list) > 0:
            _validate_method(method)
            combiner = ccdproc.Combiner(target_list)
            if method == "average":
                red_target = combiner.average_combine()
            elif method == "median":
                red_target = combiner.median_combine()
            else:
                raise ValueError("Not a valid method:", method)

            red_hdu = red_target.to_hdu()
            update_header(red_hdu, master_header)

            if save_output is True:
                config = dotenv_values(".env")
                PROCESSING = config["PROCESSING"]
                outfile = os.path.join(PROCESSING, f"{object_name}_2d.fits")
                if not os.path.isdir(PROCESSING):
                    os.mkdir(PROCESSING)
                red_hdu.writeto(outfile, overwrite=True)

            red_images.append(red_hdu)

    return red_images


def quick_2Dreduction(
    observations=None,
    subtract_overscan=False,
    trim_image=True,
    method="average",
    scale_flats=True,
    corr_flat=True,
):
    """Performs a "quick" 2D image reduction.

    Mostly default parameters are used, but should work in most cases.

    Parameters
    ----------
    observations: `~ImageFileCollection`
        Table-like object with images information.
    subtract_overscan: bool, default ``True``.
        If ``True``, the image gets overscan subtract.
    trim_image: bool, default ``True``.
        If ``True``, the image gets trimmed.
    method: str, default ``average``
        Method for combining images: ``median`` or ``average``.
    """
    if observations is None:
        observations = collect_data()

    create_master_arc(
        observations, beginning=True, method=method, trim_image=trim_image
    )
    master_bias = create_master_bias(
        observations,
        subtract_overscan=subtract_overscan,
        trim_image=trim_image,
        method=method,
    )
    master_flat = create_master_flat(
        observations,
        master_bias,
        subtract_overscan=subtract_overscan,
        trim_image=trim_image,
        method=method,
        scale_flats=scale_flats,
        corr_flat=corr_flat,
    )
    _ = reduce_images(
        observations,
        master_bias,
        master_flat,
        subtract_overscan=subtract_overscan,
        trim_image=trim_image,
        method=method,
    )
