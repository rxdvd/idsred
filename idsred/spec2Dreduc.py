import os
import numpy as np
from dotenv import dotenv_values

import astropy.units as u
from astropy.io import fits
from astropy.stats import mad_std

import ccdproc
from ccdproc import CCDData

from .utils import collect_data, update_header

import warnings
from astropy.utils.exceptions import AstropyWarning

import logging
logging.getLogger().setLevel(logging.ERROR)  # not ideal, but ignores annoying warnings

def validate_method(method):
    """Checks the validity of a method for combining images.
    
    Parameters
    ----------
    method: str
        Method for conbining images: ``median`` or ``average``. 
    """
    valid_methods = ['median', 'average']
    assert method in valid_methods, f"the method used in not valid, choose from {valid_methods}"
    
    
def create_images_list(observations, obstype, subtract_overscan=True, trim_image=True, master_bias=None):
    """Creates a list of images.
    
    The images can be overscan subtracted, trimmed and bias subtracted.
    
    Parameters
    ----------
    observations: `~ImageFileCollection`
        Table-like object with images information.
    obstype: str
        Type of Image. E.g. ``BIAS``, ``FLAT``.
    subtract_overscan: bool, default ``True``.
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
    
    for filename in observations.files_filtered(include_path=True, obstype=obstype):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyWarning)
            ccd = CCDData.read(filename, hdu=1, unit=u.adu)
            
        ## Set up the reduction based on Gain and Readout.
        ccd = ccdproc.create_deviation(
            ccd, 
            gain = ccd.header['GAIN']*u.electron / u.adu,
            readnoise = ccd.header['READNOIS'] * u.electron
        )
        ## Actually correct for gain.
        ccd = ccdproc.gain_correct(ccd, ccd.header['GAIN'] * u.electron / u.adu)

        if subtract_overscan:
            ccd = ccdproc.subtract_overscan(ccd, median=True,  overscan_axis=0, fits_section=ccd.header['BIASSEC'])
        if trim_image:
            ccd = ccdproc.trim_image(ccd, ccd.header['TRIMSEC'])
        if master_bias is not None:
            ccd = ccdproc.subtract_bias(ccd, master_bias) 
        images_list.append(ccd)
        
    return images_list


def inv_median(array):
    """Inverse median function.
    """
    return 1 / np.nanmedian(array)
    
    
def combine_images(images_list, method='average', scale=None):
    """Combines a list of images.
    
    Parameters
    ----------
    images_list: list
        List of images.
    method: str, default ``average``
        Method for conbining images: ``median`` or ``average``.
    scale: function or `numpy.ndarray`-like or None, optional
    
    Returns
    -------
    master_image: `~astropy.nddata.CCDData`
        Combined image.
    """
    validate_method(method)
    if method=='median':
        master_image = ccdproc.combine(images_list, method=method, scale=scale)
    
    else:
        # average
        master_image = ccdproc.combine(images_list, method=method, scale=scale,
                                       sigma_clip=True, sigma_clip_low_thresh=5, 
                                       sigma_clip_high_thresh=5, sigma_clip_func=np.ma.median, 
                                       signma_clip_dev_func=mad_std, mem_limit=350e6)   
        
    return master_image
        

def create_master_bias(observations, subtract_overscan=True, trim_image=True, 
                       method='average', save_output=True):
    """Creates a master bias image.
        
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
    proc_dir: bool, default ``None``
        Processing directory (where the output are saved). If ``None``, the current
        directory is used.
    save_output: bool, default ``True``
        If ``True``, the master flat image is saved in the processing directory.
    
    Returns
    -------
    master_bias: `~astropy.nddata.CCDData`
        Master bias image.
    """
    obstype = 'BIAS'
    bias_list = create_images_list(observations, obstype, subtract_overscan, trim_image)
    master_bias = combine_images(bias_list, method)
    print(f'{len(bias_list)} images combined for the master BIAS')
    
    if save_output:
        config = dotenv_values(".env")
        PROCESSING = config['PROCESSING']
        outfile = os.path.join(PROCESSING, 'master_bias.fits')
        if not os.path.isdir(PROCESSING):
            os.mkdir(PROCESSING)
        master_bias.write(outfile, overwrite=True)
    
    return master_bias
    
    
def create_master_flat(observations, master_bias=None, subtract_overscan=False, 
                       trim_image=True, method='average', scale_flats=True, save_output=True):
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
        Method for conbining images: ``median`` or ``average``.
    scale_flats: bool, default ``True``
        If ``True``, the flats are scaled by the inverse median before being combined.
    proc_dir: bool, default ``None``
        Processing directory (where the output are saved). If ``None``, the current
        directory is used.
    save_output: bool, default ``True``
        If ``True``, the master flat image is saved in the processing directory.
    
    Returns
    -------
    master_flat: `~astropy.nddata.CCDData`
        Master flat image.
    """
    obstype = 'FLAT'
    if scale_flats:
        scale = inv_median
    else:
        scale = None
        
    #subtract_overscan = False
    flat_list = create_images_list(observations, obstype, subtract_overscan, trim_image, master_bias)
    master_flat = combine_images(flat_list, method, scale=scale)
    print(f'{len(flat_list)} images combined for the master FLAT')
    
    if save_output:
        config = dotenv_values(".env")
        PROCESSING = config['PROCESSING']
        outfile = os.path.join(PROCESSING, 'master_flat.fits')
        if not os.path.isdir(PROCESSING):
            os.mkdir(PROCESSING)
        master_flat.write(outfile, overwrite=True)
    
    return master_flat


def create_master_arc(observations, beginning=True,
                       method='average', save_output=True):
    """Creates a master bias image.

    Parameters
    ----------
    observations: `~ImageFileCollection`
        Table-like object with images information.
    beginning: bool, default ``True``
        If ``True``, the arcs from the beginning of the night are used.
    method: str, default ``average``
        Method for combining images: ``median`` or ``average``.
    proc_dir: bool, default ``None``
        Processing directory (where the output are saved). If ``None``, the current
        directory is used.
    save_output: bool, default ``True``
        If ``True``, the master flat image is saved in the processing directory.

    Returns
    -------
    master_arc: `~astropy.nddata.CCDData`
        Master arc image.
    """
    obstype = 'ARC'
    obs_files = observations.filter(obstype=obstype).files
    if not beginning:
        # use the ARCs from the end of the night
        obs_files = obs_files[::-1]

    numbers = []
    iraf_names = []
    for file in obs_files:
        basename = os.path.basename(file)
        irafname = basename.split('.')[0]
        number = float(irafname[1:])

        if len(iraf_names) == 0:
            iraf_names.append(irafname)
        elif any(np.abs(number - np.array(numbers)) < 2):
            iraf_names.append(irafname)
        else:
            break
        numbers.append(number)

    irafname_mask = '|'.join(irafname for irafname in iraf_names)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        arc_observations = observations.filter(regex_match=True, irafname=irafname_mask)


    arc_list = create_images_list(arc_observations, obstype, subtract_overscan=False, trim_image=False)
    master_arc = combine_images(arc_list, method)
    print(f'{len(arc_list)} images combined for the master ARC')

    if save_output:
        config = dotenv_values(".env")
        PROCESSING = config['PROCESSING']
        outfile = os.path.join(PROCESSING, 'master_arc.fits')
        if not os.path.isdir(PROCESSING):
            os.mkdir(PROCESSING)
        master_arc.write(outfile, overwrite=True)

    return master_arc


def reduce_images(observations, master_bias=None, master_flat=None, subtract_overscan=False, 
                  trim_image=True, method='average', save_output=True):
    """Reduces science images.
    
    If more than one image of the same target is given, these are combined.
    
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
        Method for conbining images: ``median`` or ``average``.
    proc_dir: bool, default ``None``
        Processing directory (where the output are saved). If ``None``, the current
        directory is used.
    save_output: bool, default ``True``
        If ``True``, the science images are saved in the processing directory.
        
    Returns
    -------
    red_images: list
        List of reduced images.
    """
    obs_df = observations.summary.to_pandas()
    object_names = obs_df[obs_df.obstype=='TARGET'].object.unique()

    red_images = []
    for object_name in object_names:
        if 'focus' in object_name:
            continue  # skips this
        print("Reducing:", object_name)
        target_list = []
        
        for i, filename in enumerate(observations.files_filtered(include_path=True, object=object_name)):
            hdu = fits.open(filename)
            print(filename)
            header = hdu[0].header+hdu[1].header
            if i==0:
                # for the output
                master_header = header
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", AstropyWarning)
                ccd = CCDData(hdu[1].data, header=header, unit=u.adu)
                
            ccd = ccdproc.create_deviation(
                ccd,
                gain = ccd.header['GAIN'] * u.electron / u.adu,
                readnoise = ccd.header['READNOIS'] * u.electron
            )
            ccd = ccdproc.gain_correct(ccd, ccd.header['GAIN'] * u.electron / u.adu)

            try:
                ccd = ccdproc.cosmicray_lacosmic(ccd, niter=10)
                if subtract_overscan:
                    ccd = ccdproc.subtract_overscan(ccd, median=True,  overscan_axis=0, fits_section=ccd.header['BIASSEC'])
                if trim_image:
                    ccd = ccdproc.trim_image(ccd, ccd.header['TRIMSEC'])
                if master_bias is not None:
                    ccd = ccdproc.subtract_bias(ccd, master_bias)
                if master_flat is not None:
                    ccd = ccdproc.flat_correct(ccd, master_flat, min_value = 0.1)
        
                # Rotate Frame
                ccd.data = ccd.data.T
                ccd.mask = ccd.mask.T
                target_list.append(ccd)
            except Exception as error:
                print(error)

        if len(target_list)>0:
            validate_method(method)
            combiner = ccdproc.Combiner(target_list)
            if method=='average':
                red_target = combiner.average_combine()
            else:
                red_target = combiner.median_combine()

            red_hdu = red_target.to_hdu()
            update_header(red_hdu, master_header)
            
            if save_output:
                config = dotenv_values(".env")
                PROCESSING = config['PROCESSING']
                outfile = os.path.join(PROCESSING, f'{object_name}_2d.fits')
                if not os.path.isdir(PROCESSING):
                    os.mkdir(PROCESSING)
                red_hdu.writeto(outfile, overwrite=True)

            red_images.append(red_hdu)
        
    return red_images

def quick_2Dreduction(observations=None, subtract_overscan=True, trim_image=True, method='average'):

    if observations is None:
        observations = collect_data()

    create_master_arc(observations, beginning=True, method=method)
    master_bias = create_master_bias(observations, subtract_overscan, trim_image, method)
    master_flat = create_master_flat(observations, master_bias, subtract_overscan, trim_image, method)
    _ = reduce_images(observations, master_bias, master_flat, subtract_overscan, trim_image, method)