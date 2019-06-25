import numpy as np
import scipy
from scipy.interpolate import interp1d
import configparser
import tqdm
import skued

def read_cfg(path_cfg):
    config = configparser.ConfigParser()
    config.read(path_cfg)

    assert 'PATH' in config, "Could not find PATH in the config file."
    assert 'PARAMETERS' in config, "Could not find PARAMETERS in the config file."

    dict_path = {}
    for key in config['PATH']:
        dict_path.update({key: config['PATH'][key]})

    dict_numerics = {}
    for key in config['PARAMETERS']:
        dict_numerics.update({key: int(config['PARAMETERS'][key])})

    return dict_path, dict_numerics


def mask_image(mask_size, list_of_centers, list_of_radii, mask_inverse=False):
    mask = np.ones(mask_size, dtype=np.float)
    assert len(list_of_centers) == len(list_of_radii)
    for idx, center in enumerate(list_of_centers):
        xc = center[1]
        yc = center[0]
        radius = list_of_radii[idx]
        xx, yy = np.meshgrid(np.arange(0, mask.shape[0]), np.arange(0, mask.shape[1]))
        rr = np.empty_like(xx)
        rr = np.hypot(xx - xc, yy - yc)
        if mask_inverse is False:
            mask[rr <= radius] = np.nan
        elif mask_inverse is True:
            mask[rr >= radius] = np.nan
    return mask


def refine_peakpos_arb_dim(peakpos_all, image, numrefine, window_size):
    new_peakpos_all = np.empty_like(peakpos_all)
    for idx, peak in enumerate(peakpos_all):

        lbx = int(peak[1]) - window_size
        ubx = int(peak[1]) + window_size
        lby = int(peak[0]) - window_size
        uby = int(peak[0]) + window_size
        im_p = image[lby:uby, lbx:ubx]
        im_p = np.array(im_p, dtype=np.int64)
        im_p = im_p / np.max(np.abs(im_p))
        cy, cx = scipy.ndimage.measurements.center_of_mass(np.power(im_p, 2))

        new_peak = new_peakpos_all[idx, :]
        new_peak[1] = cx + peak[1] - window_size + 0.5
        new_peak[0] = cy + peak[0] - window_size + 0.5

        counter = 0
        while counter < numrefine:
            lbx = int(new_peak[1]) - window_size
            ubx = int(new_peak[1]) + window_size
            lby = int(new_peak[0]) - window_size
            uby = int(new_peak[0]) + window_size
            im_p = image[lby:uby, lbx:ubx]
            cy, cx = scipy.ndimage.measurements.center_of_mass(np.power(im_p, 2))
            counter = counter + 1

            new_peak[1] = cx + new_peak[1] - window_size - 1
            new_peak[0] = cy + new_peak[0] - window_size - 1

        new_peakpos_all[idx, 0] = new_peak[0]
        new_peakpos_all[idx, 1] = new_peak[1]
    return new_peakpos_all


def centeredDistanceMatrix(n):
    # make sure n is odd
    x, y = np.meshgrid(range(n), range(n))
    return np.sqrt((x - (n / 2) + 1) ** 2 + (y - (n / 2) + 1) ** 2)


def centeredDistanceMatrix_centered(n, xc, yc):
    # make sure n is odd
    x, y = np.meshgrid(range(n), range(n))
    return np.sqrt((x - (n / 2 + xc) + 1) ** 2 + (y - (n / 2 + yc) + 1) ** 2)


# Taken from Laurent
def azimuthal_average(image, center, mask=None, angular_bounds=None, trim=True):
    """
    This function returns an azimuthally-averaged pattern computed from an image,
    e.g. polycrystalline diffraction.
    Parameters
    ----------
    image : array_like, shape (M, N)
        Array or image.
    center : array_like, shape (2,)
        coordinates of the center (in pixels).
    mask : `~numpy.ndarray` or None, optional
        Evaluates to True on valid elements of array.
    angular_bounds : 2-tuple or None, optional
        If not None, the angles between first and second elements of `angular_bounds`
        (inclusively) will be used for the average. Angle bounds are specified in degrees.
        0 degrees is defined as the positive x-axis. Angle bounds outside [0, 360) are mapped back
        to [0, 360).
    trim : bool, optional
        If True, leading and trailing zeros (possible due to the usage of masks) are trimmed.
    Returns
    -------
    radius : `~numpy.ndarray`, ndim 1
        Radius of the average [px]. ``radius`` might not start at zero, depending on the ``trim`` parameter.
    average : `~numpy.ndarray`, ndim 1
        Angular-average of the array.
    """
    if mask is None:
        mask = np.ones_like(image, dtype=np.bool)

    xc, yc = center

    # Create meshgrid and compute radial positions of the data
    # The radial positions are rounded to the nearest integer
    # TODO: interpolation? or is that too slow?
    Y, X = np.indices(image.shape)
    R = np.hypot(X - xc, Y - yc)
    Rint = np.rint(R).astype(np.int)

    if angular_bounds:
        mi, ma = _angle_bounds(angular_bounds)
        angles = (
                np.rad2deg(np.arctan2(Y - yc, X - xc)) + 180
        )  # arctan2 is defined on [-pi, pi] but we want [0, pi]
        in_bounds = np.logical_and(mi <= angles, angles <= ma)
    else:
        in_bounds = np.ones_like(image, dtype=np.bool)

    valid = mask[in_bounds]
    image = image[in_bounds]
    Rint = Rint[in_bounds]

    px_bin = np.bincount(Rint, weights=valid * image)
    r_bin = np.bincount(Rint, weights=valid)
    radius = np.arange(0, r_bin.size)

    # Make sure r_bin is never 0 since it it used for division anyway
    np.maximum(r_bin, 1, out=r_bin)

    # We ignore the leading and trailing zeroes, which could be due to
    first, last = 0, -1
    if trim:
        first, last = _trim_bounds(px_bin)

    return radius[first:last], px_bin[first:last] / r_bin[first:last]


def _trim_bounds(arr):
    """ Returns the bounds which would be used in numpy.trim_zeros """
    first = 0
    for i in arr:
        if i != 0.0:
            break
        else:
            first = first + 1
    last = len(arr)
    for i in arr[::-1]:
        if i != 0.0:
            break
        else:
            last = last - 1
    return first, last


def rings_to_average(d, y, n):
    x = np.arange(n)
    f = interp1d(x, y, fill_value="extrapolate")
    return f(d.flat).reshape(d.shape)


def remove_bgk(image, laser_background, flatfield):
    # Locations where background is larger than the image shouldn't be negative, but zero
    image = np.array(image, dtype=np.float64)
    image[laser_background > image] = 0
    mask = laser_background <= image
    image = mask * (image - laser_background) * flatfield
    return image


def sum_peak_pixels(image, peak, window_size):
    lbx = int(peak[1]) - window_size
    ubx = int(peak[1]) + window_size
    lby = int(peak[0]) - window_size
    uby = int(peak[0]) + window_size
    im_p = image[lby:uby, lbx:ubx]
    return np.nansum(im_p)


def peakpos_evolution(file_list, mask_total, laser_bkg, FF, peakpos_all, numrefine, window_size):
    peakpos_evolution = []
    no_files  = len(file_list)
    no_peaks = np.shape(peakpos_all)[0]
    for idx, f in tqdm.tqdm(enumerate(file_list)):
        image = np.array(skued.diffread(f), dtype = np.int64)
        #checks for saturation
        #if nanmax(nanmax(Image))==65000
        #    msgbox(['Warning: Image ',num2str(k),' is saturated!'])
        #    end
        #Apply mask
        image = image*mask_total
        #Substract background and flatfield
        image = remove_bgk(image, laser_bkg, FF)
        new_peakpos_all = refine_peakpos_arb_dim(peakpos_all, image, numrefine, window_size)
        peakpos_evolution.append(new_peakpos_all)
        peakpos_all = new_peakpos_all

    peakpos_evolution = np.array(peakpos_evolution).reshape(no_files, 2*no_peaks)
    return peakpos_evolution