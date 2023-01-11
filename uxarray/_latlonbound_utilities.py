import numpy as np
import copy

from .helpers import angle_of_2_vectors, within, _normalize_in_place, _convert_node_xyz_to_lonlat_rad, \
    _convert_node_lonlat_rad_to_xyz, _pt_within_gcr


def insert_pt_in_latlonbox(old_box_rad, new_pt_rad, is_lon_periodic=True):
    """Helper function to insert a new point into an existing latlon box. This function will take care of the
    Longitude wrap-around

    Parameters
    ----------
    old_box_rad: float list
        the original lat lon box using the 2D longitude/latitude in radian [[lat_0, lat_1],[lon_0, lon_1]],required
    new_pt_rad: float list
        the inserted new lat lon point [lon, lat], required

    Returns
    ----------
    float list
        the updated lat lon box [[lat_0, lat_1],[lon_0, lon_1]]

    Raises
    ----------
    ValueError
        The longitude of the inserted point is out of range
    """

    # If the box is null (no point inserted yet)

    if old_box_rad[0][0] == old_box_rad[0][1] == 404.0:
        latlon_box = old_box_rad
        latlon_box[0] = [new_pt_rad[0], new_pt_rad[0]]

    if old_box_rad[1][0] == old_box_rad[1][1] == 404.0:
        latlon_box = old_box_rad
        latlon_box[1] = [new_pt_rad[1], new_pt_rad[1]]

    if old_box_rad[0][0] == old_box_rad[0][1] == old_box_rad[1][0] == old_box_rad[1][1] == 404.0:
        return latlon_box

    # Deal with the pole point
    if new_pt_rad[1] == 404.0 and (
            (np.absolute(new_pt_rad[0] - 0.5 * np.pi) < 1.0e-12) or (
            np.absolute(new_pt_rad[0] - (-0.5 * np.pi)) < 1.0e-12)):
        latlon_box = old_box_rad
        if np.absolute(new_pt_rad[0] - 0.5 * np.pi) < 1.0e-12:
            latlon_box[0][1] = 0.5 * np.pi
        elif np.absolute(new_pt_rad[0] - (-0.5 * np.pi)) < 1.0e-12:
            latlon_box[0][0] = -0.5 * np.pi
        return latlon_box

    old_lon_width = 2.0 * np.pi
    lat_pt = new_pt_rad[0]
    lon_pt = new_pt_rad[1]
    latlon_box = old_box_rad  # The returned box

    if lon_pt < 0.0:
        raise ValueError('lon_pt out of range ( {} < 0)"'.format(lon_pt))

    if lon_pt > old_lon_width:
        raise ValueError('lon_pt out of range ( {} > {})"'.format(
            lon_pt, old_lon_width))

    # Expand latitudes
    if lat_pt > latlon_box[0][1]:
        latlon_box[0][1] = lat_pt

    if lat_pt < latlon_box[0][0]:
        latlon_box[0][0] = lat_pt

    # Expand longitude, if non-periodic
    if not is_lon_periodic:
        if lon_pt > latlon_box[1][1]:
            latlon_box[1][1] = lon_pt
        if lon_pt < latlon_box[1][0]:
            latlon_box[1][0] = lon_pt
        return latlon_box

    # New longitude lies within existing range
    if latlon_box[1][0] <= latlon_box[1][1]:
        if latlon_box[1][0] <= lon_pt <= latlon_box[1][1]:
            return latlon_box
    else:
        if lon_pt >= latlon_box[1][0] or lon_pt <= latlon_box[1][1]:
            return latlon_box

    # New longitude lies outside of existing range
    box_a = copy.deepcopy(latlon_box)
    box_a[1][0] = lon_pt

    box_b = copy.deepcopy(latlon_box)
    box_b[1][1] = lon_pt

    # The updated box is the box of minimum width
    d_width_now = get_latlonbox_width(latlon_box)
    d_width_a = get_latlonbox_width(box_a)
    d_width_b = get_latlonbox_width(box_b)

    if (d_width_a - d_width_now) < -1.0e-14 or (d_width_b -
                                                d_width_now) < -1.0e-14:
        raise ValueError('logic error in the latlon box width')

    if d_width_a < d_width_b:
        return box_a
    else:
        return box_b


def get_latlonbox_width(latlonbox_rad, is_lon_periodic=True):
    """Helper function to calculate the latlonbox width. This function will take care of the longitude wrap-around
    case. According to the definition, the width of a latlonbox will never be greater than 180 degree

    Parameters
    ----------
    latlonbox_rad: float list
        the lat lon box using the 2D longitude/latitude in radian [[lat_0, lat_1],[lon_0, lon_1]],required
    is_lon_periodic: bool
        Flag indicating the latlonbox is a regional (default to be True).

    Returns
    ----------
    float
        the width of the latlonbox

    Raises
    ----------
    ValueError
        The longitude of the inserted point is out of range
    """

    if not is_lon_periodic:
        return latlonbox_rad[1][1] - latlonbox_rad[1][0]

    if latlonbox_rad[1][0] == latlonbox_rad[1][1]:
        return 0.0
    elif latlonbox_rad[1][0] <= latlonbox_rad[1][1]:
        return latlonbox_rad[1][1] - latlonbox_rad[1][0]
    else:
        return latlonbox_rad[1][1] - latlonbox_rad[1][0] + (2 * np.pi)