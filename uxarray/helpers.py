import os

import numpy as np
import xarray as xr
from pathlib import PurePath


# helper function to find file type
def determine_file_type(filepath):
    """Checks file path and contents to determine file type. Supports detection
    of UGrid, SCRIP, Exodus and shape file.

    Parameters: string, required
       Filepath of the file for which the filetype is to be determined.

    Returns: string
       File type: ug, exo, scrip or shp

    Raises:
       RuntimeError: Invalid file type
    """
    msg = ""
    mesh_filetype = "unknown"
    # exodus with coord
    try:
        # extract the file name and extension
        path = PurePath(filepath)
        file_extension = path.suffix

        # try to open file with xarray and test for exodus
        ext_ds = xr.open_dataset(filepath, mask_and_scale=False)["coord"]
        mesh_filetype = "exo"
    except KeyError as e:
        # exodus with coordx
        try:
            ext_ds = xr.open_dataset(filepath, mask_and_scale=False)["coordx"]
            mesh_filetype = "exo"
        except KeyError as e:
            # scrip with grid_center_lon
            try:
                ext_ds = xr.open_dataset(
                    filepath, mask_and_scale=False)["grid_center_lon"]
                mesh_filetype = "scrip"
            except KeyError as e:

                # check mesh topology and dimension
                try:
                    standard_name = lambda v: v is not None
                    # getkeys_filter_by_attribute(filepath, attr_name, attr_val)
                    # return type KeysView
                    ext_ds = xr.open_dataset(filepath, mask_and_scale=False)
                    node_coords_dv = ext_ds.filter_by_attrs(
                        node_coordinates=standard_name).keys()
                    face_conn_dv = ext_ds.filter_by_attrs(
                        face_node_connectivity=standard_name).keys()
                    topo_dim_dv = ext_ds.filter_by_attrs(
                        topology_dimension=standard_name).keys()
                    mesh_topo_dv = ext_ds.filter_by_attrs(
                        cf_role="mesh_topology").keys()
                    if list(mesh_topo_dv)[0] != "" and list(topo_dim_dv)[
                        0] != "" and list(face_conn_dv)[0] != "" and list(
                        node_coords_dv)[0] != "":
                        mesh_filetype = "ugrid"
                    else:
                        raise ValueError(
                            "cf_role is other than mesh_topology, the input NetCDF file is not UGRID format"
                        )
                except KeyError as e:
                    msg = str(e) + ': {}'.format(filepath)
    except (TypeError, AttributeError) as e:
        msg = str(e) + ': {}'.format(filepath)
    except (RuntimeError, OSError) as e:
        # check if this is a shp file
        # we won't use xarray to load that file
        if file_extension == ".shp":
            mesh_filetype = "shp"
        else:
            msg = str(e) + ': {}'.format(filepath)
    except ValueError as e:
        # check if this is a shp file
        # we won't use xarray to load that file
        if file_extension == ".shp":
            mesh_filetype = "shp"
        else:
            msg = str(e) + ': {}'.format(filepath)
    finally:
        if msg != "":  # we did not catch this above
            msg = "Unable to determine file type, mesh file not supported" + ': {}'.format(
                filepath)
            print(msg)
            os._exit(0)

    return mesh_filetype


# helper function to insert a new point into the latlon box
def insert_pt_in_latlonbox(old_box, new_pt, is_lon_periodic=True):
    """Compare the new point's latitude and longitude with the target the latlonbox.

    Parameters: old_box: float array, the original lat lon box [[lat_0, lat_1],[lon_0, lon_1]],required
                new_pt: float array, the new lat lon point [lat, lon], required
                is_lon_periodic: Flag indicating the latlonbox is a regional (default to be True).

    Returns: float array, a lat lon box [[lat_0, lat_1],[lon_0, lon_1]]

    Raises:
       Exception: Logic Errors
    """
    old_lon_width = 2.0 * np.pi
    lat_pt = new_pt[0]
    lon_pt = new_pt[1]
    latlon_box = old_box  # The returned point

    if lon_pt < 0.0:
        raise Exception('lon_pt out of range ( {} < 0)"'.format(lon_pt))

    if lon_pt > old_lon_width:
        raise Exception('lon_pt out of range ( {} > {})"'.format(lon_pt, old_lon_width))

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
        return

    # New longitude lies within existing range
    if latlon_box[1][0] <= latlon_box[1][1]:
        if lon_pt >= latlon_box[1][0] and lon_pt <= latlon_box[1][1]:
            return
    else:
        if lon_pt >= latlon_box[1][0] or lon_pt <= latlon_box[1][0]:
            return

    # New longitude lies outside of existing range
    box_a = latlon_box
    box_a[1][0] = lon_pt

    box_b = latlon_box
    box_b[1][1] = lon_pt

    # The updated box is the box of minimum width
    d_width_now = get_latlonbox_width(latlon_box)
    d_width_a = get_latlonbox_width(box_a)
    d_width_b = get_latlonbox_width(box_b)

    if (d_width_a - d_width_now) < -1.0e-14 or (d_width_b - d_width_now) < -1.0e-14:
        raise Exception('logic error')

    if d_width_a < d_width_b:
        return box_a
    else:
        return box_b



# helper function to calculate the latlonbox width
def get_latlonbox_width(latlonbox, is_lon_periodic):
    """Calculate the width of this LatLonBox
    Parameters: latlonbox: float array, lat lon box [[lat_0, lat_1],[lon_0, lon_1]],required
                is_lon_periodic: boolean, Flag indicating the latlonbox is a regional (default to be True).

    Returns: float array, a lat lon box [[lat_0, lat_1],[lon_0, lon_1]]

    Raises:
       Exception: Logic Errors
    """

    if not is_lon_periodic:
        return latlonbox[1][1] - latlonbox[1][0]

    if latlonbox[1][0] == latlonbox[1][1]:
        return 0.0
    elif latlonbox[1][0] <= latlonbox[1][1]:
        return latlonbox[1][1] - latlonbox[1][0]
    else:
        latlonbox[1][1] - latlonbox[1][0] + (2 * np.pi)

# helper function to calculate latitude and longitude from a node's normalized 3D Cartesian
# coordinates, in radians.
def convert_node_XYZ_2_latlon_rad(node_coord):
    """Calculate the latitude and longitude in radiance for a node represented in the [x, y, z] 3D Cartesian coordinates.
    Parameters: node_coord: float array, [x, y, z],required

    Returns: float array, [latitude_rad, longitude_rad]

    Raises:
       Exception: Logic Errors
    """
    reference_tolerance = 1.0e-12
    dx = node_coord[0]
    dy = node_coord[1]
    dz = node_coord[2]

    d_mag_2 = dx * dx + dy * dy + dz * dz

    if np.absolute(d_mag_2 - 1.0) >= 0.01:
        raise Exception('Grid point has non-unit magnitude:({}, {}, {}) (magnitude {})'.format(dx, dy, dz,d_mag_2 )) #"(%1.15e, %1.15e, %1.15e) (magnitude %1.15e)",

    d_mag = np.absolute(d_mag_2)
    dx /= d_mag
    dy /= d_mag
    dz /= d_mag

    d_lon_rad = 0.0
    d_lat_rad = 0.0

    if np.absolute(dz) < (1.0 - reference_tolerance):
        d_lon_rad = np.arctan(dy/dx)
        d_lat_rad = np.arcsin(dz)

        if d_lon_rad < 0.0:
            d_lon_rad += 2.0 * np.pi
    elif dz > 0.0:
        d_lon_rad = 0.0
        d_lat_rad = 0.5 * np.pi
    else:
        d_lon_rad = 0.0
        d_lat_rad = -0.5 * np.pi

    return [d_lat_rad, d_lon_rad]


# helper function to insert a new point into the latlon box
def insert_pt_in_latlonbox(old_box, new_pt, is_lon_periodic=True):
    """Compare the new point's latitude and longitude with the target the latlonbox.

    Parameters: old_box: float array, the original lat lon box [[lat_0, lat_1],[lon_0, lon_1]],required
                new_pt: float array, the new lat lon point [lat, lon], required
                is_lon_periodic: Flag indicating the latlonbox is a regional (default to be True).

    Returns: float array, a lat lon box [[lat_0, lat_1],[lon_0, lon_1]]

    Raises:
       Exception: Logic Errors
    """
    old_lon_width = 2.0 * np.pi
    lat_pt = new_pt[0]
    lon_pt = new_pt[1]
    latlon_box = old_box  # The returned point

    if lon_pt < 0.0:
        raise Exception('lon_pt out of range ( {} < 0)"'.format(lon_pt))

    if lon_pt > old_lon_width:
        raise Exception('lon_pt out of range ( {} > {})"'.format(lon_pt, old_lon_width))

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
        return

    # New longitude lies within existing range
    if latlon_box[1][0] <= latlon_box[1][1]:
        if lon_pt >= latlon_box[1][0] and lon_pt <= latlon_box[1][1]:
            return
    else:
        if lon_pt >= latlon_box[1][0] or lon_pt <= latlon_box[1][0]:
            return

    # New longitude lies outside of existing range
    box_a = latlon_box
    box_a[1][0] = lon_pt

    box_b = latlon_box
    box_b[1][1] = lon_pt

    # The updated box is the box of minimum width
    d_width_now = get_latlonbox_width(latlon_box)
    d_width_a = get_latlonbox_width(box_a)
    d_width_b = get_latlonbox_width(box_b)

    if (d_width_a - d_width_now) < -1.0e-14 or (d_width_b - d_width_now) < -1.0e-14:
        raise Exception('logic error')

    if d_width_a < d_width_b:
        return box_a
    else:
        return box_b



# helper function to calculate the latlonbox width
def get_latlonbox_width(latlonbox, is_lon_periodic):
    """Calculate the width of this LatLonBox
    Parameters: latlonbox: float array, lat lon box [[lat_0, lat_1],[lon_0, lon_1]],required
                is_lon_periodic: boolean, Flag indicating the latlonbox is a regional (default to be True).

    Returns: float array, a lat lon box [[lat_0, lat_1],[lon_0, lon_1]]

    Raises:
       Exception: Logic Errors
    """

    if not is_lon_periodic:
        return latlonbox[1][1] - latlonbox[1][0]

    if latlonbox[1][0] == latlonbox[1][1]:
        return 0.0
    elif latlonbox[1][0] <= latlonbox[1][1]:
        return latlonbox[1][1] - latlonbox[1][0]
    else:
        latlonbox[1][1] - latlonbox[1][0] + (2 * np.pi)

# helper function to calculate latitude and longitude from a node's normalized 3D Cartesian
# coordinates, in radians.
def convert_node_XYZ_2_latlon_rad(node_coord):
    """Calculate the latitude and longitude in radiance for a node represented in the [x, y, z] 3D Cartesian coordinates.
    Parameters: node_coord: float array, [x, y, z],required

    Returns: float array, [latitude_rad, longitude_rad]

    Raises:
       Exception: Logic Errors
    """
    reference_tolerance = 1.0e-12
    dx = node_coord[0]
    dy = node_coord[1]
    dz = node_coord[2]

    d_mag_2 = dx * dx + dy * dy + dz * dz

    if np.absolute(d_mag_2 - 1.0) >= 0.01:
        raise Exception('Grid point has non-unit magnitude:({}, {}, {}) (magnitude {})'.format(dx, dy, dz,d_mag_2 )) #"(%1.15e, %1.15e, %1.15e) (magnitude %1.15e)",

    d_mag = np.absolute(d_mag_2)
    dx /= d_mag
    dy /= d_mag
    dz /= d_mag

    d_lon_rad = 0.0
    d_lat_rad = 0.0

    if np.absolute(dz) < (1.0 - reference_tolerance):
        d_lon_rad = np.arctan(dy/dx)
        d_lat_rad = np.arcsin(dz)

        if d_lon_rad < 0.0:
            d_lon_rad += 2.0 * np.pi
    elif dz > 0.0:
        d_lon_rad = 0.0
        d_lat_rad = 0.5 * np.pi
    else:
        d_lon_rad = 0.0
        d_lat_rad = -0.5 * np.pi

    return [d_lat_rad, d_lon_rad]
