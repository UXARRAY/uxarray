import xarray as xr
import numpy as np
import math

from numba import njit, config

from uxarray.constants import ENABLE_JIT_CACHE, ENABLE_JIT, ERROR_TOLERANCE

config.DISABLE_JIT = not ENABLE_JIT


@njit(cache=ENABLE_JIT_CACHE)
def node_lonlat_rad_to_xyz(node_coord):
    """Helper function to Convert the node coordinate from 2D
    longitude/latitude to normalized 3D xyz.

    Parameters
    ----------
    node: float list
        2D coordinates[longitude, latitude] in radiance

    Returns
    ----------
    float list
        the result array of the unit 3D coordinates [x, y, z] vector where :math:`x^2 + y^2 + z^2 = 1`

    Raises
    ----------
    RuntimeError
        The input array doesn't have the size of 3.
    """
    if len(node_coord) != 2:
        raise RuntimeError(
            "Input array should have a length of 2: [longitude, latitude]")
    lon = node_coord[0]
    lat = node_coord[1]
    return [np.cos(lon) * np.cos(lat), np.sin(lon) * np.cos(lat), np.sin(lat)]


@njit(cache=ENABLE_JIT_CACHE)
def node_xyz_to_lonlat_rad(node_coord):
    """Calculate the latitude and longitude in radiance for a node represented
    in the [x, y, z] 3D Cartesian coordinates.

    Parameters
    ----------
    node_coord: float list
        3D Cartesian Coordinates [x, y, z] of the node

    Returns
    ----------
    float list
        the result array of longitude and latitude in radian [longitude_rad, latitude_rad]

    Raises
    ----------
    RuntimeError
        The input array doesn't have the size of 3.
    """
    if len(node_coord) != 3:
        raise RuntimeError("Input array should have a length of 3: [x, y, z]")

    [dx, dy, dz] = normalize_in_place(node_coord)
    dx /= np.absolute(dx * dx + dy * dy + dz * dz)
    dy /= np.absolute(dx * dx + dy * dy + dz * dz)
    dz /= np.absolute(dx * dx + dy * dy + dz * dz)

    if np.absolute(dz) < (1.0 - ERROR_TOLERANCE):
        d_lon_rad = math.atan2(dy, dx)
        d_lat_rad = np.arcsin(dz)

        if d_lon_rad < 0.0:
            d_lon_rad += 2.0 * np.pi
    elif dz > 0.0:
        d_lon_rad = 0.0
        d_lat_rad = 0.5 * np.pi
    else:
        d_lon_rad = 0.0
        d_lat_rad = -0.5 * np.pi

    return [d_lon_rad, d_lat_rad]


@njit(cache=ENABLE_JIT_CACHE)
def normalize_in_place(node):
    """Helper function to project an arbitrary node in 3D coordinates [x, y, z]
    on the unit sphere. It uses the `np.linalg.norm` internally to calculate
    the magnitude.

    Parameters
    ----------
    node: float list
        3D Cartesian Coordinates [x, y, z]

    Returns
    ----------
    float list
        the result unit vector [x, y, z] where :math:`x^2 + y^2 + z^2 = 1`

    Raises
    ----------
    RuntimeError
        The input array doesn't have the size of 3.
    """
    if len(node) != 3:
        raise RuntimeError("Input array should have a length of 3: [x, y, z]")
    return list(np.array(node) / np.linalg.norm(np.array(node), ord=2))


def _get_xyz_from_lonlat(node_lon, node_lat):

    # check for units and create Mesh2_node_cart_x/y/z set to grid._ds
    nodes_lon_rad = np.deg2rad(node_lon)
    nodes_lat_rad = np.deg2rad(node_lat)
    nodes_rad = np.stack((nodes_lon_rad, nodes_lat_rad), axis=1)
    nodes_cart = np.asarray(list(map(node_lonlat_rad_to_xyz, list(nodes_rad))))

    return nodes_cart[:, 0], nodes_cart[:, 1], nodes_cart[:, 2]


def _populate_cartesian_xyz_coord(grid):
    """A helper function that populates the xyz attribute in UXarray.Grid._ds.
    This function is called when we need to use the cartesian coordinates for
    each node to do the calculation but the input data only has the
    "Mesh2_node_x" and "Mesh2_node_y" in degree.

    Note
    ----
    In the UXarray, we abide the UGRID convention and make sure the following attributes will always have its
    corresponding units as stated below:

    Mesh2_node_x
     unit:  "degree_east" for longitude
    Mesh2_node_y
     unit:  "degrees_north" for latitude
    Mesh2_node_z
     unit:  "m"
    Mesh2_node_cart_x
     unit:  "m"
    Mesh2_node_cart_y
     unit:  "m"
    Mesh2_node_cart_z
     unit:  "m"
    """

    # Check if the cartesian coordinates are already populated
    if "Mesh2_node_cart_x" in grid._ds.keys():
        return

    # get Cartesian (x, y, z) coordinates from lon/lat
    x, y, z = _get_xyz_from_lonlat(grid.Mesh2_node_x.values,
                                   grid.Mesh2_node_y.values)

    grid._ds["Mesh2_node_cart_x"] = xr.DataArray(
        data=x,
        dims=["nMesh2_node"],
        attrs={
            "standard_name": "cartesian x",
            "units": "m",
        })
    grid._ds["Mesh2_node_cart_y"] = xr.DataArray(
        data=y,
        dims=["nMesh2_node"],
        attrs={
            "standard_name": "cartesian y",
            "units": "m",
        })
    grid._ds["Mesh2_node_cart_z"] = xr.DataArray(
        data=z,
        dims=["nMesh2_node"],
        attrs={
            "standard_name": "cartesian z",
            "units": "m",
        })


def _get_lonlat_from_xyz(x, y, z):
    nodes_cart = np.stack((x, y, z), axis=1).tolist()
    nodes_rad = list(map(node_xyz_to_lonlat_rad, nodes_cart))
    nodes_degree = np.rad2deg(nodes_rad)

    return nodes_degree[:, 0], nodes_degree[:, 1]


def _populate_lonlat_coord(grid):
    """Helper function that populates the longitude and latitude and store it
    into the Mesh2_node_x and Mesh2_node_y. This is called when the input data
    has "Mesh2_node_x", "Mesh2_node_y", "Mesh2_node_z" in meters. Since we want
    "Mesh2_node_x" and "Mesh2_node_y" always have the "degree" units. For more
    details, please read the following.

    Raises
    ------
        RuntimeError
            Mesh2_node_x/y/z are not represented in the cartesian format with the unit 'm'/'meters' when calling this function"

    Note
    ----
    In the UXarray, we abide the UGRID convention and make sure the following attributes will always have its
    corresponding units as stated below:

    Mesh2_node_x
     unit:  "degree_east" for longitude
    Mesh2_node_y
     unit:  "degrees_north" for latitude
    Mesh2_node_z
     unit:  "m"
    Mesh2_node_cart_x
     unit:  "m"
    Mesh2_node_cart_y
     unit:  "m"
    Mesh2_node_cart_z
     unit:  "m"
    """

    # get lon/lat coordinates from Cartesian (x, y, z)
    lon, lat = _get_lonlat_from_xyz(grid.Mesh2_node_cart_x.values,
                                    grid.Mesh2_node_cart_y.values,
                                    grid.Mesh2_node_cart_z.values)

    # populate dataset
    grid._ds["Mesh2_node_x"] = xr.DataArray(
        data=lon,
        dims=["nMesh2_node"],
        attrs={
            "standard_name": "longitude",
            "long_name": "longitude of mesh nodes",
            "units": "degrees_east",
        })
    grid._ds["Mesh2_node_y"] = xr.DataArray(
        data=lat,
        dims=["nMesh2_node"],
        attrs={
            "standard_name": "latitude",
            "long_name": "latitude of mesh nodes",
            "units": "degrees_north",
        })
