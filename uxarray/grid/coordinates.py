import warnings

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


def _populate_centroid_coord(grid, repopulate=False):
    """Finds the centroids using cartesian averaging of faces based off the
    vertices. The centroid is defined as the average of the x, y, z
    coordinates, normalized. This cannot be guaranteed to work on concave
    polygons.

    Parameters
    ----------
    repopulate : bool, optional
        Bool used to turn on/off repopulating the face coordinates of the centroids
    """
    warnings.warn(
        "This cannot be guaranteed to work correctly on concave polygons")

    node_x = grid.Mesh2_node_cart_x.values
    node_y = grid.Mesh2_node_cart_y.values
    node_z = grid.Mesh2_node_cart_z.values
    face_nodes = grid.Mesh2_face_nodes.values
    nNodes_per_face = grid.nNodes_per_face.values

    if "Mesh2_face_x" not in grid._ds or repopulate:
        # Construct the centroids if there are none stored
        if "Mesh2_face_cart_x" not in grid._ds:
            centroid_x, centroid_y, centroid_z = _construct_xyz_centroids(
                node_x, node_y, node_z, face_nodes, nNodes_per_face)

        else:
            # If there are cartesian centroids already use those instead
            centroid_x, centroid_y, centroid_z = grid.Mesh2_face_cart_x, grid.Mesh2_face_cart_y, grid.Mesh2_face_cart_z

        # Convert from xyz to latlon
        centroid_lon, centroid_lat = _get_lonlat_from_xyz(
            centroid_x, centroid_y, centroid_z)
    else:
        # Convert to xyz if there are latlon centroids already stored
        centroid_lon, centroid_lat = grid.Mesh2_face_x.values, grid.Mesh2_face_y.values
        centroid_x, centroid_y, centroid_z = _get_xyz_from_lonlat(
            centroid_lon, centroid_lat)

    if "Mesh2_face_x" not in grid._ds or repopulate:
        # Populate latlon Mesh2_face_xy
        grid._ds["Mesh2_face_x"] = xr.DataArray(
            centroid_lon,
            dims=["nMesh2_face"],
            attrs={"standard_name": "degrees_east"})
        grid._ds["Mesh2_face_y"] = xr.DataArray(
            centroid_lat,
            dims=["nMesh2_face"],
            attrs={"standard_name": "degrees_north"})

    if "Mesh2_face_cart_x" not in grid._ds or repopulate:
        # Populate cartesian coordinates Mesh2_face_cart_xyz
        grid._ds["Mesh2_face_cart_x"] = xr.DataArray(
            centroid_x,
            dims=["nMesh2_face"],
            attrs={"standard_name": "cartesian x"})

        grid._ds["Mesh2_face_cart_y"] = xr.DataArray(
            centroid_y,
            dims=["nMesh2_face"],
            attrs={"standard_name": "cartesian y"})

        grid._ds["Mesh2_face_cart_z"] = xr.DataArray(
            centroid_z,
            dims=["nMesh2_face"],
            attrs={"standard_name": "cartesian z"})


@njit(cache=ENABLE_JIT_CACHE)
def _construct_xyz_centroids(node_x, node_y, node_z, face_nodes,
                             nNodes_per_face):
    """Constructs the xyz centroid coordinate for each face using Cartesian
    Averaging."""
    centroids = np.zeros((3, face_nodes.shape[0]), dtype=np.float64)

    for face_idx, n_max_nodes in enumerate(nNodes_per_face):
        # compute cartesian average
        centroid_x = np.mean(node_x[face_nodes[face_idx, 0:n_max_nodes]])
        centroid_y = np.mean(node_y[face_nodes[face_idx, 0:n_max_nodes]])
        centroid_z = np.mean(node_z[face_nodes[face_idx, 0:n_max_nodes]])

        # normalize coordinates
        centroid_normalized_xyz = normalize_in_place(
            [centroid_x, centroid_y, centroid_z])

        # store xyz
        centroids[0, face_idx] = centroid_normalized_xyz[0]
        centroids[1, face_idx] = centroid_normalized_xyz[1]
        centroids[2, face_idx] = centroid_normalized_xyz[2]

    return centroids[0, :], centroids[1, :], centroids[2, :]
