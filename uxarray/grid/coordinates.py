import xarray as xr
import numpy as np

import warnings

from uxarray.constants import ERROR_TOLERANCE
from uxarray.conventions import ugrid

from typing import Union

from numba import njit


@njit(cache=True)
def _lonlat_rad_to_xyz(
    lon: Union[np.ndarray, float],
    lat: Union[np.ndarray, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts Spherical latitude and longitude coordinates into Cartesian x,
    y, z coordinates."""
    x = np.cos(lon) * np.cos(lat)
    y = np.sin(lon) * np.cos(lat)
    z = np.sin(lat)

    return x, y, z


def _xyz_to_lonlat_rad(
    x: Union[np.ndarray, float],
    y: Union[np.ndarray, float],
    z: Union[np.ndarray, float],
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts Cartesian x, y, z coordinates in Spherical latitude and
    longitude coordinates in degrees.

    Parameters
    ----------
    x : Union[np.ndarray, float]
        Cartesian x coordinates
    y: Union[np.ndarray, float]
        Cartesiain y coordinates
    z: Union[np.ndarray, float]
        Cartesian z coordinates
    normalize: bool
        Flag to select whether to normalize the coordinates

    Returns
    -------
    lon : Union[np.ndarray, float]
        Longitude in radians
    lat: Union[np.ndarray, float]
        Latitude in radians
    """

    if normalize:
        x, y, z = _normalize_xyz(x, y, z)
        denom = np.abs(x * x + y * y + z * z)
        x /= denom
        y /= denom
        z /= denom

    lon = np.arctan2(y, x, dtype=np.float64)
    lat = np.arcsin(z, dtype=np.float64)

    # set longitude range to [0, pi]
    lon = np.mod(lon, 2 * np.pi)

    z_mask = np.abs(z) > 1.0 - ERROR_TOLERANCE

    lat = np.where(z_mask, np.sign(z) * np.pi / 2, lat)
    lon = np.where(z_mask, 0.0, lon)

    return lon, lat


def _xyz_to_lonlat_deg(
    x: Union[np.ndarray, float],
    y: Union[np.ndarray, float],
    z: Union[np.ndarray, float],
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts Cartesian x, y, z coordinates in Spherical latitude and
    longitude coordinates in degrees.

    Parameters
    ----------
    x : Union[np.ndarray, float]
        Cartesian x coordinates
    y: Union[np.ndarray, float]
        Cartesiain y coordinates
    z: Union[np.ndarray, float]
        Cartesian z coordinates
    normalize: bool
        Flag to select whether to normalize the coordinates

    Returns
    -------
    lon : Union[np.ndarray, float]
        Longitude in degrees
    lat: Union[np.ndarray, float]
        Latitude in degrees
    """
    lon_rad, lat_rad = _xyz_to_lonlat_rad(x, y, z, normalize=normalize)

    lon = np.rad2deg(lon_rad)
    lat = np.rad2deg(lat_rad)

    lon = (lon + 180) % 360 - 180
    return lon, lat


def _normalize_xyz(
    x: Union[np.ndarray, float],
    y: Union[np.ndarray, float],
    z: Union[np.ndarray, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalizes a set of Cartesiain coordinates."""
    denom = np.linalg.norm(
        np.asarray(np.array([x, y, z]), dtype=np.float64), ord=2, axis=0
    )

    x_norm = x / denom
    y_norm = y / denom
    z_norm = z / denom
    return x_norm, y_norm, z_norm


def _populate_node_latlon(grid) -> None:
    """Populates the latitude and longitude coordinates of a Grid (`node_lon`,
    `node_lat`)"""
    lon_rad, lat_rad = _xyz_to_lonlat_rad(
        grid.node_x.values, grid.node_y.values, grid.node_z.values
    )

    lon = np.rad2deg(lon_rad)
    lat = np.rad2deg(lat_rad)

    grid._ds["node_lon"] = xr.DataArray(
        data=lon, dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_LON_ATTRS
    )
    grid._ds["node_lat"] = xr.DataArray(
        data=lat, dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_LAT_ATTRS
    )


def _populate_node_xyz(grid) -> None:
    """Populates the Cartesiain node coordinates of a Grid (`node_x`, `node_y`
    and `node_z`)"""

    node_lon_rad = np.deg2rad(grid.node_lon.values)
    node_lat_rad = np.deg2rad(grid.node_lat.values)
    x, y, z = _lonlat_rad_to_xyz(node_lon_rad, node_lat_rad)

    grid._ds["node_x"] = xr.DataArray(
        data=x, dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_X_ATTRS
    )
    grid._ds["node_y"] = xr.DataArray(
        data=y, dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_Y_ATTRS
    )
    grid._ds["node_z"] = xr.DataArray(
        data=z, dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_Z_ATTRS
    )


def _populate_face_centroids(grid, repopulate=False):
    """Finds the centroids of faces using cartesian averaging based off the
    vertices. The centroid is defined as the average of the x, y, z
    coordinates, normalized. This cannot be guaranteed to work on concave
    polygons.

    Parameters
    ----------
    repopulate : bool, optional
        Bool used to turn on/off repopulating the face coordinates of the centroids
    """
    warnings.warn("This cannot be guaranteed to work correctly on concave polygons")

    node_x = grid.node_x.values
    node_y = grid.node_y.values
    node_z = grid.node_z.values
    face_nodes = grid.face_node_connectivity.values
    n_nodes_per_face = grid.n_nodes_per_face.values

    if "face_lon" not in grid._ds or repopulate:
        # Construct the centroids if there are none stored
        if "face_x" not in grid._ds:
            centroid_x, centroid_y, centroid_z = _construct_face_centroids(
                node_x, node_y, node_z, face_nodes, n_nodes_per_face
            )

        else:
            # If there are cartesian centroids already use those instead
            centroid_x, centroid_y, centroid_z = grid.face_x, grid.face_y, grid.face_z

        # Convert from xyz to latlon TODO
        centroid_lon, centroid_lat = _xyz_to_lonlat_deg(
            centroid_x, centroid_y, centroid_z, normalize=False
        )
    else:
        # Convert to xyz if there are latlon centroids already stored
        centroid_lon, centroid_lat = grid.face_lon.values, grid.face_lat.values
        centroid_x, centroid_y, centroid_z = _lonlat_rad_to_xyz(
            centroid_lon, centroid_lat
        )

    # Populate the centroids
    if "face_lon" not in grid._ds or repopulate:
        grid._ds["face_lon"] = xr.DataArray(
            centroid_lon, dims=[ugrid.FACE_DIM], attrs=ugrid.FACE_LON_ATTRS
        )
        grid._ds["face_lat"] = xr.DataArray(
            centroid_lat, dims=[ugrid.FACE_DIM], attrs=ugrid.FACE_LAT_ATTRS
        )

    if "face_x" not in grid._ds or repopulate:
        grid._ds["face_x"] = xr.DataArray(
            centroid_x, dims=[ugrid.FACE_DIM], attrs=ugrid.FACE_X_ATTRS
        )

        grid._ds["face_y"] = xr.DataArray(
            centroid_y, dims=[ugrid.FACE_DIM], attrs=ugrid.FACE_Y_ATTRS
        )

        grid._ds["face_z"] = xr.DataArray(
            centroid_z, dims=[ugrid.FACE_DIM], attrs=ugrid.FACE_Z_ATTRS
        )


def _construct_face_centroids(node_x, node_y, node_z, face_nodes, n_nodes_per_face):
    """Constructs the xyz centroid coordinate for each face using Cartesian
    Averaging."""

    centroid_x = np.zeros((face_nodes.shape[0]), dtype=np.float64)
    centroid_y = np.zeros((face_nodes.shape[0]), dtype=np.float64)
    centroid_z = np.zeros((face_nodes.shape[0]), dtype=np.float64)

    for face_idx, n_max_nodes in enumerate(n_nodes_per_face):
        # Compute Cartesian Average
        centroid_x[face_idx] = np.mean(node_x[face_nodes[face_idx, 0:n_max_nodes]])
        centroid_y[face_idx] = np.mean(node_y[face_nodes[face_idx, 0:n_max_nodes]])
        centroid_z[face_idx] = np.mean(node_z[face_nodes[face_idx, 0:n_max_nodes]])

    return _normalize_xyz(centroid_x, centroid_y, centroid_z)


def _populate_edge_centroids(grid, repopulate=False):
    """Finds the centroids using cartesian averaging of the edges based off the
    vertices. The centroid is defined as the average of the x, y, z
    coordinates, normalized.

    Parameters
    ----------
    repopulate : bool, optional
        Bool used to turn on/off repopulating the edge coordinates of the centroids
    """

    node_x = grid.node_x.values
    node_y = grid.node_y.values
    node_z = grid.node_z.values
    edge_nodes_con = grid.edge_node_connectivity.values

    if "edge_lon" not in grid._ds or repopulate:
        # Construct the centroids if there are none stored
        if "edge_x" not in grid._ds:
            centroid_x, centroid_y, centroid_z = _construct_edge_centroids(
                node_x, node_y, node_z, edge_nodes_con
            )

        else:
            # If there are cartesian centroids already use those instead
            centroid_x, centroid_y, centroid_z = grid.edge_x, grid.edge_y, grid.edge_z

        # Convert from xyz to latlon
        centroid_lon, centroid_lat = _xyz_to_lonlat_deg(
            centroid_x, centroid_y, centroid_z, normalize=False
        )
    else:
        # Convert to xyz if there are latlon centroids already stored
        centroid_lon, centroid_lat = grid.edge_lon.values, grid.edge_lat.values
        centroid_x, centroid_y, centroid_z = _lonlat_rad_to_xyz(
            centroid_lon, centroid_lat
        )

    # Populate the centroids
    if "edge_lon" not in grid._ds or repopulate:
        grid._ds["edge_lon"] = xr.DataArray(
            centroid_lon, dims=[ugrid.EDGE_DIM], attrs=ugrid.EDGE_LON_ATTRS
        )
        grid._ds["edge_lat"] = xr.DataArray(
            centroid_lat,
            dims=[ugrid.EDGE_DIM],
            attrs=ugrid.EDGE_LAT_ATTRS,
        )

    if "edge_x" not in grid._ds or repopulate:
        grid._ds["edge_x"] = xr.DataArray(
            centroid_x,
            dims=[ugrid.EDGE_DIM],
            attrs=ugrid.EDGE_X_ATTRS,
        )

        grid._ds["edge_y"] = xr.DataArray(
            centroid_y,
            dims=[ugrid.EDGE_DIM],
            attrs=ugrid.EDGE_Y_ATTRS,
        )

        grid._ds["edge_z"] = xr.DataArray(
            centroid_z,
            dims=[ugrid.EDGE_DIM],
            attrs=ugrid.EDGE_Z_ATTRS,
        )


def _construct_edge_centroids(node_x, node_y, node_z, edge_node_conn):
    """Constructs the xyz centroid coordinate for each edge using Cartesian
    Averaging."""

    centroid_x = np.mean(node_x[edge_node_conn], axis=1)
    centroid_y = np.mean(node_y[edge_node_conn], axis=1)
    centroid_z = np.mean(node_z[edge_node_conn], axis=1)

    return _normalize_xyz(centroid_x, centroid_y, centroid_z)


def _set_desired_longitude_range(ds):
    """Sets the longitude range to [-180, 180] for all longitude variables."""

    for lon_name in ["node_lon", "edge_lon", "face_lon"]:
        if lon_name in ds:
            if ds[lon_name].max() > 180:
                ds[lon_name] = (ds[lon_name] + 180) % 360 - 180
