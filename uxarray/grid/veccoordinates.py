import xarray as xr
import numpy as np

from numba import config

from uxarray.constants import ENABLE_JIT, ERROR_TOLERANCE
from uxarray.conventions import ugrid

config.DISABLE_JIT = not ENABLE_JIT


# ======================================================================================================================
# Conversion Functions
# ======================================================================================================================
def _lonlat_rad_to_xyz(
    lon: np.ndarray, lat: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Docstring TODO."""
    x = np.cos(lon) * np.cos(lat)
    y = np.sin(lon) * np.cos(lat)
    z = np.sin(lat)
    return x, y, z


def _xyz_to_lonlat_rad(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Docstring TODO."""
    denom = np.abs(x * x + y * y + z * z)

    x /= denom
    y /= denom
    z /= denom

    lon = np.empty_like(x)
    lat = np.empty_like(x)

    # is there a way to vectorize this loop?
    for i in range(len(x)):
        if np.absolute(z[i]) < (1.0 - ERROR_TOLERANCE):
            lon[i] = np.arctan2(y[i], x[i])
            lat[i] = np.arcsin(z[i])

            if lon[i] < 0.0:
                lon[i] += 2.0 * np.pi
        elif z[i] > 0.0:
            lon[i] = 0.0
            lat[i] = 0.5 * np.pi
        else:
            lon[i] = 0.0
            lat[i] = -0.5 * np.pi

    return lon, lat


# don't use numba here unless we figure out ord=2
def _normalize_xyz(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Docstring TODO."""
    x_norm = x / np.linalg.norm(x, ord=2)
    y_norm = y / np.linalg.norm(y, ord=2)
    z_norm = z / np.linalg.norm(z, ord=2)
    return x_norm, y_norm, z_norm


# ======================================================================================================================
# Population Functions
# ======================================================================================================================
def _populate_node_latlon(grid) -> None:
    """Docstring TODO."""
    x_norm, y_norm, z_norm = _normalize_xyz(
        grid.node_x.values, grid.node_y.values, grid.node_z.values
    )
    lon_rad, lat_rad = _xyz_to_lonlat_rad(x_norm, y_norm, z_norm)

    # Convert to degrees
    lon = np.rad2deg(lon_rad)
    lat = np.rad2deg(lat_rad)

    grid._ds["node_lon"] = xr.DataArray(
        data=lon, dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_LON_ATTRS
    )
    grid._ds["node_lat"] = xr.DataArray(
        data=lat, dims=[ugrid.NODE_DIM], attrs=ugrid.NODE_LAT_ATTRS
    )


def _populate_edge_latlon(grid) -> None:
    pass


def _populate_face_latlon(grid) -> None:
    pass


def _populate_node_xyz(grid) -> None:
    """Docstring TODO."""

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


def _populate_edge_xyz(grid) -> None:
    pass


def _populate_face_xyz(grid) -> None:
    pass


# ======================================================================================================================
# Build Functions
# ======================================================================================================================
def _build_face_centroids():
    pass


def _build_edge_centroids():
    pass
