import healpix as hp
import xarray as xr
import numpy as np
from typing import Any, Tuple

import uxarray.conventions.ugrid as ugrid
from uxarray.grid.coordinates import _set_desired_longitude_range
from uxarray.constants import INT_DTYPE


def pix2corner_ang(
    nside: int, ipix: Any, nest: bool = False, lonlat: bool = False
) -> Tuple[Any, Any, Any, Any]:
    if nest:
        fu = hp._chp.nest2ang_uv
    else:
        fu = hp._chp.ring2ang_uv
    right = fu(nside, ipix, 1, 0)
    top = fu(nside, ipix, 1, 1)
    left = fu(nside, ipix, 0, 1)
    bottom = fu(nside, ipix, 0, 0)
    corners = [right, top, left, bottom]
    if lonlat:
        corners = [hp.lonlat_from_thetaphi(*x) for x in corners]
    corners = np.array(corners)
    if len(corners.shape) == 3:
        corners = corners.transpose((2, 0, 1))
    return np.nan_to_num(corners, copy=False)


def _pixels_to_ugrid(resolution_level, nest):
    ds = xr.Dataset()

    nside = hp.order2nside(resolution_level)
    npix = hp.nside2npix(nside)

    hp_lon, hp_lat = hp.pix2ang(
        nside=nside, ipix=np.arange(npix), lonlat=True, nest=nest
    )
    hp_lon = (hp_lon + 180) % 360 - 180

    ds["face_lon"] = xr.DataArray(hp_lon, dims=["n_face"], attrs=ugrid.FACE_LON_ATTRS)
    ds["face_lat"] = xr.DataArray(hp_lat, dims=["n_face"], attrs=ugrid.FACE_LAT_ATTRS)

    ds = ds.assign_attrs({"resolution_level": resolution_level})
    ds = ds.assign_attrs({"n_side": nside})
    ds = ds.assign_attrs({"n_pix": npix})
    ds = ds.assign_attrs({"nest": nest})

    return ds


def _populate_healpix_boundaries(ds):
    """Compute node locations and face-node connectivity for HEALPix."""
    n_side = ds.attrs["n_side"]
    n_pix = ds.attrs["n_pix"]
    nest = ds.attrs["nest"]

    corners = pix2corner_ang(n_side, np.arange(n_pix), nest=nest, lonlat=True)
    nodes = corners.reshape(-1, 2)  # Flattened shape: (n_cell * 4, 2)

    # Extract unique nodes
    unique_nodes, inverse_indices = np.unique(nodes, axis=0, return_inverse=True)

    # Extract node longitude and latitude
    node_lon, node_lat = unique_nodes[:, 0], unique_nodes[:, 1]

    # Reshape inverse indices to get face-node connectivity
    n_cell = corners.shape[0]
    face_node_connectivity = inverse_indices.reshape(n_cell, 4).astype(INT_DTYPE)

    # Assign data to xarray dataset
    ds["node_lon"] = xr.DataArray(node_lon, dims=["n_node"], attrs=ugrid.NODE_LON_ATTRS)
    ds["node_lat"] = xr.DataArray(node_lat, dims=["n_node"], attrs=ugrid.NODE_LAT_ATTRS)
    ds["face_node_connectivity"] = xr.DataArray(
        face_node_connectivity,
        dims=ugrid.FACE_NODE_CONNECTIVITY_DIMS,
        attrs=ugrid.FACE_NODE_CONNECTIVITY_ATTRS,
    )

    _set_desired_longitude_range(ds)
