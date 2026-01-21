from typing import Any

import healpix as hp
import numpy as np
import polars as pl
import xarray as xr

import uxarray.conventions.ugrid as ugrid
from uxarray.constants import INT_DTYPE


def get_zoom_from_cells(cells):
    """
    Compute the HEALPix zoom level n such that cells == 12 * 4**n.
    Only global HEALPix grids (i.e. exactly 12 * 4**n cells) are supported.
    Raises ValueError with detailed HEALPix guidance otherwise.
    """
    if not isinstance(cells, int) or cells < 12:
        raise ValueError(
            f"Invalid cells={cells!r}: a global HEALPix grid must have "
            f"an integer number of cells ≥ 12 (12 base pixels at zoom=0)."
        )

    if cells % 12 != 0:
        raise ValueError(
            f"Invalid cells={cells}: global HEALPix grids have exactly 12 * 4**n cells"
        )

    power = cells // 12
    zoom = 0

    while power > 1 and power % 4 == 0:
        power //= 4
        zoom += 1

    if power != 1:
        raise ValueError(
            f"Invalid cells={cells}: no integer n satisfies cells==12 * 4**n. "
            f"Only global HEALPix grids (with cells = 12 × 4^n) are supported."
        )

    return zoom


def pix2corner_ang(
    nside: int, ipix: Any, nest: bool = False, lonlat: bool = False
) -> tuple[Any, Any, Any, Any]:
    """
    Computes the corner coordinates for one or more HEALPix pixels.

    Parameters
    ----------
    nside : int
        HEALPix nside parameter.
    ipix : Any
        Pixel index or indices.
    nest : bool, optional
        If True, use nested pixel ordering; otherwise, use ring ordering. Default is False.
    lonlat : bool, optional
        If True, convert the angular coordinates to (longitude, latitude). Default is False.

    Returns
    -------
    Tuple[Any, Any, Any, Any]
        A tuple containing the corner coordinates for each pixel.

    Note
    ----
    This will be updated when https://github.com/ntessore/healpix/issues/66 is implemented.
    """
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


def _pixels_to_ugrid(zoom, nest):
    """
    Constructs an xarray Dataset representing a HEALPix grid with face coordinates.

    Parameters
    ----------
    zoom : int
        HEALPix zoom level.
    nest : bool
        If True, use nested pixel ordering; if False, use ring ordering.

    Returns
    -------
    xr.Dataset
        A dataset containing pixel longitude and latitude coordinates along with related attributes.

    """
    ds = xr.Dataset()

    nside = hp.order2nside(zoom)
    npix = hp.nside2npix(nside)

    hp_lon, hp_lat = hp.pix2ang(
        nside=nside, ipix=np.arange(npix), lonlat=True, nest=nest
    )
    hp_lon = (hp_lon + 180) % 360 - 180

    ds["face_lon"] = xr.DataArray(hp_lon, dims=["n_face"], attrs=ugrid.FACE_LON_ATTRS)
    ds["face_lat"] = xr.DataArray(hp_lat, dims=["n_face"], attrs=ugrid.FACE_LAT_ATTRS)

    ds = ds.assign_attrs({"zoom": zoom})
    ds = ds.assign_attrs({"n_side": nside})
    ds = ds.assign_attrs({"n_pix": npix})
    ds = ds.assign_attrs({"nest": nest})

    return ds


def _populate_healpix_boundaries(ds):
    """
    Populates an xarray Dataset with HEALPix grid cell boundaries.

    Parameters
    ----------
    ds : xr.Dataset
        A dataset representing a HEALPix grid in the UGRID conventions.

    """
    # Get corners of all the pixels at once

    n_side = ds.attrs["n_side"]
    n_pix = ds.attrs["n_pix"]
    nest = ds.attrs["nest"]

    # Pass 'nest' to pix2corner_ang!
    corners = pix2corner_ang(n_side, np.arange(n_pix), nest=nest, lonlat=True)

    # Flatten the cell corner data to a 2D array of shape (n_cell * 4, 2)
    nodes = corners.reshape(-1, 2)  # Shape: (n_cell * 4, 2)

    # Normalize
    nodes[:, 0] = (nodes[:, 0] + 180) % 360 - 180

    nodes_df = pl.DataFrame(nodes)
    unique_nodes_df = nodes_df.unique()

    # Add a unique index to `unique_nodes_df`
    unique_nodes_with_index = unique_nodes_df.with_row_index("unique_index")

    # Perform a left join on all columns
    merged = nodes_df.join(
        unique_nodes_with_index, on=list(nodes_df.columns), how="left"
    )

    # Extract the inverse indices
    inverse_indices_df = merged.select("unique_index")

    unique_nodes_df_np = unique_nodes_df.to_numpy()
    inverse_indices_df_np = inverse_indices_df.to_numpy()

    # Extract node longitude and latitude arrays
    node_lon = unique_nodes_df_np[:, 0]
    node_lat = unique_nodes_df_np[:, 1]

    # Reshape inverse indices to get face-node connectivity
    n_cell = corners.shape[0]
    face_node_connectivity = inverse_indices_df_np.reshape(n_cell, 4).astype(
        dtype=INT_DTYPE
    )

    ds["node_lon"] = xr.DataArray(node_lon, dims=["n_node"], attrs=ugrid.NODE_LON_ATTRS)
    ds["node_lat"] = xr.DataArray(node_lat, dims=["n_node"], attrs=ugrid.NODE_LAT_ATTRS)

    ds["face_node_connectivity"] = xr.DataArray(
        face_node_connectivity,
        dims=ugrid.FACE_NODE_CONNECTIVITY_DIMS,
        attrs=ugrid.FACE_NODE_CONNECTIVITY_ATTRS,
    )


def _compute_healpix_face_areas(grid_ds):
    """
    Compute theoretical equal face areas for HEALPix grids.

    HEALPix grids are designed to have exactly equal area pixels by construction.
    This function returns the theoretical equal areas rather than computing them
    geometrically, which preserves the fundamental equal-area property.

    Parameters
    ----------
    grid_ds : xr.Dataset
        A dataset representing a HEALPix grid with 'n_face' dimension.

    Returns
    -------
    xr.DataArray
        An array of theoretical equal face areas with shape (n_face,).

    Notes
    -----
    For HEALPix grids, all pixels have exactly the same area by design:
    area_per_pixel = 4π / n_pixels

    This approach ensures mathematical correctness for area-weighted calculations
    such as global averaging and zonal means, avoiding systematic errors that
    can arise from geometric integration of Great Circle Arc boundaries.
    """
    from uxarray.conventions.descriptors import FACE_AREAS_ATTRS, FACE_AREAS_DIMS

    # Get number of faces
    n_face = grid_ds.sizes["n_face"]

    # Compute theoretical equal area per pixel
    theoretical_area = 4.0 * np.pi / n_face
    face_areas = np.full(n_face, theoretical_area)

    # HEALPix-specific attributes
    healpix_attrs = {
        **FACE_AREAS_ATTRS,
        "long_name": "HEALPix equal area per face",
        "comment": "Theoretical equal areas enforced for HEALPix grids",
    }

    return xr.DataArray(data=face_areas, dims=FACE_AREAS_DIMS, attrs=healpix_attrs)
