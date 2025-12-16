from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    from uxarray.core.dataarray import UxDataArray
    from uxarray.core.dataset import UxDataset
    from uxarray.grid import Grid


import numpy as np

from uxarray.grid import Grid

from .utils import (
    KDTREE_DIM_MAP,
    LABEL_TO_COORD,
    SPATIAL_DIMS,
    _assert_dimension,
    _construct_remapped_ds,
    _get_remap_dims,
    _prepare_points,
    _to_dataset,
)


def _nearest_neighbor_query(
    source_grid: Grid,
    destination_grid: Grid,
    source_dim: str,
    destination_dim: str,
    k: int = 1,
    return_distances: bool = False,
):
    """
    Query the nearest neighbors from a source grid for specified destination points.

    Builds or retrieves a KDTree on the source grid, then finds the k nearest source
    points for each destination location and optionally returns their distances.

    Parameters
    ----------
    source_grid : Grid
        The grid providing source points for interpolation.
    destination_grid : Grid
        The grid defining query points where values will be interpolated.
    source_dim : str
        The spatial dimension on the source grid (e.g., 'n_node', 'n_edge', 'n_face').
    destination_dim : str
        The spatial dimension on the destination grid to query against.
    k : int, default=1
        Number of nearest neighbors to retrieve for each destination point.
    return_distances : bool, default=False
        If True, return a tuple (indices, distances), otherwise return indices only.

    Returns
    -------
    indices : numpy.ndarray
        Array of shape (n_points, k) with indices of the nearest source points.
    distances : numpy.ndarray, optional
        Distances to the nearest source points, returned only if `return_distances` is True.
    """
    source_tree = source_grid._get_scipy_kd_tree(coordinates=KDTREE_DIM_MAP[source_dim])
    destination_points = _prepare_points(destination_grid, destination_dim)
    distances, nearest_indices = source_tree.query(destination_points, k=k, workers=-1)

    if return_distances:
        return nearest_indices, distances
    else:
        return nearest_indices


def _nearest_neighbor_remap(
    source: UxDataArray | UxDataset,
    destination_grid: Grid,
    destination_dim: str = "n_face",
):
    """
    Apply nearest-neighbor remapping from a UXarray object onto another grid.

    Each value in the destination grid is assigned the value of the closest source point.

    Parameters
    ----------
    source : UxDataArray or UxDataset
        The data array or dataset to be remapped.
    destination_grid : Grid
        The UXarray Grid instance to which data will be remapped.
    destination_dim : str, default='n_face'
        The spatial dimension on the destination grid ('n_node', 'n_edge', 'n_face').

    Returns
    -------
    UxDataArray or UxDataset
        A new UXarray object with data values assigned to `destination_grid`.
    """
    _assert_dimension(destination_dim)

    # Perform remapping on a UxDataset
    ds, is_da, name = _to_dataset(source)

    # Determine which spatial dimensions we need to remap
    dims_to_remap = _get_remap_dims(ds)

    # Build Nearest Neighbor Index Arrays
    indices_map: dict[str, np.ndarray] = {
        src_dim: _nearest_neighbor_query(
            ds.uxgrid, destination_grid, src_dim, destination_dim
        )
        for src_dim in dims_to_remap
    }
    remapped_vars = {}

    for name, da in ds.data_vars.items():
        spatial_keys = set(da.dims) & SPATIAL_DIMS
        if spatial_keys:
            source_dim = spatial_keys.pop()

            indices = indices_map[source_dim]
            indexer = xr.DataArray(
                indices,
                dims=[
                    LABEL_TO_COORD[destination_dim],
                ],
            )

            da_remap = da.isel({source_dim: indexer}, ignore_grid=True)

            da_remap.uxgrid = destination_grid
            remapped_vars[name] = da_remap
        else:
            remapped_vars[name] = da

    ds_remapped = _construct_remapped_ds(
        source, remapped_vars, destination_grid, destination_dim
    )

    return ds_remapped[name] if is_da else ds_remapped
