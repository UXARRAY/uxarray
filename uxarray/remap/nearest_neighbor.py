from __future__ import annotations

from typing import TYPE_CHECKING, Dict

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
    """TODO" """
    source_tree = source_grid.get_kdtree(coordinates=KDTREE_DIM_MAP[source_dim])
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
    _assert_dimension(destination_dim)

    # Perform remapping on a UxDataset
    ds, is_da, name = _to_dataset(source)

    # Determine which spatial dimensions we need to remap
    dims_to_remap = _get_remap_dims(ds)

    # Build Nearest Neighbor Index Arrays
    indices_map: Dict[str, np.ndarray] = {
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
