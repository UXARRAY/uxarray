from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from uxarray.core.dataarray import UxDataArray
    from uxarray.core.dataset import UxDataset
    from uxarray.grid.grid import Grid

from uxarray.grid import Grid
from uxarray.remap.nearest_neighbor import (
    _nearest_neighbor_query,
    _nearest_neighbor_remap,
)

from .utils import (
    LABEL_TO_COORD,
    SPATIAL_DIMS,
    _assert_dimension,
    _construct_remapped_ds,
    _get_remap_dims,
    _to_dataset,
)


def _idw_weights(distances, power):
    weights = 1.0 / (distances**power + 1e-6)
    weights /= np.sum(weights, axis=1, keepdims=True)
    return weights


def _inverse_distance_weighted_remap(
    source: UxDataArray | UxDataset,
    destination_grid: Grid,
    destination_dim: str = "n_face",
    power: int = 2,
    k: int = 8,
):
    # Fall back onto nearest neighbor
    if k == 1:
        return _nearest_neighbor_remap(source, destination_grid, destination_dim)

    if k > source.shape[-1]:
        k = source.shape[-1]
        warnings.warn(f"k is greater than the total number of elements, setting k={k}")

    _assert_dimension(destination_dim)

    # Perform remapping on a UxDataset
    ds, is_da, name = _to_dataset(source)

    # Determine which spatial dimensions we need to remap
    dims_to_remap = _get_remap_dims(ds)

    indices_weights_map = {}

    # Build Nearest Neighbor Index & Weight Arrays
    for src_dim in dims_to_remap:
        indices, distances = _nearest_neighbor_query(
            ds.uxgrid,
            destination_grid,
            src_dim,
            destination_dim,
            k=k,
            return_distances=True,
        )

        weights = _idw_weights(distances, power)

        indices_weights_map[src_dim] = (indices, weights)

    remapped_vars = {}
    for name, da in ds.data_vars.items():
        spatial = set(da.dims) & SPATIAL_DIMS
        if spatial:
            source_dim = spatial.pop()
            inds, w = indices_weights_map[source_dim]

            # pack indices & weights into tiny DataArrays:
            indexer = xr.DataArray(inds, dims=[LABEL_TO_COORD[destination_dim], "k"])
            weight_da = xr.DataArray(w, dims=[LABEL_TO_COORD[destination_dim], "k"])

            # gather the k neighbor values:
            da_k = da.isel({source_dim: indexer}, ignore_grid=True)

            # weighted sum over the small "k" axis:
            da_idw = (da_k * weight_da).sum(dim="k")

            # attach the new grid
            da_idw.uxgrid = destination_grid
            remapped_vars[name] = da_idw
        else:
            remapped_vars[name] = da

    ds_remapped = _construct_remapped_ds(
        source, remapped_vars, destination_grid, destination_dim
    )

    return ds_remapped[name] if is_da else ds_remapped
