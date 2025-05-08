from __future__ import annotations

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
    """
    Compute inverse-distance weights for IDW interpolation.

    Parameters
    ----------
    distances : np.ndarray
        Array of distances to the k nearest neighbors for each query point,
        with shape (n_points, k).
    power : int
        Exponent controlling distance decay. Larger values reduce the influence
        of more distant neighbors.

    Returns
    -------
    weights : np.ndarray
        Normalized weights with the same shape as `distances`, where each row sums to 1.
    """
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
    """
    Apply inverse-distance-weighted (IDW) remapping to a UXarray object.

    Each value on the destination grid is computed as a weighted average of
    the k nearest source points, with weights inversely related to distance.

    Parameters
    ----------
    source : UxDataArray or UxDataset
        The data to be remapped.
    destination_grid : Grid
        The UXarray grid instance on which to interpolate data.
    destination_dim : {'n_node', 'n_edge', 'n_face'}, default='n_face'
        The spatial dimension on `destination_grid` to receive interpolated values.
    power : int, default=2
        Exponent in the inverse-distance weighting function. Larger values
        emphasize closer neighbors.
    k : int, default=8
        Number of nearest neighbors to include in the weighted average.

    Returns
    -------
    UxDataArray or UxDataset
        A new UXarray object with values interpolated onto `destination_grid`.

    Notes
    -----
    - If `k == 1`, falls back to nearest-neighbor remapping.
    - IDW remapping can blur sharp gradients and does not conserve integrated quantities.
    """
    # Fall back onto nearest neighbor
    if k == 1:
        return _nearest_neighbor_remap(source, destination_grid, destination_dim)

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
