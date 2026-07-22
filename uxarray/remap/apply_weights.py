from __future__ import annotations

from os import PathLike

import numpy as np
import xarray as xr

import uxarray.core.dataarray

from .utils import (
    LABEL_TO_COORD,
    SPATIAL_DIMS,
    _assert_dimension,
    _construct_remapped_ds,
    _to_dataset,
)
from .weights import RemapWeights, load_remap_weights


def _get_source_dim(
    da: xr.DataArray,
    weights: RemapWeights,
    source_dim: str | None,
) -> str | None:
    spatial_dims = [dim for dim in da.dims if dim in SPATIAL_DIMS]

    if len(spatial_dims) > 1:
        raise ValueError(
            f"Weight application does not support variables with multiple "
            f"spatial dimensions. Got {spatial_dims!r} for variable {da.name!r}."
        )

    if source_dim is not None:
        if source_dim not in SPATIAL_DIMS:
            raise ValueError(
                f"source_dim {source_dim!r} is not a spatial dimension. "
                f"Expected one of {sorted(SPATIAL_DIMS)}."
            )
        if source_dim not in da.dims:
            return None
        if da.sizes[source_dim] != weights.source_size:
            raise ValueError(
                f"Variable {da.name!r} dimension {source_dim!r} has size "
                f"{da.sizes[source_dim]}, expected {weights.source_size}."
            )
        return source_dim

    matches = [dim for dim in spatial_dims if da.sizes[dim] == weights.source_size]
    return matches[0] if matches else None


def _apply_weights(
    source,
    weights: str | PathLike[str] | xr.Dataset | RemapWeights,
    destination_grid,
    remap_to: str = "faces",
    source_dim: str | None = None,
):
    """Apply a sparse remap operator to UXarray data.

    Dask-backed inputs are remapped lazily: the sparse operator is applied
    blockwise over the leading (non-source) dimensions via ``apply_ufunc``, with
    the source dimension kept in a single chunk. Numpy-backed inputs are applied
    eagerly.
    """
    _assert_dimension(remap_to)

    weights_obj = load_remap_weights(weights)
    destination_dim = LABEL_TO_COORD[remap_to]
    destination_size = destination_grid.sizes[destination_dim]

    if destination_size != weights_obj.destination_size:
        raise ValueError(
            f"Destination grid size for {destination_dim!r} is {destination_size}, "
            f"but weights target size is {weights_obj.destination_size}."
        )

    ds, is_da, name = _to_dataset(source)
    remapped_vars = {}
    remapped_any = False

    for var_name, da in ds.data_vars.items():
        variable_source_dim = _get_source_dim(da, weights_obj, source_dim)
        if variable_source_dim is None:
            remapped_vars[var_name] = da
            continue

        remapped_any = True
        other_dims = [dim for dim in da.dims if dim != variable_source_dim]
        da_t = da.transpose(*other_dims, variable_source_dim)
        if isinstance(da_t.data, np.ndarray):
            # eager: apply the sparse operator directly
            remapped_values = weights_obj._apply(np.asarray(da_t.values))
        else:
            # dask: apply blockwise over the leading dims; the sparse multiply
            # spans the whole source dimension, so keep it in a single chunk
            remapped_values = xr.apply_ufunc(
                weights_obj._apply,
                da_t.chunk({variable_source_dim: -1}),
                input_core_dims=[[variable_source_dim]],
                output_core_dims=[[destination_dim]],
                dask="parallelized",
                output_dtypes=[np.result_type(weights_obj.matrix.dtype, da_t.dtype)],
                dask_gufunc_kwargs={
                    "output_sizes": {destination_dim: weights_obj.destination_size}
                },
            ).data

        other_dims_set = set(other_dims)
        coords = {
            coord_name: coord
            for coord_name, coord in da.coords.items()
            if set(coord.dims).issubset(other_dims_set)
        }
        da_out = uxarray.core.dataarray.UxDataArray(
            remapped_values,
            dims=other_dims + [destination_dim],
            coords=coords,
            name=da.name,
            attrs=da.attrs,
            uxgrid=destination_grid,
        )
        remapped_vars[var_name] = da_out

    if not remapped_any:
        if is_da:
            raise ValueError(
                f"No spatial dimension matched the weight source size {weights_obj.source_size}."
            )
        raise ValueError(
            "No dataset variables matched the supplied weight source size."
        )

    ds_remapped = _construct_remapped_ds(
        source, remapped_vars, destination_grid, remap_to
    )
    return ds_remapped[name] if is_da else ds_remapped
