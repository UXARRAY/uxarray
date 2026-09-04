"""Utilities for carrying coordinates across operations that change dimensions."""

from __future__ import annotations

from typing import Hashable, Iterable, Mapping

import numpy as np
import xarray as xr
import xarray.core.utils as xr_core_utils


def _preserve_valid_coords(
    obj: xr.DataArray | xr.Dataset,
    dropped_dim: str | None = None,
    output_dims: Iterable[Hashable] | None = None,
    exclude: Iterable[Hashable] | None = None,
) -> Mapping[Hashable, xr.DataArray]:
    """Keep only the coordinates that remain valid on the result of an operation.

    Operations such as topological aggregations, zonal and azimuthal means, and
    remapping consume one dimension and replace it with another. Any coordinate
    spanning the consumed dimension no longer matches the output shape and has to
    be dropped, but every other coordinate -- most importantly the leading ones
    such as ``time`` or ``lev`` -- is untouched and must be carried over so that
    label-based indexing keeps working on the result.

    Parameters
    ----------
    obj : xr.DataArray or xr.Dataset
        Object whose coordinates are being filtered.
    dropped_dim : str, optional
        Dimension consumed by the operation. Coordinates spanning it are dropped.
    output_dims : iterable of hashable, optional
        Dimensions present on the result. Coordinates spanning any dimension not
        in this set are dropped. Useful when the operation also removes or
        reshapes dimensions other than ``dropped_dim``.
    exclude : iterable of hashable, optional
        Coordinate names to drop regardless of their dimensions, for cases where
        the caller supplies its own replacement under the same name.

    Returns
    -------
    dict
        Mapping of coordinate name to coordinate, suitable for passing straight
        to the ``coords`` argument of a DataArray or Dataset constructor.
    """
    output_dims = None if output_dims is None else set(output_dims)
    exclude = frozenset() if exclude is None else frozenset(exclude)

    return {
        name: coord
        for name, coord in obj.coords.items()
        if name not in exclude
        and (dropped_dim is None or dropped_dim not in coord.dims)
        and (output_dims is None or set(coord.dims).issubset(output_dims))
    }


def _is_scalar_indexer(ii):
    """returns whether ii is a scalar indexer, e.g. a single integer.
    (Usefulness, e.g.: help to ensure result of isel() will not drop any dims,
    by using something like isel(dim=[ii] if _is_scalar_indexer(ii) else ii).
    """
    if isinstance(ii, slice):
        return False
    else:
        return xr_core_utils.is_scalar(ii)


def _indices1d_from_indexing(xarray_obj, dim, indexer):
    """returns 1D numpy array of indices from applying `indexer` along `dim` of `xarray_obj`
    (which can be a DataArray or Dataset).

    Equivalent: np.arange(xarray_obj.sizes[dim])[indexer].
    But, more efficient, especially for large dim sizes and small indexers.
    (E.g. with size 1e7, indexer=[0,1,2,3], this method is ~20x faster than
    the naive implementation using np.arange (~0.7ms versus ~15ms).)

    `indexer` can be an integer, slice, array-like or DataArray.
    (If scalar, it will be converted to 1D array.)
    """
    if _is_scalar_indexer(indexer):
        indexer = np.array([indexer])
    if dim in xarray_obj.coords:
        xarray_obj = xarray_obj.drop_vars(dim)
    return xarray_obj[dim].isel({dim: indexer}).values
