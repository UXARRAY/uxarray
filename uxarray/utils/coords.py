"""Utilities for carrying coordinates across operations that change dimensions."""

from __future__ import annotations

from typing import Hashable, Iterable, Mapping

import numpy as np
import xarray as xr
import xarray.core.coordinates
import xarray.core.utils as xr_core_utils

from uxarray.errors import DimensionError


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


def _assign_grid_dim_indexer_coords_if_appropriate(uxarray_obj, grid_dim, indexer):
    """returns uxarray_obj but with coords assigned from indexer if appropriate.
    "appropriate" `indexer` is a 0D or 1D xr.DataArray and has any relevant coordinates to assign.
    2D+ xr.DataArray indexers are not supported for grid dimensions and cause DimensionError here.
    All other indexers do not provide coordinate info, so uxarray_obj gets returned unchanged.

    For 0D indexer, just assign indexer.coords if nonempty (else, return uxarray_obj unchanged).
    For 1D indexer, depends on grid_dim and uxarray_obj.
        if grid_dim=="n_face" and "n_face" in uxarray_obj:
            swap indexer's 1 dim to be "n_face" instead of its original name,
            then assign indexer.coords.
        in all other cases:
            drop indexer's 1 dim, then assign indexer.coords if nonempty.
            (For "n_edge" and "n_node" indexing, the result's shape won't necessarily
            match the indexer's shape, so coords along the grid dim can't be assigned.
            Meanwhile, if grid_dim not in uxarray_obj, it is impossible to assign coords
            along that dim, so once again, coords along the grid dim can't be assigned.)
        Note: if indexer is booleans, instead use indexer.isel(indexer_dim=indexer).coords,
            because the result will only keep values wherever indexer value is True.
            (Also in this case, if indexer.to_xarray() exists, call it, to avoid recursion.)
    """
    if isinstance(indexer, xr.DataArray):
        xr.core.coordinates.assert_coordinate_consistent(uxarray_obj, indexer.coords)
        # ^ e.g. if uxarray_obj has time dim but indexer has time scalar coord, crash!
        if indexer.ndim == 0:
            coords = indexer.coords
        elif indexer.ndim == 1:
            the_dim = indexer.dims[0]
            if grid_dim == "n_face" and "n_face" in uxarray_obj.dims:
                if indexer.dtype == bool:
                    if hasattr(indexer, "to_xarray"):
                        indexer = indexer.to_xarray()
                    indexer = indexer.isel({the_dim: indexer})
                coords = indexer.swap_dims({the_dim: "n_face"}).coords
            else:
                # remove any 1D coords (but keep scalar coords)
                coords = indexer.isel({the_dim: 0}, drop=True).coords
        else:
            raise DimensionError(
                f"2D+ indexers are not supported for grid dimensions. Got xr.DataArray "
                f"indexer with ndim={indexer.ndim}, along grid_dim={grid_dim!r}."
            )
        if coords:
            return uxarray_obj.assign_coords(coords)
    return uxarray_obj
