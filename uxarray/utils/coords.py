"""Utilities for carrying coordinates across operations that change dimensions."""

from __future__ import annotations

from typing import Hashable, Iterable, Mapping

import xarray as xr


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
