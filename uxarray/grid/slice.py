from __future__ import annotations

import numpy as np
import xarray as xr

from uxarray.constants import INT_DTYPE, INT_FILL_VALUE
from uxarray.errors import DimensionError
from uxarray.utils.coords import _indices1d_from_indexing, _is_scalar_indexer

# A dense lookup table remaps in O(1) per element, but costs O(n_node) / O(n_edge)
# memory that is also captured by the Dask graph. It is only built when it stays
# within this memory budget, or when it is within this factor of the subset being
# created anyway.
#
# The binary search only becomes faster than the dense table once the source grid
# is a few hundred times larger than the slice. The ratio below is deliberately
# well under that crossover.
_DENSE_REMAP_MAX_SIZE = 8_000_000 // np.dtype(INT_DTYPE).itemsize
_DENSE_REMAP_MAX_RATIO = 32


def _remap_dense(conn, remap):
    """Remaps a connectivity array through a dense ``original -> new`` lookup
    table."""
    # blockwise-safe: mask fill values before the lookup, restore them after
    is_fill = conn == INT_FILL_VALUE
    return np.where(is_fill, INT_FILL_VALUE, remap[np.where(is_fill, 0, conn)])


def _remap_searchsorted(conn, orig_indices):
    """Remaps a connectivity array by locating each original index within the
    sorted array of selected indices."""
    if orig_indices.size == 0:
        return np.full(conn.shape, INT_FILL_VALUE, dtype=INT_DTYPE)

    pos = np.searchsorted(orig_indices, conn)
    np.clip(pos, 0, orig_indices.size - 1, out=pos)

    # entries that aren't part of the slice (fill values included) have no match
    return np.where(orig_indices[pos] == conn, pos, INT_FILL_VALUE).astype(
        INT_DTYPE, copy=False
    )


def _remap_kernel(orig_indices, size):
    """Prepares a blockwise-safe kernel that maps each original element index
    onto its position in ``orig_indices``, or to ``INT_FILL_VALUE`` if it isn't
    part of the slice.

    Parameters
    ----------
    orig_indices : np.ndarray
        Sorted, unique indices of the elements kept by the slice
    size : int
        Number of elements along that dimension in the source grid

    Returns
    -------
    tuple
        The kernel and the keyword arguments to call it with
    """
    n_selected = len(orig_indices)

    if size <= _DENSE_REMAP_MAX_SIZE or size <= _DENSE_REMAP_MAX_RATIO * n_selected:
        remap = np.full(size, INT_FILL_VALUE, dtype=INT_DTYPE)
        remap[orig_indices] = np.arange(n_selected, dtype=INT_DTYPE)
        return _remap_dense, {"remap": remap}

    return _remap_searchsorted, {"orig_indices": orig_indices}


def _slice_node_indices(grid, indices):
    """Slices (indexes) an unstructured grid given a list/array of node
    indices, returning a new Grid composed of elements that contain the nodes
    specified in the indices.

    Parameters
    ----------
    grid : ux.Grid
        Source unstructured grid
    indices: integer, slice, array-like or DataArray
        Indicates which node(s) to select. 2D+ indexers are not supported.
    """
    # faces that saddle nodes given in 'indices'
    face_indices = np.unique(
        grid.node_face_connectivity.isel(n_node=indices).values.ravel()
    )
    face_indices = face_indices[face_indices != INT_FILL_VALUE]

    return _slice_face_indices(grid, face_indices)


def _slice_edge_indices(grid, indices):
    """Slices (indexes) an unstructured grid given a list/array of edge
    indices, returning a new Grid composed of elements that contain the edges
    specified in the indices.

    Parameters
    ----------
    grid : ux.Grid
        Source unstructured grid
    indices: integer, slice, array-like or DataArray
        Indicates which edge(s) to select. 2D+ indexers are not supported.
    """
    # faces that saddle nodes given in 'indices'
    face_indices = np.unique(
        grid.edge_face_connectivity.isel(n_edge=indices).values.ravel()
    )
    face_indices = face_indices[face_indices != INT_FILL_VALUE]

    return _slice_face_indices(grid, face_indices)


def _slice_face_indices(
    grid,
    indices,
    *,
    inverse_indices: list[str] | set[str] | bool = False,
):
    """Slices (indexes) an unstructured grid given a list/array of face
    indices, returning a new Grid composed of elements that contain the faces
    specified in the indices.

    Parameters
    ----------
    grid : ux.Grid
        Source unstructured grid
    indices: integer, slice, array-like or DataArray
        Indicates which face(s) to select. 2D+ indexers are not supported.
    inverse_indices : list[str] | set[str] | bool, optional
        Indicates whether to store the original grids indices. Passing `True` stores the original face centers,
        other reverse indices can be stored by passing any or all of the following: (["face", "edge", "node"], True)
    """
    from uxarray.grid import Grid

    ds = grid._ds
    if _is_scalar_indexer(indices):
        face_indexer = np.array([indices])
    else:
        face_indexer = indices

    if getattr(face_indexer, "ndim", 0) > 1:
        raise DimensionError(
            "Only 1D indexers are supported when slicing a grid. "
            f"Got indexer.ndim={face_indexer.ndim} for indexer of type {type(face_indexer)}."
        )

    # face_indices from the original ds
    # (but still use face_indexer inside isel() below, for efficiency)
    face_indices = _indices1d_from_indexing(ds, "n_face", face_indexer)

    # nodes of each face (inclusive)
    node_indices = np.unique(
        grid.face_node_connectivity.isel(n_face=face_indexer).values.ravel()
    )
    node_indices = node_indices[node_indices != INT_FILL_VALUE]

    # Index Node and Face variables
    ds = ds.isel(n_node=node_indices)
    ds = ds.isel(n_face=face_indexer)

    # Only slice edge dimension if we have the face edge connectivity
    if "face_edge_connectivity" in ds:
        edge_indices = np.unique(
            grid.face_edge_connectivity.isel(n_face=face_indexer).values.ravel()
        )
        edge_indices = edge_indices[edge_indices != INT_FILL_VALUE]
        ds = ds.isel(n_edge=edge_indices)
        ds["_subgrid_edge_indices"] = xr.DataArray(edge_indices, dims=["n_edge"])
    # Otherwise, drop any edge variables
    else:
        if "n_edge" in ds.dims:
            ds = ds.drop_dims(["n_edge"])
        edge_indices = None

    ds["_subgrid_node_indices"] = xr.DataArray(node_indices, dims=["n_node"])
    ds["_subgrid_face_indices"] = xr.DataArray(face_indices, dims=["n_face"])
    # _subgrid_..._indices are used internaly in _slice_from_grid.
    # Caution: if performing multiple isel() operations, these indices map from the
    # latest operation only, not all the way back to the original grid.
    # E.g. grid.isel(n_face=[7,8,9]).isel(n_face=[0]) --> _subgrid_face_indices=[0], not [7].

    # `node_indices` and `edge_indices` come out of `np.unique`, so both kernels
    # can rely on them being sorted and free of duplicates
    node_remap = _remap_kernel(node_indices, grid.n_node)
    edge_remap = (
        _remap_kernel(edge_indices, grid.n_edge) if edge_indices is not None else None
    )

    for conn_name in list(ds.data_vars):
        if conn_name.endswith("_node_connectivity"):
            remap_func, remap_kwargs = node_remap

        elif conn_name.endswith("_edge_connectivity"):
            if edge_remap is None:
                ds = ds.drop_vars(conn_name)
                continue
            remap_func, remap_kwargs = edge_remap

        elif "_connectivity" in conn_name:
            # anything else we can't remap
            ds = ds.drop_vars(conn_name)
            continue

        else:
            # not a connectivity var, skip
            continue

        # Apply remapping (vectorized; stays lazy when the connectivity is dask)
        ds[conn_name] = xr.apply_ufunc(
            remap_func,
            ds[conn_name],
            kwargs=remap_kwargs,
            dask="parallelized",
            output_dtypes=[INT_DTYPE],
            keep_attrs=True,
        )

    if inverse_indices:
        inverse_indices_ds = xr.Dataset()

        index_types = {
            "face": face_indices,
            "node": node_indices,
        }

        if edge_indices is not None:
            index_types["edge"] = edge_indices

        if isinstance(inverse_indices, bool):
            inverse_indices_ds["face"] = face_indices
        else:
            for index_type in inverse_indices[0]:
                if index_type in index_types:
                    inverse_indices_ds[index_type] = index_types[index_type]
                else:
                    raise ValueError(
                        "Incorrect type of index for `inverse_indices`. Try passing one of the following "
                        "instead: 'face', 'edge', 'node'"
                    )

        return Grid.from_dataset(
            ds,
            source_grid_spec=grid.source_grid_spec,
            is_subset=True,
            inverse_indices=inverse_indices_ds,
        )

    return Grid.from_dataset(ds, source_grid_spec=grid.source_grid_spec, is_subset=True)
