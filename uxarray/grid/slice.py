from __future__ import annotations

import numpy as np
import xarray as xr
from uxarray.constants import INT_FILL_VALUE

from uxarray.grid import Grid
import polars as pl
from numba import njit, types, prange
from numba.typed import Dict

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@njit(
    "int64[:,:](int64[:,:], DictType(int64, int64), int64)", cache=True, parallel=True
)
def update_connectivity(conn, indices_dict, fill_value):
    dim_a, dim_b = conn.shape
    conn_flat = conn.flatten()
    result = np.empty_like(conn_flat)

    for i in prange(len(conn_flat)):
        if conn_flat[i] == fill_value:
            result[i] = fill_value
        else:
            # Use the dictionary to find the new index
            result[i] = indices_dict.get(conn_flat[i], fill_value)

    return result.reshape(dim_a, dim_b)


@njit(
    "DictType(int64, int64)(int64[:])",
    cache=True,
)
def create_indices_dict(indices):
    indices_dict = Dict.empty(key_type=types.int64, value_type=types.int64)
    for new_idx, old_idx in enumerate(indices):
        indices_dict[old_idx] = new_idx
    return indices_dict


def _slice_node_indices(
    grid,
    indices,
    inclusive=True,
):
    """Slices (indexes) an unstructured grid given a list/array of node
    indices, returning a new Grid composed of elements that contain the nodes
    specified in the indices.

    Parameters
    ----------
    grid : ux.Grid
        Source unstructured grid
    indices: array-like
        A list or 1-D array of node indices
    inclusive: bool
        Whether to perform inclusive (i.e. elements must contain at least one desired feature from a slice) as opposed
        to exclusive (i.e elements be made up all desired features from a slice)
    """

    if inclusive is False:
        raise ValueError("Exclusive slicing is not yet supported.")

    node_indices = _validate_indices(indices)

    # Identify face indices from edge_face_connectivity
    node_face_connectivity = grid.node_face_connectivity.isel(
        n_node=node_indices
    ).values
    face_indices = (
        pl.DataFrame(node_face_connectivity.flatten()).unique().to_numpy().squeeze()
    )
    face_indices = face_indices[face_indices >= 0]

    return _slice_face_indices(grid, face_indices)


def _slice_edge_indices(
    grid,
    indices,
    inclusive=True,
):
    """Slices (indexes) an unstructured grid given a list/array of edge
    indices, returning a new Grid composed of elements that contain the edges
    specified in the indices.

    Parameters
    ----------
    grid : ux.Grid
        Source unstructured grid
    indices: array-like
        A list or 1-D array of edge indices
    inclusive: bool
        Whether to perform inclusive (i.e. elements must contain at least one desired feature from a slice) as opposed
        to exclusive (i.e elements be made up all desired features from a slice)
    """

    if inclusive is False:
        raise ValueError("Exclusive slicing is not yet supported.")

    edge_indices = _validate_indices(indices)

    # Identify face indices from edge_face_connectivity
    edge_face_connectivity = grid.edge_face_connectivity.isel(
        n_edge=edge_indices
    ).values
    face_indices = (
        pl.DataFrame(edge_face_connectivity.flatten()).unique().to_numpy().squeeze()
    )
    face_indices = face_indices[face_indices >= 0]

    return _slice_face_indices(grid, face_indices)


def _slice_face_indices(grid, indices):
    ds = grid._ds

    face_indices = _validate_indices(indices)

    # Identify node indices from face_node_connectivity
    face_node_connectivity = grid.face_node_connectivity.isel(
        n_face=face_indices
    ).values
    node_indices = (
        pl.DataFrame(face_node_connectivity.flatten()).unique().to_numpy().squeeze()
    )
    node_indices = node_indices[node_indices >= 0]

    # Prepare indexers and source indices
    indexers = {"n_node": node_indices, "n_face": face_indices}

    source_indices = {
        "source_node_indices": xr.DataArray(node_indices, dims=["n_node"]),
        "source_face_indices": xr.DataArray(face_indices, dims=["n_face"]),
    }

    # Identify edge indices from face_edge_connectivity and prepare indexers and source indices
    if "n_edge" in ds.dims:
        face_edge_connectivity = grid.face_edge_connectivity.isel(
            n_face=face_indices
        ).values
        edge_indices = (
            pl.DataFrame(face_edge_connectivity.flatten()).unique().to_numpy()
        )
        edge_indices = edge_indices[edge_indices >= 0]
        indexers["n_edge"] = edge_indices
        source_indices["source_edge_indices"] = xr.DataArray(
            edge_indices, dims=["n_edge"]
        )

    ds = ds.isel(indexers)
    ds = ds.assign(source_indices)

    # Update existing connectivity to match valid indices
    node_conn_names = [
        conn_name for conn_name in ds.data_vars if "_node_connectivity" in conn_name
    ]
    edge_conn_names = [
        conn_name for conn_name in ds.data_vars if "_edge_connectivity" in conn_name
    ]
    face_conn_names = [
        conn_name for conn_name in ds.data_vars if "_face_connectivity" in conn_name
    ]

    # Update Node Connectivity Variables
    if node_conn_names:
        node_indices_dict = create_indices_dict(node_indices)

        for conn_name in node_conn_names:
            ds[conn_name].data = update_connectivity(
                ds[conn_name].values, node_indices_dict, INT_FILL_VALUE
            )

    # Update Edge Connectivity Variables
    if edge_conn_names:
        edge_indices_dict = create_indices_dict(edge_indices)

        for conn_name in edge_conn_names:
            ds[conn_name].data = update_connectivity(
                ds[conn_name].values, edge_indices_dict, INT_FILL_VALUE
            )

    # Update Face Connectivity Variables (TODO)
    if face_conn_names:
        ds = ds.drop_vars(face_conn_names)

    return Grid.from_dataset(ds, source_grid_spec=grid.source_grid_spec, is_subset=True)


def _validate_indices(indices):
    if hasattr(indices, "ndim") and indices.ndim == 0:
        # Handle scalar numpy array case
        return [indices.item()]
    elif np.isscalar(indices):
        # Handle Python scalar case
        return [indices]
    else:
        # Already array-like
        return indices
