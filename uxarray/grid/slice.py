from __future__ import annotations

import numpy as np
import xarray as xr
from uxarray.constants import INT_FILL_VALUE

from uxarray.grid import Grid
import polars as pl
from numba import njit, types
from numba.typed import Dict

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@njit(cache=True)
def update_connectivity(conn, indices_dict, fill_value):
    dim_a, dim_b = conn.shape
    conn_flat = conn.flatten()
    result = np.empty_like(conn_flat)

    for i in range(len(conn_flat)):
        if conn_flat[i] == fill_value:
            result[i] = fill_value
        else:
            # Use the dictionary to find the new index
            result[i] = indices_dict.get(conn_flat[i], fill_value)

    return result.reshape(dim_a, dim_b)


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

    # faces that saddle nodes given in 'indices'
    face_indices = np.unique(grid.node_face_connectivity.values[indices].ravel())
    face_indices = face_indices[face_indices != INT_FILL_VALUE]

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

    # faces that saddle nodes given in 'indices'
    face_indices = np.unique(grid.edge_face_connectivity.values[indices].ravel())
    face_indices = face_indices[face_indices != INT_FILL_VALUE]

    return _slice_face_indices(grid, face_indices)


def _slice_face_indices(grid, indices):
    ds = grid._ds

    if hasattr(indices, "ndim") and indices.ndim == 0:
        # Handle scalar numpy array case
        face_indices = [indices.item()]
    elif np.isscalar(indices):
        # Handle Python scalar case
        face_indices = [indices]
    else:
        # Already array-like
        face_indices = indices

    # Identify node indices from face_node_connectivity
    face_node_connectivity = grid.face_node_connectivity.isel(
        n_face=face_indices
    ).values
    node_indices = (
        pl.DataFrame(face_node_connectivity.flatten()).unique().to_numpy().squeeze()
    )
    node_indices = node_indices[node_indices > 0]

    # Prepare indexers and source indices
    indexers = {"n_node": node_indices, "n_face": face_indices}

    source_indices = {
        "source_node_indices": xr.DataArray(node_indices, dims=["n_node"]),
        "source_face_indices": xr.DataArray(face_indices, dims=["n_face"]),
    }

    # Identify edge indicies from face_edge_connectivity and prepare indexers and source indicies
    if "n_edge" in ds.dims:
        face_edge_connectivity = grid.face_edge_connectivity.isel(
            n_face=face_indices
        ).values
        edge_indices = (
            pl.DataFrame(face_edge_connectivity.flatten()).unique().to_numpy()
        )
        edge_indices = edge_indices[edge_indices > 0]
        indexers["n_edge"] = edge_indices
        source_indices["source_edge_indices"] = xr.DataArray(
            edge_indices, dims=["n_edge"]
        )

    ds = ds.isel(indexers)
    ds = ds.assign(source_indices)

    # Update existing connectivity to match valid indices
    conn_names = ds.data_vars
    node_conn_names = [
        conn_name for conn_name in conn_names if "_node_connectivity" in conn_name
    ]
    edge_conn_names = [
        conn_name for conn_name in conn_names if "_edge_connectivity" in conn_name
    ]
    face_conn_names = [
        conn_name for conn_name in conn_names if "_face_connectivity" in conn_name
    ]

    if node_conn_names:
        node_indices_dict = Dict.empty(key_type=types.int64, value_type=types.int64)
        for new_idx, old_idx in enumerate(node_indices):
            node_indices_dict[old_idx] = new_idx

        for conn_name in node_conn_names:
            ds[conn_name].data = update_connectivity(
                ds[conn_name].values, node_indices_dict, INT_FILL_VALUE
            )

    if edge_conn_names:
        edge_indices_dict = Dict.empty(key_type=types.int64, value_type=types.int64)
        for new_idx, old_idx in enumerate(edge_indices):
            edge_indices_dict[old_idx] = new_idx

        for conn_name in edge_conn_names:
            ds[conn_name].data = update_connectivity(
                ds[conn_name].values, edge_indices_dict, INT_FILL_VALUE
            )

    if face_conn_names:
        ds = ds.drop_vars(face_conn_names)

    return Grid.from_dataset(ds, source_grid_spec=grid.source_grid_spec, is_subset=True)
