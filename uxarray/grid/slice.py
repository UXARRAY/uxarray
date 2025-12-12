from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from uxarray.constants import INT_DTYPE, INT_FILL_VALUE

if TYPE_CHECKING:
    pass


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


def _slice_face_indices(
    grid,
    indices,
    inclusive=True,
    inverse_indices: list[str] | set[str] | bool = False,
):
    """Slices (indexes) an unstructured grid given a list/array of face
    indices, returning a new Grid composed of elements that contain the faces
    specified in the indices.

    Parameters
    ----------
    grid : ux.Grid
        Source unstructured grid
    indices: array-like
        A list or 1-D array of face indices
    inclusive: bool
        Whether to perform inclusive (i.e. elements must contain at least one desired feature from a slice) as opposed
        to exclusive (i.e elements be made up all desired features from a slice)
    inverse_indices : list[str] | set[str] | bool, optional
        Indicates whether to store the original grids indices. Passing `True` stores the original face centers,
        other reverse indices can be stored by passing any or all of the following: (["face", "edge", "node"], True)
    """
    from uxarray.grid import Grid

    if inclusive is False:
        raise ValueError("Exclusive slicing is not yet supported.")

    ds = grid._ds
    face_indices = np.atleast_1d(np.asarray(indices, dtype=INT_DTYPE))

    # nodes of each face (inclusive)
    node_indices = np.unique(grid.face_node_connectivity.values[face_indices].ravel())
    node_indices = node_indices[node_indices != INT_FILL_VALUE]

    # Index Node and Face variables
    ds = ds.isel(n_node=node_indices)
    ds = ds.isel(n_face=face_indices)

    # Only slice edge dimension if we have the face edge connectivity
    if "face_edge_connectivity" in ds:
        edge_indices = np.unique(
            grid.face_edge_connectivity.values[face_indices].ravel()
        )
        edge_indices = edge_indices[edge_indices != INT_FILL_VALUE]
        ds = ds.isel(n_edge=edge_indices)
        ds["subgrid_edge_indices"] = xr.DataArray(edge_indices, dims=["n_edge"])
    # Otherwise, drop any edge variables
    else:
        if "n_edge" in ds.dims:
            ds = ds.drop_dims(["n_edge"])
        edge_indices = None

    ds["subgrid_node_indices"] = xr.DataArray(node_indices, dims=["n_node"])
    ds["subgrid_face_indices"] = xr.DataArray(face_indices, dims=["n_face"])

    # Construct updated Node Index Map
    node_indices_dict = {orig: new for new, orig in enumerate(node_indices)}
    node_indices_dict[INT_FILL_VALUE] = INT_FILL_VALUE

    # Construct updated Edge Index Map
    if edge_indices is not None:
        edge_indices_dict = {orig: new for new, orig in enumerate(edge_indices)}
        edge_indices_dict[INT_FILL_VALUE] = INT_FILL_VALUE
    else:
        edge_indices_dict = None

    def map_node_indices(i):
        return node_indices_dict.get(i, INT_FILL_VALUE)

    if edge_indices is not None:

        def map_edge_indices(i):
            return edge_indices_dict.get(i, INT_FILL_VALUE)
    else:
        map_edge_indices = None

    for conn_name in list(ds.data_vars):
        if conn_name.endswith("_node_connectivity"):
            map_fn = map_node_indices

        elif conn_name.endswith("_edge_connectivity"):
            if edge_indices_dict is None:
                ds = ds.drop_vars(conn_name)
                continue
            map_fn = map_edge_indices

        elif "_connectivity" in conn_name:
            # anything else we can't remap
            ds = ds.drop_vars(conn_name)
            continue

        else:
            # not a connectivity var, skip
            continue

        # Apply Remapping
        ds[conn_name] = xr.DataArray(
            np.vectorize(map_fn, otypes=[INT_DTYPE])(ds[conn_name].values),
            dims=ds[conn_name].dims,
            attrs=ds[conn_name].attrs,
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
