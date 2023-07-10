import numpy as np
import xarray as xr

from uxarray.utils.constants import INT_DTYPE, INT_FILL_VALUE


def close_face_nodes(Mesh2_face_nodes, nMesh2_face, nMaxMesh2_face_nodes):
    """Closes (``Mesh2_face_nodes``) by inserting the first node index after
    the last non-fill-value node.

    Parameters
    ----------
    Mesh2_face_nodes : np.ndarray
        Connectivity array for constructing a face from its nodes
    nMesh2_face : constant
        Number of faces
    nMaxMesh2_face_nodes : constant
        Max number of nodes that compose a face

    Returns
    ----------
    closed : ndarray
        Closed (padded) Mesh2_face_nodes

    Example
    ----------
    Given face nodes with shape [2 x 5]
        [0, 1, 2, 3, FILL_VALUE]
        [4, 5, 6, 7, 8]
    Pads them to the following with shape [2 x 6]
        [0, 1, 2, 3, 0, FILL_VALUE]
        [4, 5, 6, 7, 8, 4]
    """

    # padding to shape [nMesh2_face, nMaxMesh2_face_nodes + 1]
    closed = np.ones((nMesh2_face, nMaxMesh2_face_nodes + 1),
                     dtype=INT_DTYPE) * INT_FILL_VALUE

    # set all non-paded values to original face nodee values
    closed[:, :-1] = Mesh2_face_nodes.copy()

    # instance of first fill value
    first_fv_idx_2d = np.argmax(closed == INT_FILL_VALUE, axis=1)

    # 2d to 1d index for np.put()
    first_fv_idx_1d = first_fv_idx_2d + (
        (nMaxMesh2_face_nodes + 1) * np.arange(0, nMesh2_face))

    # column of first node values
    first_node_value = Mesh2_face_nodes[:, 0].copy()

    # insert first node column at occurrence of first fill value
    np.put(closed.ravel(), first_fv_idx_1d, first_node_value)

    return closed


def _build_nNodes_per_face(grid):
    """Constructs ``nNodes_per_face``, which contains the number of non- fill-
    value nodes for each face in ``Mesh2_face_nodes``"""

    # padding to shape [nMesh2_face, nMaxMesh2_face_nodes + 1]
    closed = np.ones((grid.nMesh2_face, grid.nMaxMesh2_face_nodes + 1),
                     dtype=INT_DTYPE) * INT_FILL_VALUE

    closed[:, :-1] = grid.Mesh2_face_nodes.copy()

    nNodes_per_face = np.argmax(closed == INT_FILL_VALUE, axis=1)

    # add to internal dataset
    grid._ds["nNodes_per_face"] = xr.DataArray(
        data=nNodes_per_face,
        dims=["nMesh2_face"],
        attrs={"long_name": "number of non-fill value nodes for each face"})


def _build_edge_node_connectivity(grid, repopulate=False):
    """Constructs the UGRID connectivity variable (``Mesh2_edge_nodes``) and
    stores it within the internal (``Grid._ds``) and through the attribute
    (``Grid.Mesh2_edge_nodes``).

    Additionally, the attributes (``inverse_indices``) and
    (``fill_value_mask``) are stored for constructing other
    connectivity variables.

    Parameters
    ----------
    repopulate : bool, optional
        Flag used to indicate if we want to overwrite the existed `Mesh2_edge_nodes` and generate a new
        inverse_indices, default is False
    """

    # need to derive edge nodes
    if "Mesh2_edge_nodes" not in grid._ds or repopulate:
        padded_face_nodes = close_face_nodes(grid.Mesh2_face_nodes.values,
                                             grid.nMesh2_face,
                                             grid.nMaxMesh2_face_nodes)

        # array of empty edge nodes where each entry is a pair of indices
        edge_nodes = np.empty((grid.nMesh2_face * grid.nMaxMesh2_face_nodes, 2),
                              dtype=INT_DTYPE)

        # first index includes starting node up to non-padded value
        edge_nodes[:, 0] = padded_face_nodes[:, :-1].ravel()

        # second index includes second node up to padded value
        edge_nodes[:, 1] = padded_face_nodes[:, 1:].ravel()
    else:
        # If "Mesh2_edge_nodes" already exists, directly return the function call
        return

    # sorted edge nodes
    edge_nodes.sort(axis=1)

    # unique edge nodes
    edge_nodes_unique, inverse_indices = np.unique(edge_nodes,
                                                   return_inverse=True,
                                                   axis=0)
    # find all edge nodes that contain a fill value
    fill_value_mask = np.logical_or(edge_nodes_unique[:, 0] == INT_FILL_VALUE,
                                    edge_nodes_unique[:, 1] == INT_FILL_VALUE)

    # all edge nodes that do not contain a fill value
    non_fill_value_mask = np.logical_not(fill_value_mask)
    edge_nodes_unique = edge_nodes_unique[non_fill_value_mask]

    # Update inverse_indices accordingly
    indices_to_update = np.where(fill_value_mask)[0]

    remove_mask = np.isin(inverse_indices, indices_to_update)
    inverse_indices[remove_mask] = INT_FILL_VALUE

    # Compute the indices where inverse_indices exceeds the values in indices_to_update
    indexes = np.searchsorted(indices_to_update, inverse_indices, side='right')
    # subtract the corresponding indexes from `inverse_indices`
    for i in range(len(inverse_indices)):
        if inverse_indices[i] != INT_FILL_VALUE:
            inverse_indices[i] -= indexes[i]

    # add Mesh2_edge_nodes to internal dataset
    grid._ds['Mesh2_edge_nodes'] = xr.DataArray(
        edge_nodes_unique,
        dims=["nMesh2_edge", "Two"],
        attrs={
            "cf_role": "edge_node_connectivity",
            "_FillValue": INT_FILL_VALUE,
            "long_name": "Maps every edge to the two nodes that it connects",
            "start_index": INT_DTYPE(0),
            "inverse_indices": inverse_indices,
            "fill_value_mask": fill_value_mask
        })


def _build_face_edges_connectivity(grid):
    """Constructs the UGRID connectivity variable (``Mesh2_face_edges``) and
    stores it within the internal (``Grid._ds``) and through the attribute
    (``Grid.Mesh2_face_edges``)."""
    if ("Mesh2_edge_nodes" not in grid._ds or
            "inverse_indices" not in grid._ds['Mesh2_edge_nodes'].attrs):
        _build_edge_node_connectivity(grid, repopulate=True)

    inverse_indices = grid._ds['Mesh2_edge_nodes'].inverse_indices
    inverse_indices = inverse_indices.reshape(grid.nMesh2_face,
                                              grid.nMaxMesh2_face_nodes)

    grid._ds["Mesh2_face_edges"] = xr.DataArray(
        data=inverse_indices,
        dims=["nMesh2_face", "nMaxMesh2_face_edges"],
        attrs={
            "cf_role": "face_edges_connectivity",
            "start_index": INT_DTYPE(0),
            "long_name": "Maps every edge to the two nodes that it connects",
        })
