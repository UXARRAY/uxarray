import numpy as np
import xarray as xr
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE
from uxarray.helpers import close_face_nodes


def _build_edge_nodes(grid):
    """Constructs the UGRID connectivity variable (``Mesh2_edge_nodes``) and
    stores it within the internal (``Grid.ds``) and through the attribute
    (``Grid.Mesh2_edge_nodes``).

    Additionally, the attributes (``inverse_indices``) and
    (``fill_value_mask``) are stored for constructing other
    connectivity variables.

    Parameters
    ----------
    grid : ux.Grid
        Source Grid for Connectivity Construction
    """
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
    grid.ds['Mesh2_edge_nodes'] = xr.DataArray(
        edge_nodes_unique,
        dims=["nMesh2_edge", "Two"],
        attrs={
            "cf_role": "edge_node_connectivity",
            "long_name": "Maps every edge to the two nodes that it connects",
            "start_index": INT_DTYPE(0),
            "inverse_indices": inverse_indices,
            "fill_value_mask": fill_value_mask
        })

    # set standardized attributes
    setattr(grid, "Mesh2_edge_nodes", grid.ds['Mesh2_edge_nodes'])
    setattr(grid, "nMesh2_edge", edge_nodes_unique.shape[0])


def _build_face_dimension(grid):
    """Constructs (``Mesh2_face_dimension``), which defines the dimension of
    each face in (``Mesh2_face_nodes``)

    Parameters
    ----------
    grid : ux.Grid
        Source Grid for Connectivity Construction
    """

    # Triangular Mesh
    if not hasattr(grid, "nMaxMesh2_face_nodes"):
        nMaxMesh2_face_nodes = grid.Mesh2_face_nodes.shape[1]
        setattr(grid, "nMaxMesh2_face_nodes", nMaxMesh2_face_nodes)

    # padding to shape [nMesh2_face, nMaxMesh2_face_nodes + 1]
    closed = np.ones((grid.nMesh2_face, grid.nMaxMesh2_face_nodes + 1),
                     dtype=INT_DTYPE) * INT_FILL_VALUE

    closed[:, :-1] = grid.Mesh2_face_nodes.copy()

    face_dimension = np.argmax(closed == INT_FILL_VALUE, axis=1)

    # add to internal dataset
    grid.ds["Mesh2_face_dimension"] = xr.DataArray(
        data=face_dimension,
        dims=["nMesh2_face"],
        attrs={"long_name": "number of non-fill value nodes for each face"})

    # standardized attribute
    setattr(grid, "Mesh2_face_dimension", grid.ds["Mesh2_face_dimension"])
