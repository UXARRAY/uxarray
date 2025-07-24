import numpy as np
import xarray as xr
from numba import njit

from uxarray.constants import INT_DTYPE, INT_FILL_VALUE
from uxarray.conventions import ugrid


def close_face_nodes(face_node_connectivity, n_face, n_max_face_nodes):
    """Closes (``face_node_connectivity``) by inserting the first node index
    after the last non-fill-value node.

    Parameters
    ----------
    face_node_connectivity : np.ndarray
        Connectivity array for constructing a face from its nodes
    n_face : constant
        Number of faces
    n_max_face_nodes : constant
        Max number of nodes that compose a face

    Returns
    ----------
    closed : ndarray
        Closed (padded) face_node_connectivity

    Example
    ----------
    Given face nodes with shape [2 x 5]
        [0, 1, 2, 3, FILL_VALUE]
        [4, 5, 6, 7, 8]
    Pads them to the following with shape [2 x 6]
        [0, 1, 2, 3, 0, FILL_VALUE]
        [4, 5, 6, 7, 8, 4]
    """

    # padding to shape [n_face, n_max_face_nodes + 1]
    closed = np.ones((n_face, n_max_face_nodes + 1), dtype=INT_DTYPE) * INT_FILL_VALUE

    # set all non-paded values to original face nodee values
    closed[:, :-1] = face_node_connectivity.copy()

    # instance of first fill value
    first_fv_idx_2d = np.argmax(closed == INT_FILL_VALUE, axis=1)

    # 2d to 1d index for np.put()
    first_fv_idx_1d = first_fv_idx_2d + ((n_max_face_nodes + 1) * np.arange(0, n_face))

    # column of first node values
    first_node_value = face_node_connectivity[:, 0].copy()

    # insert first node column at occurrence of first fill value
    np.put(closed.ravel(), first_fv_idx_1d, first_node_value)

    return closed


def _replace_fill_values(grid_var, original_fill, new_fill, new_dtype=None):
    """Replaces all instances of the current fill value (``original_fill``) in
    (``grid_var``) with (``new_fill``) and converts to the dtype defined by
    (``new_dtype``)

    Parameters
    ----------
    grid_var : xr.DataArray
        Grid variable to be modified
    original_fill : constant
        Original fill value used in (``grid_var``)
    new_fill : constant
        New fill value to be used in (``grid_var``)
    new_dtype : np.dtype, optional
        New data type to convert (``grid_var``) to

    Returns
    -------
    grid_var : xr.DataArray
        Modified DataArray with updated fill values and dtype
    """

    # Identify fill value locations
    if original_fill is not None and np.isnan(original_fill):
        # For NaN fill values
        fill_val_idx = grid_var.isnull()
        # Temporarily replace NaNs with a placeholder if dtype conversion is needed
        if new_dtype is not None and np.issubdtype(new_dtype, np.floating):
            grid_var = grid_var.fillna(0.0)
        else:
            # Choose an appropriate placeholder for non-floating types
            grid_var = grid_var.fillna(new_fill)
    else:
        # For non-NaN fill values
        fill_val_idx = grid_var == original_fill

    # Convert to the new data type if specified
    if new_dtype is not None and new_dtype != grid_var.dtype:
        grid_var = grid_var.astype(new_dtype)

    # Validate that the new_fill can be represented in the new_dtype
    if new_dtype is not None:
        if np.issubdtype(new_dtype, np.integer):
            int_min = np.iinfo(new_dtype).min
            int_max = np.iinfo(new_dtype).max
            if not (int_min <= new_fill <= int_max):
                raise ValueError(
                    f"New fill value: {new_fill} not representable by integer dtype: {new_dtype}"
                )
        elif np.issubdtype(new_dtype, np.floating):
            if not (
                np.isnan(new_fill)
                or (np.finfo(new_dtype).min <= new_fill <= np.finfo(new_dtype).max)
            ):
                raise ValueError(
                    f"New fill value: {new_fill} not representable by float dtype: {new_dtype}"
                )
        else:
            raise ValueError(f"Data type {new_dtype} not supported for grid variables")

    grid_var = grid_var.where(~fill_val_idx, new_fill)

    return grid_var


def _populate_n_nodes_per_face(grid):
    """Constructs the connectivity variable (``n_nodes_per_face``) and stores
    it within the internal (``Grid._ds``) and through the attribute
    (``Grid.n_nodes_per_face``)."""

    n_nodes_per_face = _build_n_nodes_per_face(
        grid.face_node_connectivity.values, grid.n_face, grid.n_max_face_nodes
    )

    if n_nodes_per_face.ndim == 0:
        # convert scalar value into a [1, 1] array
        n_nodes_per_face = np.expand_dims(n_nodes_per_face, 0)

    # add to internal dataset
    grid._ds["n_nodes_per_face"] = xr.DataArray(
        data=n_nodes_per_face,
        dims=ugrid.N_NODES_PER_FACE_DIMS,
        attrs=ugrid.N_NODES_PER_FACE_ATTRS,
    )


@njit(cache=True)
def _build_n_nodes_per_face(face_nodes, n_face, n_max_face_nodes):
    """Constructs ``n_nodes_per_face``, which contains the number of non-fill-
    value nodes for each face in ``face_node_connectivity``"""

    n_face, n_max_face_nodes = face_nodes.shape
    n_nodes_per_face = np.empty(n_face, dtype=INT_DTYPE)
    for i in range(n_face):
        c = 0
        for j in range(n_max_face_nodes):
            if face_nodes[i, j] != INT_FILL_VALUE:
                c += 1
        n_nodes_per_face[i] = c
    return n_nodes_per_face


def _populate_edge_node_connectivity(grid):
    """Constructs the UGRID connectivity variable (``edge_node_connectivity``)
    and stores it within the internal (``Grid._ds``) and through the attribute
    (``Grid.edge_node_connectivity``)."""

    edge_nodes, inverse_indices, fill_value_mask = _build_edge_node_connectivity(
        grid.face_node_connectivity.values, grid.n_face, grid.n_max_face_nodes
    )

    edge_node_attrs = ugrid.EDGE_NODE_CONNECTIVITY_ATTRS
    edge_node_attrs["inverse_indices"] = inverse_indices

    # add edge_node_connectivity to internal dataset
    grid._ds["edge_node_connectivity"] = xr.DataArray(
        edge_nodes, dims=ugrid.EDGE_NODE_CONNECTIVITY_DIMS, attrs=edge_node_attrs
    )


def _build_edge_node_connectivity(face_nodes, n_face, n_max_face_nodes):
    """Constructs the UGRID connectivity variable (``edge_node_connectivity``)
    and stores it within the internal (``Grid._ds``) and through the attribute
    (``Grid.edge_node_connectivity``).

    Additionally, the attributes (``inverse_indices``) and
    (``fill_value_mask``) are stored for constructing other
    connectivity variables.

    Parameters
    ----------
    repopulate : bool, optional
        Flag used to indicate if we want to overwrite the existed `edge_node_connectivity` and generate a new
        inverse_indices, default is False
    """

    padded_face_nodes = close_face_nodes(face_nodes, n_face, n_max_face_nodes)

    # array of empty edge nodes where each entry is a pair of indices
    edge_nodes = np.empty((n_face * n_max_face_nodes, 2), dtype=INT_DTYPE)

    # first index includes starting node up to non-padded value
    edge_nodes[:, 0] = padded_face_nodes[:, :-1].ravel()

    # second index includes second node up to padded value
    edge_nodes[:, 1] = padded_face_nodes[:, 1:].ravel()

    # sorted edge nodes
    edge_nodes.sort(axis=1)

    # unique edge nodes
    edge_nodes_unique, inverse_indices = np.unique(
        edge_nodes, return_inverse=True, axis=0
    )
    # find all edge nodes that contain a fill value
    fill_value_mask = np.logical_or(
        edge_nodes_unique[:, 0] == INT_FILL_VALUE,
        edge_nodes_unique[:, 1] == INT_FILL_VALUE,
    )

    # all edge nodes that do not contain a fill value
    non_fill_value_mask = np.logical_not(fill_value_mask)
    edge_nodes_unique = edge_nodes_unique[non_fill_value_mask]

    # Update inverse_indices accordingly
    indices_to_update = np.where(fill_value_mask)[0]

    remove_mask = np.isin(inverse_indices, indices_to_update)
    inverse_indices[remove_mask] = INT_FILL_VALUE

    # Compute the indices where inverse_indices exceeds the values in indices_to_update
    indexes = np.searchsorted(indices_to_update, inverse_indices, side="right")
    # subtract the corresponding indexes from `inverse_indices`
    for i in range(len(inverse_indices)):
        if inverse_indices[i] != INT_FILL_VALUE:
            inverse_indices[i] -= indexes[i]

    return edge_nodes_unique, inverse_indices, fill_value_mask


def _populate_edge_face_connectivity(grid):
    """Constructs the UGRID connectivity variable (``edge_node_connectivity``)
    and stores it within the internal (``Grid._ds``) and through the attribute
    (``Grid.edge_node_connectivity``)."""
    edge_faces = _build_edge_face_connectivity(
        grid.face_edge_connectivity.values, grid.n_nodes_per_face.values, grid.n_edge
    )

    grid._ds["edge_face_connectivity"] = xr.DataArray(
        data=edge_faces,
        dims=ugrid.EDGE_FACE_CONNECTIVITY_DIMS,
        attrs=ugrid.EDGE_FACE_CONNECTIVITY_ATTRS,
    )


@njit(cache=True)
def _build_edge_face_connectivity(face_edges, n_nodes_per_face, n_edge):
    """Helper for (``edge_face_connectivity``) construction."""
    edge_faces = np.ones(shape=(n_edge, 2), dtype=face_edges.dtype) * INT_FILL_VALUE

    for face_idx, (cur_face_edges, n_edges) in enumerate(
        zip(face_edges, n_nodes_per_face)
    ):
        # obtain all the edges that make up a face (excluding fill values)
        edges = cur_face_edges[:n_edges]
        for edge_idx in edges:
            if edge_faces[edge_idx, 0] == INT_FILL_VALUE:
                edge_faces[edge_idx, 0] = face_idx
            else:
                edge_faces[edge_idx, 1] = face_idx

    return edge_faces


def _populate_face_edge_connectivity(grid):
    """Constructs the UGRID connectivity variable (``face_edge_connectivity``)
    and stores it within the internal (``Grid._ds``) and through the attribute
    (``Grid.face_edge_connectivity``)."""

    if (
        "edge_node_connectivity" not in grid._ds
        or "inverse_indices" not in grid._ds["edge_node_connectivity"].attrs
    ):
        _populate_edge_node_connectivity(grid)

    face_edges = _build_face_edge_connectivity(
        grid.edge_node_connectivity.attrs["inverse_indices"],
        grid.n_face,
        grid.n_max_face_nodes,
    )

    grid._ds["face_edge_connectivity"] = xr.DataArray(
        data=face_edges,
        dims=ugrid.FACE_EDGE_CONNECTIVITY_DIMS,
        attrs=ugrid.FACE_EDGE_CONNECTIVITY_ATTRS,
    )


def _build_face_edge_connectivity(inverse_indices, n_face, n_max_face_nodes):
    """Helper for (``face_edge_connectivity``) construction."""
    inverse_indices = inverse_indices.reshape(n_face, n_max_face_nodes)
    return inverse_indices


def _populate_node_face_connectivity(grid):
    """Constructs the UGRID connectivity variable (``node_face_connectivity``)
    and stores it within the internal (``Grid._ds``) and through the attribute
    (``Grid.node_face_connectivity``)."""

    node_faces, n_max_faces_per_node = _build_node_faces_connectivity(
        grid.face_node_connectivity.values, grid.n_node
    )

    grid._ds["node_face_connectivity"] = xr.DataArray(
        node_faces,
        dims=ugrid.NODE_FACE_CONNECTIVITY_DIMS,
        attrs=ugrid.NODE_FACE_CONNECTIVITY_ATTRS,
    )


def _build_node_faces_connectivity(face_nodes, n_node):
    """Builds the `Grid.node_faces_connectivity`: integer DataArray of size
    (n_node, n_max_faces_per_node) (optional) A DataArray of indices indicating
    faces that are neighboring each node.

    This function converts the face-node connectivity data into a sparse matrix, and then constructs the node-face
    connectivity by iterating over each node in the mesh and retrieving the set of neighboring faces.

    Raises
    ------
    RuntimeError
        If the Mesh object does not contain a 'face_node_connectivity' variable.
    """

    node_face_conn = {node_i: [] for node_i in range(n_node)}
    for face_i, face_nodes in enumerate(face_nodes):
        for node_i in face_nodes:
            if node_i != INT_FILL_VALUE:
                node_face_conn[node_i].append(face_i)

    n_max_node_faces = -1
    for face_indicies in node_face_conn.values():
        if len(face_indicies) > n_max_node_faces:
            n_max_node_faces = len(face_indicies)

    node_face_connectivity = np.full(
        (n_node, n_max_node_faces), INT_FILL_VALUE, dtype=INT_DTYPE
    )

    for node_idx, face_indices in enumerate(node_face_conn.values()):
        n_faces = len(face_indices)
        node_face_connectivity[node_idx, 0:n_faces] = face_indices

    return node_face_connectivity, n_max_node_faces


def _face_nodes_to_sparse_matrix(dense_matrix: np.ndarray) -> tuple:
    """Converts a given dense matrix connectivity to a sparse matrix format
    where the locations of non fill-value entries are stored using COO
    (coordinate list) standard. It is represented by three arrays: row indices,
    column indices, and non-filled element flags.

    Parameters
    ----------
    dense_matrix : np.ndarray
        The dense matrix to be converted.
    Returns
    -------
    tuple
        A tuple containing three arrays:
        - face_indices : np.ndarray
            Array containing the face indices for each non fill-value element.
        - node_indices : np.ndarray
            Array containing the node indices for each non fill-value element.
        - non_filled_elements_flag : np.ndarray
            Array containing flags indicating if a non fill-value element is present in the corresponding row and column
            index.
    Example
    -------
    >>> face_nodes_conn = np.array(
    ...     [[3, 4, 5, INT_FILL_VALUE], [3, 0, 2, 5], [3, 4, 1, 0], [0, 1, 2, -999]]
    ... )
    >>> face_indices, nodes_indices, non_filled_flag = _face_nodes_to_sparse_matrix(
    ...     face_nodes_conn
    ... )
    >>> face_indices = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3])
    >>> nodes_indices = np.array([3, 4, 5, 3, 0, 2, 5, 3, 4, 1, 0, 0, 1, 2])
    >>> non_filled_flag = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    """
    n_rows, n_cols = dense_matrix.shape
    flattened_matrix = dense_matrix.ravel()
    valid_node_mask = flattened_matrix != INT_FILL_VALUE
    face_indices = np.repeat(np.arange(n_rows), n_cols)[valid_node_mask]
    node_indices = flattened_matrix[valid_node_mask]
    non_filled_element_flags = np.ones(len(node_indices))
    return face_indices, node_indices, non_filled_element_flags


def get_face_node_partitions(n_nodes_per_face):
    """Returns the indices of how to partition `face_node_connectivity` by
    element size."""

    # sort number of nodes per face in ascending order
    n_nodes_per_face_sorted_ind = np.argsort(n_nodes_per_face)

    # unique element sizes and their respective counts
    element_sizes, size_counts = np.unique(n_nodes_per_face, return_counts=True)
    element_sizes_sorted_ind = np.argsort(element_sizes)

    # sort elements by their size
    element_sizes = element_sizes[element_sizes_sorted_ind]
    size_counts = size_counts[element_sizes_sorted_ind]

    # find the index at the point where the geometry changes from one shape to another
    change_ind = np.cumsum(size_counts)
    change_ind = np.concatenate((np.array([0]), change_ind))

    return change_ind, n_nodes_per_face_sorted_ind, element_sizes, size_counts


def _populate_face_face_connectivity(grid):
    """Constructs the UGRID connectivity variable (``face_face_connectivity``)
    and stores it within the internal (``Grid._ds``) and through the attribute
    (``Grid.face_face_connectivity``)."""
    face_face = _build_face_face_connectivity(grid)

    grid._ds["face_face_connectivity"] = xr.DataArray(
        data=face_face,
        dims=ugrid.FACE_FACE_CONNECTIVITY_DIMS,
        attrs=ugrid.FACE_FACE_CONNECTIVITY_ATTRS,
    )


def _build_face_face_connectivity(grid):
    """Returns face-face connectivity."""

    # Dictionary to store each faces adjacent faces
    face_neighbors = {i: [] for i in range(grid.n_face)}

    # Loop through each edge_face and add to the dictionary every face that shares an edge
    for edge_face in grid.edge_face_connectivity.values:
        face1, face2 = edge_face
        if face1 != INT_FILL_VALUE and face2 != INT_FILL_VALUE:
            # Append to each face's dictionary index the opposite face index
            face_neighbors[face1].append(face2)
            face_neighbors[face2].append(face1)

    # Convert to an array and pad it with fill values
    face_face_conn = list(face_neighbors.values())
    face_face_connectivity = [
        np.pad(
            arr, (0, grid.n_max_face_edges - len(arr)), constant_values=INT_FILL_VALUE
        )
        for arr in face_face_conn
    ]

    return face_face_connectivity


def _populate_node_edge_connectivity(grid):
    """Constructs the UGRID connectivity variable (``edge_node_connectivity``)
    and stores it within the internal (``Grid._ds``) and through the attribute
    (``Grid.edge_node_connectivity``)."""
    node_edge_connectivity = _build_node_edge_connectivity(
        grid.edge_node_connectivity.values, grid.n_node
    )

    grid._ds["node_edge_connectivity"] = xr.DataArray(
        data=node_edge_connectivity,
        dims=ugrid.NODE_EDGE_CONNECTIVITY_DIMS,
        attrs=ugrid.NODE_EDGE_CONNECTIVITY_ATTRS,
    )


@njit
def _build_node_edge_connectivity(edge_nodes, n_node):
    """Constructs the Node Edge Connectivity, which stores the indices of the edges that are shared by each node."""
    n_edge, nodes_per_edge = edge_nodes.shape

    # count how many edges touch each node
    counts = np.zeros(n_node, dtype=INT_DTYPE)
    for e in range(n_edge):
        for j in range(nodes_per_edge):
            node = edge_nodes[e, j]
            if node != INT_FILL_VALUE:
                counts[node] += 1

    # find the maximum
    max_edges = 0
    for i in range(n_node):
        if counts[i] > max_edges:
            max_edges = counts[i]

    # allocate output, pad with fill
    node_edge = np.full((n_node, max_edges), INT_FILL_VALUE, dtype=INT_DTYPE)

    ptr = np.zeros(n_node, dtype=INT_DTYPE)

    # fill in
    for e in range(n_edge):
        for j in range(nodes_per_edge):
            node = edge_nodes[e, j]
            if node != INT_FILL_VALUE:
                idx = ptr[node]
                node_edge[node, idx] = e
                ptr[node] += 1

    return node_edge
