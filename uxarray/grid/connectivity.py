import numpy as np
import xarray as xr
from numba import njit, prange

from uxarray.constants import INT_DTYPE, INT_FILL_VALUE
from uxarray.conventions import ugrid
from uxarray.grid.utils import (
    _build_pair_index,
    _count_unique_in_bucket,
    _search_bucket,
    _sort_bucket,
)


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

    n_nodes_per_face = xr.apply_ufunc(
        _build_n_nodes_per_face,
        grid.face_node_connectivity,
        input_core_dims=[[ugrid.N_MAX_FACE_NODES_DIM]],
        dask="parallelized",
        output_dtypes=[INT_DTYPE],
    )

    # add to internal dataset
    grid._ds["n_nodes_per_face"] = xr.DataArray(
        data=n_nodes_per_face.data,
        dims=ugrid.N_NODES_PER_FACE_DIMS,
        attrs=ugrid.N_NODES_PER_FACE_ATTRS,
    )


@njit(cache=True, nogil=True)
def _build_n_nodes_per_face(face_nodes):
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

    # Check edge coordinates already exist, if they do this might cause issues

    if "n_edge" in grid.dims:
        stale = sorted(n for n in grid._ds if ugrid.EDGE_DIM in grid._ds[n].dims)
        raise ValueError(
            f"Constructing 'edge_node_connectivity' on a grid that already has "
            f"grid variables indexed by the edge dimension ({', '.join(stale)}). "
            f"Constructed edges are numbered in lexicographic order by node pair, which "
            f"may not match the original order of the edges those variables were stored with."
        )

    # This is in lieu of an xarray equivalent to `da.compute(a, b)`. We traverse the
    # grid once to gather both variables, possibly as chunks if dask is enabled
    computed = xr.Dataset(
        {
            "face_nodes": grid.face_node_connectivity.variable,
            "n_nodes_per_face": grid.n_nodes_per_face.variable,
        }
    ).compute()

    edge_nodes, face_edges = _build_edge_node_connectivity(
        computed.face_nodes.data, computed.n_nodes_per_face.data, grid.n_node
    )

    grid._ds["edge_node_connectivity"] = xr.DataArray(
        edge_nodes,
        dims=ugrid.EDGE_NODE_CONNECTIVITY_DIMS,
        attrs=ugrid.EDGE_NODE_CONNECTIVITY_ATTRS,
    )

    grid._ds["face_edge_connectivity"] = xr.DataArray(
        face_edges,
        dims=ugrid.FACE_EDGE_CONNECTIVITY_DIMS,
        attrs=ugrid.FACE_EDGE_CONNECTIVITY_ATTRS,
    )


@njit(cache=True, inline="always")
def _canonical_half_edge(face_node_connectivity, face_idx, local_idx, n_edges):
    """The ``(low, high)`` node pair of the half edge leaving face slot ``local_idx``,
    wrapping back to slot 0 after ``n_edges``."""
    start_node = face_node_connectivity[face_idx, local_idx]
    end_node = face_node_connectivity[face_idx, (local_idx + 1) % n_edges]

    if start_node > end_node:
        return end_node, start_node
    return start_node, end_node


@njit(cache=True)
def _count_half_edges_per_node(face_node_connectivity, n_nodes_per_face, n_node):
    """Bucket offsets keyed on each half edge's lower node: bucket ``a`` will occupy
    ``[bucket_offset[a], bucket_offset[a + 1])``."""
    bucket_offset = np.zeros(n_node + 1, dtype=INT_DTYPE)

    for face_idx in range(face_node_connectivity.shape[0]):
        n_edges = n_nodes_per_face[face_idx]
        for local_idx in range(n_edges):
            node_a, _ = _canonical_half_edge(
                face_node_connectivity, face_idx, local_idx, n_edges
            )
            bucket_offset[node_a + 1] += 1

    for n in range(n_node):
        bucket_offset[n + 1] += bucket_offset[n]

    return bucket_offset


@njit(cache=True)
def _scatter_half_edges(
    face_node_connectivity, n_nodes_per_face, bucket_offset, n_half_edge
):
    """Fills every bucket with its half edges, leaving ``bucket_offset`` as it found it.

    Each half edge is identified by ``half_edge_slot``, its flattened position
    ``face_idx * n_max_face_nodes + local_idx`` in the face node connectivity, and keyed on
    ``end_node``, the higher of its two nodes."""
    n_max_face_nodes = face_node_connectivity.shape[1]

    half_edge_slot = np.empty(n_half_edge, dtype=INT_DTYPE)
    end_node = np.empty(n_half_edge, dtype=INT_DTYPE)

    for face_idx in range(face_node_connectivity.shape[0]):
        n_edges = n_nodes_per_face[face_idx]
        for local_idx in range(n_edges):
            node_a, node_b = _canonical_half_edge(
                face_node_connectivity, face_idx, local_idx, n_edges
            )

            slot = bucket_offset[node_a]
            half_edge_slot[slot] = face_idx * n_max_face_nodes + local_idx
            end_node[slot] = node_b
            bucket_offset[node_a] = slot + 1

    # The scatter left each entry at its bucket's end, i.e. one slot right of where the
    # convention above wants it. One backward pass puts it back.
    for n in range(bucket_offset.shape[0] - 1, 0, -1):
        bucket_offset[n] = bucket_offset[n - 1]
    bucket_offset[0] = 0

    return half_edge_slot, end_node


@njit(cache=True)
def _emit_bucket_edges(
    end_node,
    half_edge_slot,
    bucket_start,
    bucket_end,
    node_a,
    first_edge_idx,
    edge_node_connectivity,
    face_edge_flat,
):
    """Numbers a sorted bucket's unique edges from ``first_edge_idx`` and points each of its
    half edges at the edge it belongs to."""
    edge_idx = first_edge_idx - 1
    previous_end_node = INT_FILL_VALUE

    for i in range(bucket_start, bucket_end):
        if end_node[i] != previous_end_node:
            # Duplicate half edges are adjacent, so a new key starts a new edge
            edge_idx += 1
            edge_node_connectivity[edge_idx, 0] = node_a
            edge_node_connectivity[edge_idx, 1] = end_node[i]
            previous_end_node = end_node[i]

        face_edge_flat[half_edge_slot[i]] = edge_idx


@njit(cache=True, parallel=True)
def _build_edge_node_connectivity(face_node_connectivity, n_nodes_per_face, n_node):
    """Constructs the ``edge_node_connectivity`` variable, which represents the indices of the two nodes that make up
    each edge. Additionally, the ``face_edge_connectivity`` is derived during construction,  which represents the
    indices of the edges that make up each face.

    Each edge is stored as an ascending ``(node_a, node_b)`` pair, and the edges are numbered in lexicographic
    order of that pair.

    Every half edge is bucketed on its lower node, each bucket is sorted on its higher node, and the duplicates
    that this makes adjacent are then collapsed into one edge apiece.

    Parameters
    ----------
    face_node_connectivity : np.ndarray
        Face Node Connectivity
    n_nodes_per_face : np.ndarray
        Number of nodes/edges per face
    n_node : int
        Total number of nodes, used as the number of buckets for the counting sort

    Returns
    -------
    edge_node_connectivity : np.ndarray
        Edge Node Connectivity with shape (n_edge, 2)
    face_edge_connectivity : np.ndarray
        Face Edge Connectivity with shape (n_face, n_max_face_edges)

    """
    # ``np.full`` rather than ``np.full_like``, which would inherit a Fortran-ordered
    # prototype's layout and make the flat view below unobtainable
    face_edge_connectivity = np.full(
        face_node_connectivity.shape, INT_FILL_VALUE, dtype=INT_DTYPE
    )

    n_half_edge = np.sum(n_nodes_per_face)

    if n_half_edge == 0:
        return np.empty((0, 2), dtype=INT_DTYPE), face_edge_connectivity

    bucket_offset = _count_half_edges_per_node(
        face_node_connectivity, n_nodes_per_face, n_node
    )
    half_edge_slot, end_node = _scatter_half_edges(
        face_node_connectivity, n_nodes_per_face, bucket_offset, n_half_edge
    )

    # Sort each bucket and count its unique edges while the bucket is in cache. Buckets are
    # disjoint, so this runs one bucket per thread.
    unique_per_bucket = np.empty(n_node, dtype=INT_DTYPE)
    for n in prange(n_node):
        bucket_start = bucket_offset[n]
        bucket_end = bucket_offset[n + 1]

        _sort_bucket(end_node, half_edge_slot, bucket_start, bucket_end - bucket_start)
        unique_per_bucket[n] = _count_unique_in_bucket(
            end_node, bucket_start, bucket_end
        )

    # Hand each bucket the edge index its first unique edge takes, so the emit below can run
    # one bucket per thread as well
    edge_offset = np.empty(n_node + 1, dtype=INT_DTYPE)
    n_edge = 0
    for n in range(n_node):
        edge_offset[n] = n_edge
        n_edge += unique_per_bucket[n]
    edge_offset[n_node] = n_edge

    edge_node_connectivity = np.empty((n_edge, 2), dtype=INT_DTYPE)
    face_edge_flat = face_edge_connectivity.reshape(-1)

    for n in prange(n_node):
        _emit_bucket_edges(
            end_node,
            half_edge_slot,
            bucket_offset[n],
            bucket_offset[n + 1],
            n,
            edge_offset[n],
            edge_node_connectivity,
            face_edge_flat,
        )

    return edge_node_connectivity, face_edge_connectivity


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
    """Helper for (``edge_faces``) construction."""
    edge_faces = np.full((n_edge, 2), INT_FILL_VALUE, dtype=INT_DTYPE)

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

    if "edge_node_connectivity" not in grid._ds:
        # Constructing the edges derives this variable in the same pass
        _populate_edge_node_connectivity(grid)
        return

    # In lieu of an xarray equivalent to `da.compute(a, b)`, we can batch these variables as
    # an xarray Dataset and re-extract after graph traversal.
    computed = xr.Dataset(
        {
            "face_nodes": grid.face_node_connectivity.variable,
            "n_nodes_per_face": grid.n_nodes_per_face.variable,
            "edge_nodes": grid.edge_node_connectivity.variable,
        }
    ).compute()

    face_edges = _build_face_edge_connectivity(
        computed.face_nodes.data,
        computed.n_nodes_per_face.data,
        computed.edge_nodes.data,
        grid.n_node,
    )

    grid._ds["face_edge_connectivity"] = xr.DataArray(
        face_edges,
        dims=ugrid.FACE_EDGE_CONNECTIVITY_DIMS,
        attrs=ugrid.FACE_EDGE_CONNECTIVITY_ATTRS,
    )


@njit(cache=True, parallel=True)
def _build_face_edge_connectivity(
    face_node_connectivity, n_nodes_per_face, edge_node_connectivity, n_node
):
    """Constructs the ``face_edge_connectivity`` variable, which represents the indices of the edges that make up
    each face, by looking each face's edges up in an existing ``edge_node_connectivity``. The edges keep the
    numbering they arrived with.

    Edges are bucketed on their lower node so that each of a face's edges can be found by a binary search of one
    bucket. Edges already in the canonical order that :func:`_build_edge_node_connectivity` emits are bucketed
    without being sorted again.

    Parameters
    ----------
    face_node_connectivity : np.ndarray
        Face Node Connectivity
    n_nodes_per_face : np.ndarray
        Number of nodes/edges per face
    edge_node_connectivity : np.ndarray
        Edge Node Connectivity with shape (n_edge, 2), in any order or orientation
    n_node : int
        Total number of nodes, used as the number of buckets

    Returns
    -------
    face_edge_connectivity : np.ndarray
        Face Edge Connectivity with shape (n_face, n_max_face_edges). Edges of a face that are absent from
        ``edge_node_connectivity`` are left as ``INT_FILL_VALUE``, as are the padding slots of a face with
        fewer than ``n_max_face_edges`` edges.

    """
    face_edge_connectivity = np.full(
        face_node_connectivity.shape, INT_FILL_VALUE, dtype=INT_DTYPE
    )

    bucket_offset, end_node, edge_id = _build_pair_index(edge_node_connectivity, n_node)

    for face_idx in prange(face_node_connectivity.shape[0]):
        n_edges = n_nodes_per_face[face_idx]
        for local_idx in range(n_edges):
            node_a, node_b = _canonical_half_edge(
                face_node_connectivity, face_idx, local_idx, n_edges
            )
            face_edge_connectivity[face_idx, local_idx] = _search_bucket(
                end_node,
                edge_id,
                bucket_offset[node_a],
                bucket_offset[node_a + 1],
                node_b,
            )

    return face_edge_connectivity


def _populate_node_face_connectivity(grid):
    """Constructs the UGRID connectivity variable (``node_face_connectivity``)
    and stores it within the internal (``Grid._ds``) and through the attribute
    (``Grid.node_face_connectivity``)."""

    node_faces, n_max_faces_per_node = _build_node_face_connectivity(
        grid.face_node_connectivity.values, grid.n_node
    )

    grid._ds["node_face_connectivity"] = xr.DataArray(
        node_faces,
        dims=ugrid.NODE_FACE_CONNECTIVITY_DIMS,
        attrs=ugrid.NODE_FACE_CONNECTIVITY_ATTRS,
    )


def _build_node_face_connectivity(face_nodes, n_node):
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
    face_face = _build_face_face_connectivity(
        grid.edge_face_connectivity.values, grid.n_face, grid.n_max_face_nodes
    )

    grid._ds["face_face_connectivity"] = xr.DataArray(
        data=face_face,
        dims=ugrid.FACE_FACE_CONNECTIVITY_DIMS,
        attrs=ugrid.FACE_FACE_CONNECTIVITY_ATTRS,
    )


@njit(cache=True)
def _build_face_face_connectivity(edge_face_connectivity, n_face, n_max_face_nodes):
    face_face_connectivity = np.full(
        (n_face, n_max_face_nodes), INT_FILL_VALUE, INT_DTYPE
    )
    face_index_position = np.zeros(n_face, dtype=INT_DTYPE)

    for edge_faces in edge_face_connectivity:
        face_a, face_b = edge_faces
        if face_a != INT_FILL_VALUE and face_b != INT_FILL_VALUE:
            face_face_connectivity[face_a, face_index_position[face_a]] = face_b
            face_index_position[face_a] += 1

            face_face_connectivity[face_b, face_index_position[face_b]] = face_a
            face_index_position[face_b] += 1

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
