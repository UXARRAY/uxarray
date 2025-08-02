import numpy as np
from numba import njit, prange

from uxarray.constants import INT_DTYPE, INT_FILL_VALUE


def construct_dual(grid):
    """Constructs a dual mesh from a given grid, by connecting the face centers
    to make a grid centered over the primal mesh face centers, with the nodes
    of the primal mesh being the face centers of the dual mesh.

    Parameters
    ----------
    grid: Grid
        Primal mesh to construct dual from

    Returns
    --------
    new_node_face_connectivity : ndarray
        Constructed node_face_connectivity for the dual mesh
    """

    # Get the dual node xyz, which is the face centers
    dual_node_x = grid.face_x.values
    dual_node_y = grid.face_y.values
    dual_node_z = grid.face_z.values

    # Get other information from the grid needed
    node_x = grid.node_x.values
    node_y = grid.node_y.values
    node_z = grid.node_z.values
    node_face_connectivity = grid.node_face_connectivity.values

    # Get an array with the number of edges for each face
    n_edges_mask = node_face_connectivity != INT_FILL_VALUE
    n_edges = np.sum(n_edges_mask, axis=1)
    max_edges = node_face_connectivity.shape[1]

    # Only nodes with 3+ edges can form valid dual faces
    valid_node_indices = np.where(n_edges >= 3)[0]

    construct_node_face_connectivity = np.full(
        (len(valid_node_indices), max_edges), INT_FILL_VALUE, dtype=INT_DTYPE
    )

    # Construct and return the faces
    new_node_face_connectivity = construct_faces(
        valid_node_indices,
        n_edges,
        dual_node_x,
        dual_node_y,
        dual_node_z,
        node_face_connectivity,
        node_x,
        node_y,
        node_z,
        construct_node_face_connectivity,
        max_edges,
    )

    return new_node_face_connectivity


@njit(cache=True, parallel=True)
def construct_faces(
    valid_node_indices,
    n_edges,
    dual_node_x,
    dual_node_y,
    dual_node_z,
    node_face_connectivity,
    node_x,
    node_y,
    node_z,
    construct_node_face_connectivity,
    max_edges,
):
    """Construct the faces of the dual mesh based on a given
    node_face_connectivity.

    Parameters
    ----------
    valid_node_indices: np.ndarray
        Array of node indices with at least 3 connections in the primal mesh
    n_edges: np.ndarray
        Array of the number of edges for each node in the primal mesh
    dual_node_x: np.ndarray
        x coordinates for the dual mesh nodes (face centers of primal mesh)
    dual_node_y: np.ndarray
        y coordinates for the dual mesh nodes (face centers of primal mesh)
    dual_node_z: np.ndarray
        z coordinates for the dual mesh nodes (face centers of primal mesh)
    node_face_connectivity: np.ndarray
        Node-to-face connectivity of the primal mesh
    node_x: np.ndarray
        x coordinates of nodes from the primal mesh
    node_y: np.ndarray
        y coordinates of nodes from the primal mesh
    node_z: np.ndarray
        z coordinates of nodes from the primal mesh
    construct_node_face_connectivity: np.ndarray
        Pre-allocated array to store the dual mesh connectivity
    max_edges: int
        The max number of edges in a face


    Returns
    --------
    construct_node_face_connectivity : ndarray
        Constructed node_face_connectivity for the dual mesh

    Notes
    -----
    In dual mesh construction, the "valid node indices" are face indices from
    the primal mesh's node_face_connectivity that are not fill values. These
    represent the actual faces that each primal node connects to, which become
    the nodes of the dual mesh faces.
    """
    n_valid = valid_node_indices.shape[0]

    for out_idx in prange(n_valid):
        i = valid_node_indices[out_idx]

        # Construct temporary face to hold unordered face nodes
        temp_face = np.array(
            [INT_FILL_VALUE for _ in range(n_edges[i])], dtype=INT_DTYPE
        )

        # Get the face indices this node connects to (these become dual face nodes)
        connected_faces = node_face_connectivity[i][0 : n_edges[i]]

        # Connect the face centers around the node to make dual face
        for index, node_idx in enumerate(connected_faces):
            if node_idx != INT_FILL_VALUE:
                temp_face[index] = node_idx

        # Order the nodes using the angles so the faces have nodes in counter-clockwise sequence
        node_central = np.array([node_x[i], node_y[i], node_z[i]])
        node_0 = np.array(
            [
                dual_node_x[temp_face[0]],
                dual_node_y[temp_face[0]],
                dual_node_z[temp_face[0]],
            ]
        )

        # Order the face nodes properly in a counter-clockwise fashion
        if temp_face[0] != INT_FILL_VALUE:
            _face = _order_nodes(
                temp_face,
                node_0,
                node_central,
                n_edges[i],
                dual_node_x,
                dual_node_y,
                dual_node_z,
                max_edges,
            )
            construct_node_face_connectivity[out_idx] = _face

    return construct_node_face_connectivity


@njit(cache=True)
def _order_nodes(
    temp_face,
    node_0,
    node_central,
    n_edges,
    dual_node_x,
    dual_node_y,
    dual_node_z,
    max_edges,
):
    """Correctly order the nodes of each face in a counter-clockwise fashion.

    Parameters
    ----------
    temp_face: np.ndarray
        Face holding unordered nodes
    node_0: np.ndarray
        Starting node
    node_central: np.ndarray
        Face center
    n_edges: int
        Number of edges in the face
    dual_node_x: np.ndarray
        x node coordinates for the dual mesh
    dual_node_y: np.ndarray
        y node coordinates for the dual mesh
    dual_node_z: np.ndarray
        z node coordinates for the dual mesh
    max_edges: int
        Max number of edges a face could have


    Returns
    --------
    final_face : np.ndarray
        The face in proper counter-clockwise order
    """
    # Add numerical stability check for degenerate cases
    if n_edges < 3:
        return np.full(max_edges, INT_FILL_VALUE, dtype=INT_DTYPE)

    node_zero = node_0 - node_central
    node_zero_mag = np.linalg.norm(node_zero)

    # Check for numerical stability
    if node_zero_mag < 1e-15:
        return np.full(max_edges, INT_FILL_VALUE, dtype=INT_DTYPE)

    node_cross = np.cross(node_0, node_central)

    d_angles = np.zeros(n_edges, dtype=np.float64)
    d_angles[0] = 0.0
    final_face = np.array([INT_FILL_VALUE for _ in range(max_edges)], dtype=INT_DTYPE)
    for j in range(1, n_edges):
        _cur_face_temp_idx = temp_face[j]

        if _cur_face_temp_idx is not INT_FILL_VALUE:
            sub = np.array(
                [
                    dual_node_x[_cur_face_temp_idx],
                    dual_node_y[_cur_face_temp_idx],
                    dual_node_z[_cur_face_temp_idx],
                ]
            )
            node_diff = sub - node_central
            node_diff_mag = np.linalg.norm(node_diff)

            # Skip if node difference is too small (numerical stability)
            if node_diff_mag < 1e-15:
                d_angles[j] = 0.0
                continue

            d_side = np.dot(node_cross, node_diff)
            d_dot_norm = np.dot(node_zero, node_diff) / (node_zero_mag * node_diff_mag)

            # Clamp to valid range for arccos to avoid numerical errors
            d_dot_norm = max(-1.0, min(1.0, d_dot_norm))

            d_angles[j] = np.arccos(d_dot_norm)

            if d_side > 0.0:
                d_angles[j] = -d_angles[j] + 2.0 * np.pi

    d_current_angle = 0.0

    final_face[0] = temp_face[0]
    for j in range(1, n_edges):
        ix_next_node = -1
        d_next_angle = 2.0 * np.pi

        for k in range(1, n_edges):
            if d_current_angle < d_angles[k] < d_next_angle:
                ix_next_node = k
                d_next_angle = d_angles[k]

        if ix_next_node == -1:
            continue

        final_face[j] = temp_face[ix_next_node]

        d_current_angle = d_next_angle

    return final_face
