import numpy as np
from uxarray.constants import INT_FILL_VALUE, INT_DTYPE

from numba import njit


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
    n_node = grid.n_node
    node_x = grid.node_x.values
    node_y = grid.node_y.values
    node_z = grid.node_z.values
    node_face_connectivity = grid.node_face_connectivity.values

    # Get an array with the number of edges for each face
    n_edges_mask = node_face_connectivity != INT_FILL_VALUE
    n_edges = np.sum(n_edges_mask, axis=1)

    # Construct and return the faces
    new_node_face_connectivity = construct_faces(
        n_node,
        n_edges,
        dual_node_x,
        dual_node_y,
        dual_node_z,
        node_face_connectivity,
        node_x,
        node_y,
        node_z,
    )

    return new_node_face_connectivity


@njit(cache=True)
def construct_faces(
    n_node,
    n_edges,
    dual_node_x,
    dual_node_y,
    dual_node_z,
    node_face_connectivity,
    node_x,
    node_y,
    node_z,
):
    """Construct the faces of the dual mesh based on a given
    node_face_connectivity.

    Parameters
    ----------
    n_node: np.ndarray
        number of nodes in the primal mesh
    n_edges: np.ndarray
        array of the number of edges for each dual face
    dual_node_x: np.ndarray
        x node coordinates for the dual mesh
    dual_node_y: np.ndarray
        y node coordinates for the dual mesh
    dual_node_z: np.ndarray
        z node coordinates for the dual mesh
    node_face_connectivity: np.ndarray
        `node_face_connectivity` of the primal mesh
    node_x: np.ndarray
        x node coordinates from the primal mesh
    node_y: np.ndarray
        y node coordinates from the primal mesh
    node_z: np.ndarray
        z node coordinates from the primal mesh


    Returns
    --------
    node_face_connectivity : ndarray
        Constructed node_face_connectivity for the dual mesh
    """
    correction = 0
    max_edges = len(node_face_connectivity[0])
    construct_node_face_connectivity = np.full(
        (np.sum(n_edges > 2), max_edges), INT_FILL_VALUE, dtype=INT_DTYPE
    )
    for i in range(n_node):
        # If we have less than 3 edges, we can't construct anything but a line
        if n_edges[i] < 3:
            correction += 1
            continue

        # Construct temporary face to hold unordered face nodes
        temp_face = np.array(
            [INT_FILL_VALUE for _ in range(n_edges[i])], dtype=INT_DTYPE
        )

        # Get a list of the valid non fill value nodes
        valid_node_indices = node_face_connectivity[i][0 : n_edges[i]]
        index = 0

        # Connect the face centers around the node to make dual face
        for node_idx in valid_node_indices:
            temp_face[index] = node_idx
            index += 1

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
        if temp_face[0] is not INT_FILL_VALUE:
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
            construct_node_face_connectivity[i - correction] = _face
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
    node_zero = node_0 - node_central

    node_cross = np.cross(node_0, node_central)
    node_zero_mag = np.linalg.norm(node_zero)

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

            d_side = np.dot(node_cross, node_diff)
            d_dot_norm = np.dot(node_zero, node_diff) / (node_zero_mag * node_diff_mag)

            if d_dot_norm > 1.0:
                d_dot_norm = 1.0

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
