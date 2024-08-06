import numpy as np
from numba import njit
from uxarray.constants import INT_FILL_VALUE, INT_DTYPE


def construct_dual(grid):
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
    return construct_faces(
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
    for i in range(n_node):
        # If we have less than 3 edges, we can't construct anything but a line
        if n_edges[i] < 3:
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
            )
            node_face_connectivity[i] = _face
    return node_face_connectivity


@njit(cache=True)
def _order_nodes(
    temp_face,
    node_0,
    node_central,
    n_edges,
    dual_node_x,
    dual_node_y,
    dual_node_z,
):
    node_zero = node_0 - node_central

    node_cross = np.cross(node_0, node_central)
    node_zero_mag = np.linalg.norm(node_zero)

    d_angles = np.zeros(n_edges, dtype=np.float64)
    d_angles[0] = 0.0
    final_face = np.array([INT_FILL_VALUE for _ in range(n_edges)], dtype=INT_DTYPE)
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
