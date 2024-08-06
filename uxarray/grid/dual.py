import numpy as np
from numba import njit
from uxarray.constants import INT_FILL_VALUE, INT_DTYPE


def construct_dual(grid):
    # Get the dual node xyz, which is the face centers
    dual_node_x = grid.face_x.values
    dual_node_y = grid.face_y.values
    dual_node_z = grid.face_z.values

    node_face_connectivity = grid.node_face_connectivity.values

    for i in range(grid.n_node):
        # Get the number of edges, which is just how many faces are in the node_face
        # TODO: outside of the loop with a vectorized implementation (1 sum call)
        n_edges = sum(1 for w in node_face_connectivity[i] if w != INT_FILL_VALUE)

        # If we have less than 3 edges, we can't construct anything but a line
        if n_edges < 3:
            continue

        _face_edges = np.array(
            [[INT_FILL_VALUE, INT_FILL_VALUE] for _ in range(n_edges)],
            dtype=INT_DTYPE,
        )
        # _temp_face_edges = np.array(
        #     [[INT_FILL_VALUE, INT_FILL_VALUE] for _ in range(n_edges)],
        #     dtype=INT_DTYPE,
        # )
        temp_face = np.array([INT_FILL_VALUE for _ in range(n_edges)], dtype=INT_DTYPE)

        valid_node_indices = node_face_connectivity[i][0:n_edges]

        index = 0

        # Connect the face centers around the node to make dual face
        for node_idx in valid_node_indices:
            # face_temp.set_node(index, node_idx)
            temp_face[index] = node_idx
            # _set_node(_temp_face_edges, index, node_idx)
            index += 1

        # Order the nodes using the angles so the faces have nodes in counter-clockwise sequence
        node_central = np.array([grid.node_x[i], grid.node_y[i], grid.node_z[i]])
        node_0 = np.array(
            [
                dual_node_x[temp_face[0]],
                dual_node_y[temp_face[0]],
                dual_node_z[temp_face[0]],
            ]
        )

        if temp_face[0] is not INT_FILL_VALUE:
            _face = _order_nodes(
                _face_edges,
                temp_face,
                node_0,
                node_central,
                n_edges,
                dual_node_x,
                dual_node_y,
                dual_node_z,
            )
            node_face_connectivity[i] = _face
            # final_faces.append(_face)
    return node_face_connectivity

    # # Empty array to hold `node_face_connectivity`
    # node_face_connectivity = np.full(
    #     (len(final_faces), grid.n_max_node_faces),
    #     INT_FILL_VALUE,
    #     dtype=ux.INT_DTYPE,
    # )
    #
    # # Populate the node_face array
    # for i, face in enumerate(final_faces):
    #     faces = [
    #         edge[0] for edge in face
    #     ]  # Extract the first element of each edge tuple
    #     node_face_connectivity[i, : len(faces)] = (
    #         faces  # Assign the faces to the node_face array
    #     )
    #
    # return node_face_connectivity


@njit(cache=True)
def _order_nodes(
    face_edges,
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
    # face.set_node(0, face_temp[0])

    # _set_node(face_edges, 0, temp_face[0])

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

        # face.set_node(j, face_temp[ix_next_node])

        final_face[j] = temp_face[ix_next_node]
        # _set_node(face_edges, j, temp_face[ix_next_node])

        d_current_angle = d_next_angle
    final_face[n_edges] = temp_face[0]

    return final_face
