import numpy as np
from numba import njit
from uxarray.constants import INT_FILL_VALUE


@njit
def _face_weights_from_edge_magnitudes(
    edge_magnitudes, edge_face_connectivity, face_indices, edge_indices
):
    idx_dict = {}
    for i in range(len(face_indices)):
        idx_dict[face_indices[i]] = i

    weights = np.zeros(len(face_indices), dtype=edge_magnitudes.dtype)

    for i in range(len(edge_indices)):
        cur_edge_index = edge_indices[i]
        saddle_face_indices = edge_face_connectivity[cur_edge_index]

        face_index_a = saddle_face_indices[0]
        face_index_b = saddle_face_indices[1]

        if saddle_face_indices[1] == INT_FILL_VALUE:
            weights[idx_dict[face_index_a]] = edge_magnitudes[cur_edge_index]
        else:
            half_weight = 0.5 * edge_magnitudes[cur_edge_index]
            weights[idx_dict[face_index_a]] = half_weight
            weights[idx_dict[face_index_b]] = half_weight

    return weights
