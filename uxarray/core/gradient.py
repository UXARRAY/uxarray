import numpy as np

from numba import njit

from typing import Optional
from uxarray.constants import INT_FILL_VALUE

from sklearn import preprocessing


@njit
def _calculate_edge_face_difference(d_var, edge_faces, n_edge):
    """Helper function for computing the aboslute difference between the data
    values on each face that saddle a each edge.

    Edges with only a single neighbor will default to a value of zero.
    """
    edge_face_diff = np.zeros(n_edge)

    saddle_mask = edge_faces[:, 1] != INT_FILL_VALUE

    edge_face_diff[saddle_mask] = (
        d_var[..., edge_faces[saddle_mask, 0]] - d_var[..., edge_faces[saddle_mask, 1]]
    )

    return np.abs(edge_face_diff)


def _calculate_edge_node_difference():
    pass


def _calculate_grad_on_edge_from_faces(
    d_var, edge_faces, n_edge, edge_face_distances, norm: Optional[bool] = None
):
    """Helper function for computing the horizontal gradient of a field on each
    cell using values at adjacent cells.

    The expression for calculating the gradient on each edge comes from Eq. 22 in Ringler et al. (2010), J. Comput. Phys.

    Code is adapted from https://github.com/theweathermanda/MPAS_utilities/blob/main/mpas_calc_operators.py
    """

    # obtain all edges that saddle two faces
    saddle_mask = edge_faces[:, 1] != INT_FILL_VALUE

    grad = _calculate_edge_face_difference(d_var, edge_faces, n_edge)

    grad[saddle_mask] = grad[saddle_mask] / edge_face_distances[saddle_mask]

    if norm is not None:
        # normalize
        grad = preprocessing.normalize(grad.reshape(1, -1), norm=norm)[0]

    return grad
