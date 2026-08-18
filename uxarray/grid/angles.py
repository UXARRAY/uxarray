"""
Purpose: angle calculations on a grid
"""

import numpy as np
from numba import njit, prange

from uxarray.grid.utils import _numba_norm3, _small_angle_of_2_vectors


@njit(cache=True, parallel=True)
def _compute_face_node_angles_convex(
    node_x,
    node_y,
    node_z,
    face_node_connectivity,
    n_nodes_per_face,
):
    """
    Calculate the angles at each node for each face, assuming convex faces
    and a spherical geometry (these assumptions occur throughout uxarray).

    Parameters
    ----------
    node_x : np.ndarray with shape (n_nodes,)
        X coordinates of the nodes.
    node_y : np.ndarray with shape (n_nodes,)
        Y coordinates of the nodes.
    node_z : np.ndarray with shape (n_nodes,)
        Z coordinates of the nodes.
    face_node_connectivity : np.ndarray with shape (n_faces, n_max_face_nodes)
        Connectivity array defining which nodes form each face.
    n_nodes_per_face : np.ndarray with shape (n_faces,)
        Number of nodes for each face.

    Returns
    -------
    np.ndarray with shape (n_faces, n_max_face_nodes)
        Angles at each node of each face.
        INT_FILL_VALUE elements from face_node_connectivity correspond with np.nan in the result.
    """
    n_faces, _n_max_face_nodes = face_node_connectivity.shape
    result = np.full(face_node_connectivity.shape, np.nan, dtype=np.float64)
    for i in prange(n_faces):
        n_nodes = n_nodes_per_face[i]
        for j in range(n_nodes):
            ihere = face_node_connectivity[i, j]
            iprev = face_node_connectivity[i, (j - 1) % n_nodes]
            inext = face_node_connectivity[i, (j + 1) % n_nodes]
            xhere = node_x[ihere]
            yhere = node_y[ihere]
            zhere = node_z[ihere]
            v1 = (node_x[iprev] - xhere, node_y[iprev] - yhere, node_z[iprev] - zhere)
            v2 = (node_x[inext] - xhere, node_y[inext] - yhere, node_z[inext] - zhere)
            # Spherical geometry: project onto tangent plane at the current node
            normal = (xhere, yhere, zhere)
            normal_norm = _numba_norm3(normal)  # |normal|
            normal = (
                normal[0] / normal_norm,
                normal[1] / normal_norm,
                normal[2] / normal_norm,
            )
            # v1 -= np.dot(v1, normal) * normal
            v1_dot_normal = v1[0] * normal[0] + v1[1] * normal[1] + v1[2] * normal[2]
            v2_dot_normal = v2[0] * normal[0] + v2[1] * normal[1] + v2[2] * normal[2]
            v1 = (
                v1[0] - v1_dot_normal * normal[0],
                v1[1] - v1_dot_normal * normal[1],
                v1[2] - v1_dot_normal * normal[2],
            )
            v2 = (
                v2[0] - v2_dot_normal * normal[0],
                v2[1] - v2_dot_normal * normal[1],
                v2[2] - v2_dot_normal * normal[2],
            )
            result[i, j] = _small_angle_of_2_vectors(v1, v2)
    return result
