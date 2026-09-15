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
    """Returns angles [in radians] at each node for each face, assuming convex faces
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


def _compute_equiangle_skewness(face_node_angles, n_nodes_per_face):
    """Returns the equiangle skewness at each face:
        max((Amax - Areg) / (pi - Areg), (Areg - Amin) / Areg)
    where
        Amin, Amax = min, max of the angles at the nodes of the face
        Areg = internal angle at all nodes for a regular polygon with
            the same number of sides and covering the same area as this face.

    In a flat geometry, the sum of angles in a polygon with n sides is (n-2)*pi.
    Splitting the angles equally to form a regular polygon yields Areg_flat = (n-2)*pi/n.
    However, for a spherical geometry, the sum of angles depends on face area:
        sum(angles) = (n-2)*pi + face_area / sphere_radius^2
    Areg should be based on a regular polygon with same area as the corresponding face,
    so, splitting the angles equally to form a regular polygon yields simply:
        Areg = sum(angles)/n.

    Parameters
    ----------
    face_node_angles : xr.DataArray or UxDataArray with dims 'n_face', 'n_max_face_nodes'
        Angles [in radians] at each node of each face.
    n_nodes_per_face : xr.DataArray or UxDataArray with dims 'n_face'
        Number of nodes for each face.

    Returns
    -------
    xr.DataArray or UxDataArray with dims 'n_face'
        Equiangle skewness for each face.
        Type matches the input type (xr.DataArray or UxDataArray).
    """
    Amin = face_node_angles.min("n_max_face_nodes", skipna=True)
    Amax = face_node_angles.max("n_max_face_nodes", skipna=True)
    Areg = face_node_angles.sum("n_max_face_nodes", skipna=True) / n_nodes_per_face
    term0 = (Amax - Areg) / (np.pi - Areg)
    term1 = (Areg - Amin) / Areg
    # Should just use np.maximum(term0, term1), but that drops UxDataArray type currently,
    # so use where as a workaround for now. TODO: swap to np.maximum after fixing issue #1685.
    return term0.where(term0 > term1, term1)
