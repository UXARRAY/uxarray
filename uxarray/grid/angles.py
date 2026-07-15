"""
Purpose: angle calculations on a grid
"""

from numba import njit, prange
import numpy as np
import xarray as xr

from uxarray.grid.utils import _small_angle_of_2_vectors


@njit(cache=True, parallel=True)
def _compute_face_node_angles_convex(
    node_x,
    node_y,
    node_z,
    face_node_connectivity,
    n_nodes_per_face,
    *,
    geometry="spherical",
):
    """
    Calculate the angles at each node for each face assuming the faces are convex.

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
    geometry : str, "spherical" or "flat", default "spherical"
        The geometry to use. "spherical" respects the true underlying geometry,
        by projecting edges onto the tangent plane at each node.
        "flat" finds angles between chords, which is less accurate but also less expensive.

    Returns
    -------
    np.ndarray with shape (n_faces, n_max_face_nodes)
        Angles at each node of each face.
        INT_FILL_VALUE elements from face_node_connectivity correspond with np.nan in the result.
    """
    n_faces, n_max_face_nodes = face_node_connectivity.shape
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
            v1 = np.array([node_x[iprev] - xhere, node_y[iprev] - yhere, node_z[iprev] - zhere])
            v2 = np.array([node_x[inext] - xhere, node_y[inext] - yhere, node_z[inext] - zhere])
            if geometry == "spherical":
                # Project onto tangent plane at the current node
                normal = np.array([xhere, yhere, zhere])
                normal /= np.linalg.norm(normal)
                v1 -= np.dot(v1, normal) * normal
                v2 -= np.dot(v2, normal) * normal
            result[i, j] = _small_angle_of_2_vectors(v1, v2)
    return result
