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


def _compute_face_node_angles_convex_nonumba(
    node_x,
    node_y,
    node_z,
    face_node_connectivity,
    n_nodes_per_face,
    *,
    geometry="spherical",
):
    """
    Naive padded/masked numpy vectorization of _compute_face_node_angles_convex.

    (Provided by claude to help with "should we vectorize?" question (see #1566).
    Skimmed for correctness and compared results to the numba implementation;
    results seem to be the same for all tutorial grids, to within rounding errors.

    >> Not intended to be merged as-is; just for timing comparisons <<
    Be sure to take a closer look before considering merge,
    and only do so if it is faster than numpy version.)

    Computes angle at every (face, slot) position up to n_max_face_nodes,
    including padded slots that don't correspond to a real node, then masks
    those out. This is the "just vectorize it" approach a reviewer might
    picture -- it trades the numba loop's "skip padding entirely" behavior
    for full-array numpy ops, at the cost of doing wasted work on padding.

    Parameters mirror _compute_face_node_angles_convex exactly.

    Returns
    -------
    np.ndarray with shape (n_faces, n_max_face_nodes)
        Angles at each node of each face. Padded/invalid slots are np.nan.
    """
    n_faces, n_max_face_nodes = face_node_connectivity.shape

    # slot index per face, broadcast against each face's actual node count
    j_idx = np.arange(n_max_face_nodes)[None, :]                      # (1, n_max)
    n_nodes = n_nodes_per_face[:, None].astype(np.int64)               # (n_faces, 1)
    valid_mask = j_idx < n_nodes                                       # (n_faces, n_max)

    # avoid mod-by-zero for fully-padded rows (shouldn't happen, but be safe)
    safe_n_nodes = np.where(n_nodes == 0, 1, n_nodes)

    prev_j = (j_idx - 1) % safe_n_nodes                                 # (n_faces, n_max)
    next_j = (j_idx + 1) % safe_n_nodes

    ihere = face_node_connectivity
    iprev = np.take_along_axis(face_node_connectivity, prev_j, axis=1)
    inext = np.take_along_axis(face_node_connectivity, next_j, axis=1)

    # clip any fill-value / garbage indices so gathers don't go out of bounds
    # (values at invalid slots are discarded via valid_mask at the end anyway)
    n_pts = node_x.shape[0]
    ihere_safe = np.clip(ihere, 0, n_pts - 1)
    iprev_safe = np.clip(iprev, 0, n_pts - 1)
    inext_safe = np.clip(inext, 0, n_pts - 1)

    xhere, yhere, zhere = node_x[ihere_safe], node_y[ihere_safe], node_z[ihere_safe]
    xprev, yprev, zprev = node_x[iprev_safe], node_y[iprev_safe], node_z[iprev_safe]
    xnext, ynext, znext = node_x[inext_safe], node_y[inext_safe], node_z[inext_safe]

    here = np.stack([xhere, yhere, zhere], axis=-1)   # (n_faces, n_max, 3)
    v1 = np.stack([xprev - xhere, yprev - yhere, zprev - zhere], axis=-1)
    v2 = np.stack([xnext - xhere, ynext - yhere, znext - zhere], axis=-1)

    if geometry == "spherical":
        normal = here / np.linalg.norm(here, axis=-1, keepdims=True)
        v1 = v1 - np.sum(v1 * normal, axis=-1, keepdims=True) * normal
        v2 = v2 - np.sum(v2 * normal, axis=-1, keepdims=True) * normal

    # same formula as _small_angle_of_2_vectors, applied elementwise
    norm_v1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    norm_v2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    v_norm_times_u = norm_v2 * v1
    u_norm_times_v = norm_v1 * v2
    vec_minus = v_norm_times_u - u_norm_times_v
    vec_sum = v_norm_times_u + u_norm_times_v
    angles = 2 * np.arctan2(
        np.linalg.norm(vec_minus, axis=-1), np.linalg.norm(vec_sum, axis=-1)
    )

    result = np.where(valid_mask, angles, np.nan)
    return result
