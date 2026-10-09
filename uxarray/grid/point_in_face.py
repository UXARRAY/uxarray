from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

from uxarray.constants import ERROR_TOLERANCE, INT_DTYPE, INT_FILL_VALUE
from uxarray.grid.arcs import point_within_gca
from uxarray.grid.utils import _small_angle_of_2_vectors
from uxarray.utils.numba_math import (
    _numba_allclose3,
    _numba_cross3,
    _numba_dot3,
    _numba_norm3,
    _numba_sub3,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from uxarray.grid.grid import Grid


def _point_in_face_from_grid(point: np.ndarray, grid: Grid, fidx: int):
    """Returns whether this point lies within the indicated face of this grid.
    Helper function providing convenient entry point into `_point_in_face`;
    see `_point_in_face` for full docstring.
    """
    n_nodes_in_face = grid.n_nodes_per_face[fidx].item()
    nodes_idx = grid.face_node_connectivity[fidx][:n_nodes_in_face].values
    nodes_x = grid.node_x.values
    nodes_y = grid.node_y.values
    nodes_z = grid.node_z.values
    return _point_in_face(point, nodes_idx, nodes_x, nodes_y, nodes_z)


@njit(cache=True)
def _point_in_face(
    point: np.ndarray | tuple[float, float, float],
    nodes_idx: np.ndarray,
    node_x: np.ndarray,
    node_y: np.ndarray,
    node_z: np.ndarray,
) -> bool:
    """Returns whether this point lies within the face formed by these nodes.

    Uses the spherical winding-number method, which
    sums the signed central angles between successive vertices of the face
    as seen from `point`.  If the total absolute winding exceeds π, the point is inside.
    Points exactly on a node or edge also count as inside.

    Parameters
    ----------
    point : iterable of length 3
        3D unit-vector of the query point on the unit sphere.
    nodes_idx : np.ndarray, shape (n_nodes,)
        Node indices (within node_x, node_y, node_z) for precisely all nodes in this face.
        Likely from `face_node_connectivity[fidx][:n_nodes_per_face[fidx]]`.
    node_x, node_y, node_z : np.ndarray, shape (n_nodes,)
        Cartesian coordinates of all nodes.
        (This method uses the values at indices indicated by nodes_idx.)

    Returns
    -------
    inside : bool
        True if the point is inside the face or lies exactly on a node/edge; False otherwise.
    """
    # Rewritten from _face_contains_point_from_edges to avoids creating tiny numpy arrays.
    # Creating tiny numpy arrays from scratch inside numba is very inefficient.
    # (Creating tiny numpy arrays from indexing larger arrays is fine, though.)
    # This is the main reason to provide inputs as nodes_idx and node_x, ..., instead of
    # simply asking to provide a single array of x, y, z coordinates for all nodes;
    # the former avoids any need to create a tiny numpy array to store the coordinates.

    n_nodes = len(nodes_idx)
    max_i_node = n_nodes - 1

    # Check for an exact hit with any of the corner nodes
    for i in range(n_nodes):
        node_idx = nodes_idx[i]
        node_xyz = (node_x[node_idx], node_y[node_idx], node_z[node_idx])
        if _numba_allclose3(
            node_xyz, point, rtol=ERROR_TOLERANCE, atol=ERROR_TOLERANCE
        ):
            return True

    # Check whether point lies on any edge of the face
    # (edges are great-circle arcs between successive nodes)
    for i in range(n_nodes):
        # edge is formed by nodes (a, b)
        ai = nodes_idx[i]
        bi = nodes_idx[i + 1] if i < max_i_node else nodes_idx[0]

        a = (node_x[ai], node_y[ai], node_z[ai])
        b = (node_x[bi], node_y[bi], node_z[bi])
        if point_within_gca(point, a, b):
            return True

    # Apply spherical winding-number method:
    total = 0.0
    for i in range(n_nodes):
        # edge is formed by nodes (a, b).
        ai = nodes_idx[i]
        bi = nodes_idx[i + 1] if i < max_i_node else nodes_idx[0]

        a = (node_x[ai], node_y[ai], node_z[ai])
        b = (node_x[bi], node_y[bi], node_z[bi])

        vi = _numba_sub3(a, point)
        vj = _numba_sub3(b, point)

        # check if you’re right on a vertex
        if _numba_norm3(vi) < ERROR_TOLERANCE or _numba_norm3(vj) < ERROR_TOLERANCE:
            return True

        ang = _small_angle_of_2_vectors(vi, vj)

        # determine sign from cross
        c = _numba_cross3(vi, vj)
        sign = 1.0 if _numba_dot3(c, point) >= 0.0 else -1.0

        total += sign * ang

    return np.abs(total) > np.pi


@njit(cache=True)
def _set_faces_containing_point(
    result: np.ndarray,
    i: int,
    point: np.ndarray | tuple[float, float, float],
    candidate_indices: np.ndarray,
    face_node_connectivity: np.ndarray,
    n_nodes_per_face: np.ndarray,
    node_x: np.ndarray,
    node_y: np.ndarray,
    node_z: np.ndarray,
) -> int:
    """
    Test each candidate face to see if it contains the query point,
    setting result[i, j] = candidate_indices[k] for all k where the
    point is inside the face. j starts at 0 and increments by 1 for
    each hit. Returns the total number of hits.

    Parameters
    ----------
    result : np.ndarray, shape (n_points, max_possible_hits)
        Preallocated array to store face indices for each point.
        max_possible_hits can be much less than max_candidates;
        the maximum number of hits is n_max_face_nodes, because
        the "worst case" of point being a node would lead to hits
        of all faces it is a part of, but nothing else.
    point : iterable of length 3
        Cartesian unit-vector of the query point.
    candidate_indices : np.ndarray, shape (k,)
        Array of face indices to test (e.g., from a k-d tree cull).
    face_node_connectivity : np.ndarray, shape (n_faces, max_nodes)
        Node connectivity (node indices) for each face.
    n_nodes_per_face : np.ndarray, shape (n_faces,)
        Number of valid nodes per face.
    node_x, node_y, node_z : np.ndarray, shape (n_nodes,)
        Cartesian coordinates of each grid node.

    Returns
    -------
    n_hits : int
        Number of candidate faces that contain the point.
    """
    count = 0
    for k in range(candidate_indices.shape[0]):
        fidx = candidate_indices[k]
        nodes_idx = face_node_connectivity[fidx][: n_nodes_per_face[fidx]]
        if _point_in_face(point, nodes_idx, node_x, node_y, node_z):
            result[i, count] = fidx
            count += 1
    return count


@njit(cache=True, parallel=True, nogil=True)
def _batch_point_in_face(
    points: np.ndarray,
    flat_candidate_indices: np.ndarray,
    offsets: np.ndarray,
    face_node_connectivity: np.ndarray,
    n_nodes_per_face: np.ndarray,
    node_x: np.ndarray,
    node_y: np.ndarray,
    node_z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parallel entry-point: for each point, test all candidate faces in batch.

    Parameters
    ----------
    points : np.ndarray, shape (n_points, 3)
        Cartesian coordinates of query points.
    flat_candidate_indices : np.ndarray, shape (total_candidates,)
        Flattened array of all candidate face indices across points.
    offsets : np.ndarray, shape (n_points + 1,)
        Offsets into `flat_candidate_indices` marking each point’s slice.
    face_node_connectivity, n_nodes_per_face, node_x, node_y, node_z
        As in `_get_faces_containing_point`.

    Returns
    -------
    results : np.ndarray, shape (n_points, max_candidates)
        Each row lists face indices containing the corresponding point;
        unused entries are filled with `INT_FILL_VALUE`.
    counts : np.ndarray, shape (n_points,)
        Number of valid face-hits per point.
    """
    n_points = offsets.shape[0] - 1
    results = np.full(
        (n_points, face_node_connectivity.shape[1]), INT_FILL_VALUE, dtype=INT_DTYPE
    )
    counts = np.zeros(n_points, dtype=INT_DTYPE)

    for i in prange(n_points):
        start = offsets[i]
        end = offsets[i + 1]
        p = (points[i][0], points[i][1], points[i][2])
        cands = flat_candidate_indices[start:end]

        n_hits = _set_faces_containing_point(
            results,
            i,
            p,
            cands,
            face_node_connectivity,
            n_nodes_per_face,
            node_x,
            node_y,
            node_z,
        )
        counts[i] = n_hits

    return results, counts


def _point_in_face_query(
    source_grid: Grid, points: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find grid faces that contain given Cartesian point(s) on the unit sphere.

    This function first uses a SciPy k-d tree (in Cartesian space) to cull
    candidate faces within the maximum face‐radius, then calls the Numba‐
    accelerated winding‐number tester in parallel.

    Parameters
    ----------
    source_grid : Grid
        UXarray Grid object, which must provide:
        - ._get_scipy_kd_tree(): a SciPy cKDTree over node‐centroids,
        - .max_face_radius: float search radius,
        - .face_node_connectivity, .n_nodes_per_face, .node_x, .node_y, .node_z
          arrays for reconstructing face edges.
    points : array_like, shape (3,) or (n_points, 3)
        Cartesian coordinates of query point(s).

    Returns
    -------
    face_indices : np.ndarray, shape (n_points, max_candidates)
        2D array of face indices containing each point; unused entries are
        padded with `INT_FILL_VALUE`.
    counts : np.ndarray, shape (n_points,)
        Number of valid face‐hits per point.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[np.newaxis, :]
    # Cull with k-d tree
    kdt = source_grid._get_scipy_kd_tree()
    radius = source_grid.max_face_radius * 1.05
    cand_lists = kdt.query_ball_point(x=pts, r=radius, workers=-1)

    # Prepare flattened candidates and offsets
    flat_cands = np.concatenate([np.array(lst, dtype=np.int64) for lst in cand_lists])
    lens = np.array([len(lst) for lst in cand_lists], dtype=np.int64)
    offs = np.empty(len(lens) + 1, dtype=np.int64)
    offs[0] = 0
    np.cumsum(lens, out=offs[1:])

    # Perform the batch winding‐number test
    return _batch_point_in_face(
        pts,
        flat_cands,
        offs,
        source_grid.face_node_connectivity.values,
        source_grid.n_nodes_per_face.values,
        source_grid.node_x.values,
        source_grid.node_y.values,
        source_grid.node_z.values,
    )
