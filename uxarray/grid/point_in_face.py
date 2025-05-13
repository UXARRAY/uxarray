from __future__ import annotations

from typing import TYPE_CHECKING, Tuple
import numpy as np

from numba import njit, prange

from uxarray.grid.utils import _get_cartesian_face_edge_nodes
from uxarray.constants import ERROR_TOLERANCE, INT_DTYPE, INT_FILL_VALUE
from uxarray.grid.arcs import point_within_gca

if TYPE_CHECKING:
    from uxarray.grid.grid import Grid
    from numpy.typing import ArrayLike


@njit(cache=True)
def _face_contains_point(face_edges: np.ndarray, point: np.ndarray) -> bool:
    """
    Determine whether a point lies within a face using the spherical winding-number method.

    This function sums the signed central angles between successive vertices of the face
    as seen from `point`.  If the total absolute winding exceeds π, the point is inside.
    Points exactly on a node or edge also count as inside.

    Parameters
    ----------
    face_edges : np.ndarray, shape (n_edges, 2, 3)
        Cartesian coordinates (unit-vectors) of each great-circle edge of the face.
        Each row is [start_xyz, end_xyz].
    point : np.ndarray, shape (3,)
        3D unit-vector of the query point on the unit sphere.

    Returns
    -------
    inside : bool
        True if the point is inside the face or lies exactly on a node/edge; False otherwise.
    """
    # 1a) quick tests for exact node‐ or edge‐hits
    for e in range(face_edges.shape[0]):
        if np.allclose(face_edges[e, 0], point,
                       rtol=ERROR_TOLERANCE, atol=ERROR_TOLERANCE):
            return True
        if np.allclose(face_edges[e, 1], point,
                       rtol=ERROR_TOLERANCE, atol=ERROR_TOLERANCE):
            return True
        if point_within_gca(point,
                            face_edges[e, 0],
                            face_edges[e, 1]):
            return True

    # 1b) build closed loop of vertices
    n = face_edges.shape[0]
    verts = np.empty((n + 1, 3), dtype=np.float64)
    for i in range(n):
        verts[i] = face_edges[i, 0]
    verts[n] = face_edges[0, 0]

    # 2) accumulate signed angle
    total = 0.0
    p = point
    for i in range(n):
        vi = verts[i]   - p
        vj = verts[i+1] - p

        ni = np.linalg.norm(vi)
        nj = np.linalg.norm(vj)
        # degenerate on-vertex case
        if ni < ERROR_TOLERANCE or nj < ERROR_TOLERANCE:
            return True

        ui = vi / ni
        uj = vj / nj

        cosang = ui.dot(uj)
        # clamp for numerical safety
        if cosang >  1.0: cosang =  1.0
        if cosang < -1.0: cosang = -1.0
        ang = np.arccos(cosang)

        # sign = sign of (ui × uj)·p
        cx = ui[1]*uj[2] - ui[2]*uj[1]
        cy = ui[2]*uj[0] - ui[0]*uj[2]
        cz = ui[0]*uj[1] - ui[1]*uj[0]
        sign = 1.0 if (cx*p[0] + cy*p[1] + cz*p[2]) >= 0.0 else -1.0

        total += sign * ang

    return np.abs(total) > np.pi



@njit(cache=True)
def _get_faces_containing_point(
        point: np.ndarray,
        candidate_indices: np.ndarray,
        face_node_connectivity: np.ndarray,
        n_nodes_per_face: np.ndarray,
        node_x: np.ndarray,
        node_y: np.ndarray,
        node_z: np.ndarray
    ) -> np.ndarray:
    """
    Test each candidate face to see if it contains the query point.

    Parameters
    ----------
    point : np.ndarray, shape (3,)
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
    hits : np.ndarray, shape (h,)
        Subset of `candidate_indices` for which the point is inside the face.
    """
    hits = []
    for idx in candidate_indices:
        face_edges = _get_cartesian_face_edge_nodes(
            idx,
            face_node_connectivity,
            n_nodes_per_face,
            node_x, node_y, node_z
        )
        if _face_contains_point(face_edges, point):
            hits.append(idx)
    return np.asarray(hits, dtype=INT_DTYPE)


@njit(cache=True, parallel=True)
def _batch_point_in_face(
        points: np.ndarray,
        flat_candidate_indices: np.ndarray,
        offsets: np.ndarray,
        face_node_connectivity: np.ndarray,
        n_nodes_per_face: np.ndarray,
        node_x: np.ndarray,
        node_y: np.ndarray,
        node_z: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        (n_points, face_node_connectivity.shape[1]),
        INT_FILL_VALUE,
        dtype=INT_DTYPE
    )
    counts = np.zeros(n_points, dtype=INT_DTYPE)

    for i in prange(n_points):
        start = offsets[i]
        end = offsets[i + 1]
        p = points[i]
        cands = flat_candidate_indices[start:end]

        hits = _get_faces_containing_point(
            p,
            cands,
            face_node_connectivity,
            n_nodes_per_face,
            node_x, node_y, node_z
        )
        for j, fi in enumerate(hits):
            results[i, j] = fi
        counts[i] = hits.shape[0]

    return results, counts



def _point_in_face_query(
        source_grid: Grid,
        points: ArrayLike
    ) -> Tuple[np.ndarray, np.ndarray]:
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

    # 1) cull with k-d tree
    kdt = source_grid._get_scipy_kd_tree()
    radius = source_grid.max_face_radius
    cand_lists = kdt.query_ball_point(x=pts, r=radius)

    # 2) prepare flattened candidates + offsets
    flat_cands = np.concatenate([np.array(lst, dtype=np.int64) for lst in cand_lists])
    lens = np.array([len(lst) for lst in cand_lists], dtype=np.int64)
    offs = np.empty(len(lens) + 1, dtype=np.int64)
    offs[0] = 0
    np.cumsum(lens, out=offs[1:])

    # 3) perform the batch winding‐number test
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
