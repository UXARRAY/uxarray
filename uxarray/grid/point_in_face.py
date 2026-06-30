from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

from uxarray.constants import INT_DTYPE, INT_FILL_VALUE
from uxarray.grid.arcs import on_minor_arc, orient3d_on_sphere
from uxarray.grid.utils import _get_cartesian_face_edge_nodes

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from uxarray.grid.grid import Grid

# Return codes for _point_in_polygon_sphere.
_LOC_OUTSIDE = 0
_LOC_INSIDE = 1
_LOC_ON_VERTEX = 2
_LOC_ON_EDGE = 3

# Sign codes for orient3d_on_sphere results.
_SIGN_NEG = -1
_SIGN_ZERO = 0
_SIGN_POS = 1

_VERTEX_TOL = 1e-12
_EDGE_TOL = 1e-10
_RAY_EPS = 1e-8


@njit(cache=True)
def _ray_endpoint(q):
    """Return a unit vector R perpendicular to q for use as the SPIP ray target.

    Constructs R by projecting the coordinate axis least parallel to q onto
    the plane perpendicular to q and normalizing.  This gives q·R = 0 exactly
    (a 90° arc), so q×R has magnitude ≈ 1 — making orient3d_on_sphere calls
    numerically robust regardless of q's position.

    A small perturbation is added to reduce the chance that R falls exactly on
    a polygon edge's great circle, which would trigger the -1 degenerate path.
    """
    ax, ay, az = abs(q[0]), abs(q[1]), abs(q[2])
    if ax <= ay and ax <= az:
        # Project the x-axis: (1,0,0) - q[0]*q
        r0 = 1.0 - q[0] * q[0]
        r1 = -q[1] * q[0]
        r2 = -q[2] * q[0]
    elif ay <= ax and ay <= az:
        r0 = -q[0] * q[1]
        r1 = 1.0 - q[1] * q[1]
        r2 = -q[2] * q[1]
    else:
        r0 = -q[0] * q[2]
        r1 = -q[1] * q[2]
        r2 = 1.0 - q[2] * q[2]
    r0 += _RAY_EPS
    r1 -= _RAY_EPS * 0.7
    r2 += _RAY_EPS * 0.3
    n = math.sqrt(r0 * r0 + r1 * r1 + r2 * r2)
    r = np.empty(3)
    inv = 1.0 / n
    r[0] = r0 * inv
    r[1] = r1 * inv
    r[2] = r2 * inv
    return r


@njit(cache=True)
def _counts_as_crossing(A, B, q, R):
    """Return 1 if edge AB crosses the minor arc q->R, 0 if not, -1 if degenerate.

    An edge AB crosses ray q->R iff q and R lie on opposite sides of the great
    circle plane through AB AND A and B lie on opposite sides of the great
    circle plane through q->R. Uses orient3d_on_sphere (compensated) for all
    side-of-plane tests. Returns -1 when R lies exactly on plane(AB), which
    signals the caller to perturb R and retry.
    """
    s_AB_q = orient3d_on_sphere(A, B, q)
    s_AB_R = orient3d_on_sphere(A, B, R)

    # q on great circle AB: already caught by edge-membership check; not a crossing.
    if s_AB_q == _SIGN_ZERO:
        return 0
    # R on great circle AB: degenerate ray, caller must perturb R.
    if s_AB_R == _SIGN_ZERO:
        return -1
    # q and R on the same side of plane(AB): no crossing possible.
    if s_AB_q == s_AB_R:
        return 0

    # q and R are strictly on opposite sides of plane(AB).
    # Now check whether the intersection of the two great circles falls
    # inside the minor arc A->B, i.e. A and B are on opposite sides of plane(qR).
    s_qR_A = orient3d_on_sphere(q, R, A)
    s_qR_B = orient3d_on_sphere(q, R, B)

    # A or B on great circle qR: vertex lies exactly on the ray plane.
    # Apply the half-edge rule: count the edge only if the other endpoint is
    # strictly on the negative side, so adjacent edges sharing this vertex
    # are not double-counted.
    if s_qR_A == _SIGN_ZERO or s_qR_B == _SIGN_ZERO:
        if s_qR_A == _SIGN_ZERO and s_qR_B == _SIGN_ZERO:
            return 0  # entire edge coplanar with ray: degenerate
        s_other = s_qR_B if s_qR_A == _SIGN_ZERO else s_qR_A
        return 1 if s_other == _SIGN_NEG else 0

    return 1 if s_qR_A != s_qR_B else 0


@njit(cache=True)
def _point_in_polygon_sphere(q, polygon):
    """Spherical point-in-polygon test using the perturbed-antipode ray-casting method.

    Casts a great-circle ray from q toward its perturbed antipode R and counts
    how many polygon edges the ray crosses. Uses ``orient3d_on_sphere``
    (compensated) for the crossing test, avoiding the ``arctan2`` calls in the
    winding-number approach and the large number of ``np.cross`` allocations.

    Returns one of _LOC_INSIDE, _LOC_OUTSIDE, _LOC_ON_VERTEX, _LOC_ON_EDGE.

    Degenerate-ray handling: when R falls on a polygon edge's great circle, R
    is nudged by a fixed perturbation and the loop restarts (up to 4 retries).
    AccuSphGeom's Tier-3 approach instead uses Simulation of Simplicity (SoS)
    with global vertex IDs to resolve degeneracies without any branching or
    retries. SoS requires per-vertex IDs that are not available in the current
    UXarray polygon representation, so it is left as future work.

    Parameters
    ----------
    q : np.ndarray, shape (3,)
        Query point (unit vector).
    polygon : np.ndarray, shape (n, 3)
        Polygon vertices on the unit sphere, ordered.

    Returns
    -------
    int
        Location code: _LOC_OUTSIDE (0), _LOC_INSIDE (1),
        _LOC_ON_VERTEX (2), _LOC_ON_EDGE (3).
    """
    n = polygon.shape[0]

    # 1. Vertex coincidence check.
    for i in range(n):
        dx = polygon[i, 0] - q[0]
        dy = polygon[i, 1] - q[1]
        dz = polygon[i, 2] - q[2]
        if dx * dx + dy * dy + dz * dz < _VERTEX_TOL * _VERTEX_TOL:
            return _LOC_ON_VERTEX

    # 2. Edge membership check.
    for i in range(n):
        A = polygon[i]
        B = polygon[(i + 1) % n]
        if on_minor_arc(q, A, B, _EDGE_TOL):
            return _LOC_ON_EDGE

    # 3. Ray-casting crossing count.
    # When R hits a degenerate edge, nudge and restart from i=0 so that all
    # edges are counted with the same ray — a mid-loop nudge corrupts parity.
    R = _ray_endpoint(q)
    for _retry in range(4):
        inside = False
        need_retry = False
        for i in range(n):
            A = polygon[i]
            B = polygon[(i + 1) % n]
            c = _counts_as_crossing(A, B, q, R)
            if c < 0:
                R[0] += 1e-7
                R[1] -= 1e-7
                R[2] += 5e-8
                n2 = R[0] * R[0] + R[1] * R[1] + R[2] * R[2]
                inv = 1.0 / math.sqrt(n2)
                R[0] *= inv
                R[1] *= inv
                R[2] *= inv
                need_retry = True
                break
            if c == 1:
                inside = not inside
        if not need_retry:
            return _LOC_INSIDE if inside else _LOC_OUTSIDE

    return _LOC_OUTSIDE


@njit(cache=True)
def _face_contains_point(face_edges: np.ndarray, point: np.ndarray) -> bool:
    """Determine whether a point lies within a face using spherical ray casting.

    Delegates to ``_point_in_polygon_sphere`` after extracting the vertex
    array from the edge array. Returns True for points strictly inside the
    face and for points exactly on an edge or vertex.

    Parameters
    ----------
    face_edges : np.ndarray, shape (n_edges, 2, 3)
        Cartesian unit-vector coordinates of each great-circle edge.
        Each row is [start_xyz, end_xyz].
    point : np.ndarray, shape (3,)
        3D unit-vector of the query point on the unit sphere.

    Returns
    -------
    bool
        True if the point is inside the face or on its boundary.
    """
    n = face_edges.shape[0]
    # Build the (n, 3) vertex array from the edge start points.
    polygon = np.empty((n, 3))
    for i in range(n):
        polygon[i, 0] = face_edges[i, 0, 0]
        polygon[i, 1] = face_edges[i, 0, 1]
        polygon[i, 2] = face_edges[i, 0, 2]
    loc = _point_in_polygon_sphere(point, polygon)
    return loc != _LOC_OUTSIDE


@njit(cache=True)
def _get_faces_containing_point(
    point: np.ndarray,
    candidate_indices: np.ndarray,
    face_node_connectivity: np.ndarray,
    n_nodes_per_face: np.ndarray,
    node_x: np.ndarray,
    node_y: np.ndarray,
    node_z: np.ndarray,
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
    hit_buf = np.empty(candidate_indices.shape[0], dtype=INT_DTYPE)
    count = 0
    for k in range(candidate_indices.shape[0]):
        fidx = candidate_indices[k]
        face_edges = _get_cartesian_face_edge_nodes(
            fidx, face_node_connectivity, n_nodes_per_face, node_x, node_y, node_z
        )
        if _face_contains_point(face_edges, point):
            hit_buf[count] = fidx
            count += 1
    return hit_buf[:count]


@njit(cache=True, parallel=True)
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
        p = points[i]
        cands = flat_candidate_indices[start:end]

        hits = _get_faces_containing_point(
            p, cands, face_node_connectivity, n_nodes_per_face, node_x, node_y, node_z
        )
        for j, fi in enumerate(hits):
            results[i, j] = fi
        counts[i] = hits.shape[0]

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
