from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange

from uxarray.constants import INT_DTYPE, INT_FILL_VALUE, POINT_IN_FACE_CULL_THRESHOLDS
from uxarray.grid.arcs import _normal_dot_value, on_minor_arc
from uxarray.utils.computing import accucross
from uxarray.utils.numba_math import _numba_dot3, _numba_sub3
from uxarray.utils.parallel import parallel_njit

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from uxarray.grid.grid import Grid

# Return codes for _point_in_polygon_sphere.
_LOC_OUTSIDE = 0
_LOC_INSIDE = 1
_LOC_ON_VERTEX = 2
_LOC_ON_EDGE = 3

_VERTEX_TOL = 1e-12
_EDGE_TOL = 1e-10


@njit(cache=True)
def _point_in_polygon_sphere(q, polygon):
    """Classify a point using signed spherical winding and compensated predicates.

    The signed solid angle determines the ring orientation, preventing the
    antipodal image from being mistaken for the polygon interior. Faces are
    simple rings representing the smaller spherical region (less than 2*pi).
    Boundary checks precede winding and include both vertices and minor arcs.

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

    # Signed tangent-plane winding distinguishes the polygon from its antipodal
    # image. An absolute winding test incorrectly accepts both of them.
    winding = 0.0
    for i in range(n):
        a = polygon[i]
        b = polygon[(i + 1) % n]
        nx, ny, nz, lx, ly, lz = accucross(a[0], a[1], a[2], b[0], b[1], b[2])
        numerator = _normal_dot_value(nx, ny, nz, lx, ly, lz, q[0], q[1], q[2])
        denominator = np.dot(a, b) - np.dot(q, a) * np.dot(q, b)
        winding += math.atan2(numerator, denominator)

    # The signed solid angle fixes the ring orientation independently of q.
    # Triangulation is algebraic, so concave rings do not need a convex centroid.
    area = 0.0
    a = polygon[0]
    for i in range(1, n - 1):
        b = polygon[i]
        c = polygon[i + 1]
        nx, ny, nz, lx, ly, lz = accucross(b[0], b[1], b[2], c[0], c[1], c[2])
        numerator = _normal_dot_value(nx, ny, nz, lx, ly, lz, a[0], a[1], a[2])
        denominator = 1.0 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a)
        area += 2.0 * math.atan2(numerator, denominator)
    return _LOC_INSIDE if winding * np.sign(area) > math.pi else _LOC_OUTSIDE


def _point_in_face_from_grid(point: np.ndarray, grid: Grid, fidx: int):
    """Test a grid face through the node-index-based spherical predicate."""
    n_nodes = grid.n_nodes_per_face[fidx].item()
    nodes_idx = grid.face_node_connectivity[fidx][:n_nodes].values
    return _point_in_face(
        point, nodes_idx, grid.node_x.values, grid.node_y.values, grid.node_z.values
    )


@njit(cache=True)
def _point_in_face(
    point: np.ndarray | tuple[float, float, float],
    nodes_idx: np.ndarray,
    node_x: np.ndarray,
    node_y: np.ndarray,
    node_z: np.ndarray,
) -> bool:
    """Test a face's ordered node indices; edges and vertices count as inside."""
    polygon = np.empty((len(nodes_idx), 3))
    for i in range(len(nodes_idx)):
        idx = nodes_idx[i]
        polygon[i, 0] = node_x[idx]
        polygon[i, 1] = node_y[idx]
        polygon[i, 2] = node_z[idx]
    return _point_in_polygon_sphere(np.asarray(point), polygon) != _LOC_OUTSIDE


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


@parallel_njit
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


@njit(inline="always", cache=True)
def _point_within_face_radius(point, face_center, face_radius):
    """True if point is within face_radius of face_center.

    point, face_center: iterables of length 3
        the (x,y,z) coordinates of the point and face center
    face_radius: float
        the maximum distance from face center to any point on the face
    """
    diff = _numba_sub3(point, face_center)
    dist2 = _numba_dot3(diff, diff)
    return dist2 <= face_radius * face_radius


@parallel_njit
def _cull_candidates_by_radius(
    pts, flat_cands, offsets, face_x, face_y, face_z, face_radii
):
    """
    Keep only candidates where dist(point, face_center) <= face_radius.

    Returns new flat candidates and new offsets.

    Parameters
    ----------
    pts : np.ndarray, shape (n_pts, 3)
        Cartesian coordinates of query points.
    flat_cands : np.ndarray, shape (total_candidates,)
        Flattened array of all candidate face indices across points.
    offsets : np.ndarray, shape (n_pts + 1,)
        Offsets into `flat_cands` marking each point’s slice.
    face_x, face_y, face_z : np.ndarray, shape (n_faces,)
        Cartesian coordinates of face centers.
    face_radii : np.ndarray, shape (n_faces,)
        Maximum distance from face center to any point on the face.
    """
    n_pts = pts.shape[0]

    # figure out which faces to keep:
    keep = np.zeros(flat_cands.shape[0], dtype=np.bool_)
    for i in prange(n_pts):
        point = pts[i]
        for k in range(offsets[i], offsets[i + 1]):
            f = flat_cands[k]
            face_center = (face_x[f], face_y[f], face_z[f])
            face_radius = face_radii[f]
            keep[k] = _point_within_face_radius(point, face_center, face_radius)

    # count survivors per point
    new_lens = np.empty(n_pts, dtype=np.int64)
    for i in prange(n_pts):
        new_lens[i] = np.count_nonzero(keep[offsets[i] : offsets[i + 1]])

    # build new offsets
    new_offsets = np.empty(n_pts + 1, dtype=np.int64)
    new_offsets[0] = 0
    for i in range(n_pts):  # cheap serial cumsum
        new_offsets[i + 1] = new_offsets[i] + new_lens[i]

    # build new flat candidates
    new_flat = np.empty(new_offsets[n_pts], dtype=np.int64)
    for i in prange(n_pts):
        w = new_offsets[i]
        for k in range(offsets[i], offsets[i + 1]):
            if keep[k]:
                new_flat[w] = flat_cands[k]
                w += 1

    return new_flat, new_offsets


def _point_in_face_query(
    source_grid: Grid, points: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find grid faces that contain given Cartesian point(s) on the unit sphere.

    Follows these steps:
    (1) Cull candidate faces using a SciPy k-d tree (in Cartesian space),
        checking whether the point is within `max_face_radius` of each face's center
    (2) Maybe cull candidate faces further by checking whether the point is within
        each face's radius of the face center. This cull has a nontrivial cost,
        so it is gated to only run if there are many candidates.
    (3) call the Numba‐accelerated winding‐number tester in parallel.

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
    SAFETY = 1.05

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[np.newaxis, :]

    # Cull with k-d tree: a face can only contain a point if
    # the point is within max_face_radius of the face's center.
    kdt = source_grid._get_scipy_kd_tree()
    radius = source_grid.max_face_radius * SAFETY
    cand_lists = kdt.query_ball_point(x=pts, r=radius, workers=-1)

    # Prepare flattened candidates and offsets
    flat_cands = np.concatenate([np.array(lst, dtype=np.int64) for lst in cand_lists])
    lens = np.array([len(lst) for lst in cand_lists], dtype=np.int64)
    offs = np.empty(len(lens) + 1, dtype=np.int64)
    offs[0] = 0
    np.cumsum(lens, out=offs[1:])

    # Cull by face radius: a face can only contain a point if
    # the point is within the face's radius of the face's center.

    # The cull has a nontrivial cost, so for efficiency, gate it to only run if there
    # are "many" candidates, i.e. large (A) n_cands / n_points, and large (B) n_cands / n_face.
    # Large (A) ratio happens in variable-resolution grids (many faces' radii < max_face_radius).
    # Small (B) indicates not enough candidates to make up for compute_face_radii() cost, which
    # scales with n_face. (Could remove gate (B) if compute_face_radii() result was cached.)
    n_cands_per_point = len(flat_cands) / len(pts)
    n_cands_per_face = len(flat_cands) / source_grid.n_face
    if (n_cands_per_point > POINT_IN_FACE_CULL_THRESHOLDS["CANDS_PER_POINT"]) and (
        n_cands_per_face > POINT_IN_FACE_CULL_THRESHOLDS["CANDS_PER_FACE"]
    ):
        radii = source_grid.compute_face_radii().values * SAFETY
        flat_cands, offs = _cull_candidates_by_radius(
            pts,
            flat_cands,
            offs,
            source_grid.face_x.values,
            source_grid.face_y.values,
            source_grid.face_z.values,
            radii,
        )

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
