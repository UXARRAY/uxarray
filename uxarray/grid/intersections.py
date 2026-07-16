import math

import numpy as np
from numba import njit, prange

from uxarray.constants import ERROR_TOLERANCE, INT_DTYPE
from uxarray.grid.arcs import _on_minor_arc_xyz, on_minor_arc
from uxarray.utils.computing import (
    _cdp2,
    _cdp4,
    _sum_of_squares_c,
    acc_sqrt_re,
    accucross,
    accucross_pair,
    two_prod,
    two_sum,
)

# ---------------------------------------------------------------------------
# Edge screeners (pre-existing, unrelated to the EFT intersection kernels below).
#
# These two functions are fast O(n) passes used by Grid.get_edges_at_constant_*
# to identify candidate edges before the expensive GCA intersection is computed.
# "no_extreme" means arc z-extrema along the great circle are not considered —
# only the endpoint z/lon values are checked.  They are not part of the
# AccuSphGeom-derived EFT stack.
# ---------------------------------------------------------------------------


@njit(parallel=True, nogil=True, cache=True)
def constant_lat_intersections_no_extreme(lat, edge_node_z, n_edge):
    """Determine which edges intersect a constant line of latitude on a
    sphere, without wrapping to the opposite longitude, with extremes
    along each great circle arc not considered.

    Parameters
    ----------
    lat:
        Constant latitude value in degrees.
    edge_node_x:
        Array of shape (n_edge, 2) containing x-coordinates of the edge nodes.
    edge_node_y:
        Array of shape (n_edge, 2) containing y-coordinates of the edge nodes.
    n_edge:
        Total number of edges to check.

    Returns
    -------
    intersecting_edges:
        array of indices of edges that intersect the constant latitude.
    """
    lat = np.deg2rad(lat)

    intersecting_edges_mask = np.zeros(n_edge, dtype=np.int32)

    # Calculate the constant z-value for the given latitude
    z_constant = np.sin(lat)

    # Iterate through each edge and check for intersections
    for i in prange(n_edge):
        # Check if the edge crosses the constant latitude or lies exactly on it
        if edge_intersects_constant_lat_no_extreme(edge_node_z[i], z_constant):
            intersecting_edges_mask[i] = 1

    intersecting_edges = np.argwhere(intersecting_edges_mask)

    return np.unique(intersecting_edges)


@njit(cache=True, nogil=True)
def edge_intersects_constant_lat_no_extreme(edge_node_z, z_constant):
    """Helper to compute whether an edge intersects a line of constant latitude."""

    # z coordinate of edge nodes
    z0 = edge_node_z[0]
    z1 = edge_node_z[1]

    if (z0 - z_constant) * (z1 - z_constant) < 0.0 or (
        abs(z0 - z_constant) < ERROR_TOLERANCE
        and abs(z1 - z_constant) < ERROR_TOLERANCE
    ):
        return True
    else:
        return False


@njit(parallel=True, nogil=True, cache=True)
def constant_lon_intersections_no_extreme(lon, edge_node_x, edge_node_y, n_edge):
    """Determine which edges intersect a constant line of longitude on a
    sphere, without wrapping to the opposite longitude, with extremes
    along each great circle arc not considered.

    Parameters
    ----------
    lon:
        Constant longitude value in degrees.
    edge_node_x:
        Array of shape (n_edge, 2) containing x-coordinates of the edge nodes.
    edge_node_y:
        Array of shape (n_edge, 2) containing y-coordinates of the edge nodes.
    n_edge:
        Total number of edges to check.

    Returns
    -------
    intersecting_edges:
        array of indices of edges that intersect the constant longitude.
    """

    lon = np.deg2rad(lon)

    intersecting_edges_mask = np.zeros(n_edge, dtype=np.int32)

    # calculate the cos and sin of the constant longitude
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)

    for i in prange(n_edge):
        # get the x and y coordinates of the edge's nodes
        x0, x1 = edge_node_x[i, 0], edge_node_x[i, 1]
        y0, y1 = edge_node_y[i, 0], edge_node_y[i, 1]

        # calculate the dot products to determine on which side of the constant longitude the points lie
        dot0 = x0 * sin_lon - y0 * cos_lon
        dot1 = x1 * sin_lon - y1 * cos_lon

        # ensure that both points are not on the opposite longitude (180 degrees away)
        if (x0 * cos_lon + y0 * sin_lon) < 0.0 or (x1 * cos_lon + y1 * sin_lon) < 0.0:
            continue

        # check if the edge crosses the constant longitude or lies exactly on it
        if dot0 * dot1 < 0.0 or (
            abs(dot0) < ERROR_TOLERANCE and abs(dot1) < ERROR_TOLERANCE
        ):
            intersecting_edges_mask[i] = 1

    intersecting_edges = np.argwhere(intersecting_edges_mask)

    return np.unique(intersecting_edges)


@njit(cache=True)
def constant_lat_intersections_face_bounds(lat: float, face_bounds_lat: np.ndarray):
    """
    Identify candidate faces that intersect with a given constant latitude line.

    Parameters
    ----------
    lat : float
        The latitude in degrees for which to find intersecting faces.
    face_bounds_lat : numpy.ndarray
        A 2D array of shape (n_faces, 2), where each row represents the latitude
        bounds of a face. The first element of each row is the minimum latitude
        and the second element is the maximum latitude of the face.

    Returns
    -------
    candidate_faces : numpy.ndarray
        A 1D array of integers containing the indices of the faces that intersect
        the given latitude.
    """
    face_bounds_lat_min = face_bounds_lat[:, 0]
    face_bounds_lat_max = face_bounds_lat[:, 1]

    within_bounds = (face_bounds_lat_min <= lat) & (face_bounds_lat_max >= lat)
    candidate_faces = np.where(within_bounds)[0]
    return candidate_faces


@njit(cache=True)
def constant_lon_intersections_face_bounds(lon: float, face_bounds_lon: np.ndarray):
    """
    Identify candidate faces that intersect with a given constant longitude line.

    Parameters
    ----------
    lon : float
        The longitude in degrees for which to find intersecting faces.
    face_bounds_lon : numpy.ndarray
        A 2D array of shape (n_faces, 2), where each row represents the longitude
        bounds of a face. The first element of each row is the minimum longitude
        and the second element is the maximum longitude of the face.

    Returns
    -------
    candidate_faces : numpy.ndarray
        A 1D array of integers containing the indices of the faces that intersect
        the given longitude.
    """
    face_bounds_lon_min = face_bounds_lon[:, 0]
    face_bounds_lon_max = face_bounds_lon[:, 1]
    n_face = face_bounds_lon.shape[0]

    candidate_faces = []
    for i in range(n_face):
        cur_face_bounds_lon_min = face_bounds_lon_min[i]
        cur_face_bounds_lon_max = face_bounds_lon_max[i]

        if cur_face_bounds_lon_min < cur_face_bounds_lon_max:
            if (lon >= cur_face_bounds_lon_min) and (lon <= cur_face_bounds_lon_max):
                candidate_faces.append(i)
        else:
            # antimeridian case
            if (lon >= cur_face_bounds_lon_min) or (lon <= cur_face_bounds_lon_max):
                candidate_faces.append(i)

    return np.array(candidate_faces, dtype=INT_DTYPE)


@njit(cache=True)
def faces_within_lon_bounds(lons, face_bounds_lon):
    """
    Identify candidate faces that lie within a specified longitudinal interval.

    Parameters
    ----------
    lons : tuple or list of length 2
        A pair (min_lon, max_lon) specifying the query interval. If `min_lon <= max_lon`,
        the interval is [min_lon, max_lon]. If `min_lon > max_lon`, the interval
        crosses the antimeridian and should be interpreted as [min_lon, 180] U [-180, max_lon].
    face_bounds_lon : numpy.ndarray
        A 2D array of shape (n_faces, 2), where each row represents the longitude bounds
        of a face. The first element is the minimum longitude and the second is the maximum
        longitude for that face. Bounds may cross the antimeridian.

    Returns
    -------
    candidate_faces : numpy.ndarray
        A 1D array of integers containing the indices of the faces whose longitude bounds
        overlap with the specified interval.
    """
    face_bounds_lon_min = face_bounds_lon[:, 0]
    face_bounds_lon_max = face_bounds_lon[:, 1]
    n_face = face_bounds_lon.shape[0]

    min_lon, max_lon = lons

    # For example, a query of (160, -160) would cross the antimeridian
    antimeridian = min_lon > max_lon

    candidate_faces = []
    for i in range(n_face):
        cur_face_min = face_bounds_lon_min[i]
        cur_face_max = face_bounds_lon_max[i]

        # Check if the face itself crosses the antimeridian
        face_crosses_antimeridian = cur_face_min > cur_face_max

        if not antimeridian:
            # Normal case: min_lon <= max_lon
            # Face must be strictly contained within [min_lon, max_lon]
            if not face_crosses_antimeridian:
                if (cur_face_min >= min_lon) and (cur_face_max <= max_lon):
                    candidate_faces.append(i)
            else:
                # If face crosses antimeridian, it cannot be strictly contained
                # in a non-antimeridian query interval
                continue
        else:
            # Antimeridian case: interval crosses the -180/180 boundary
            # The query interval is effectively [min_lon, 180] U [-180, max_lon]

            if face_crosses_antimeridian:
                # If face crosses antimeridian, check if it's contained in the full query range
                if (cur_face_min >= min_lon) and (cur_face_max <= max_lon):
                    candidate_faces.append(i)
            else:
                # For non-crossing faces, check if they're strictly contained in either part
                contained_part1 = (cur_face_min >= min_lon) and (cur_face_max <= 180)
                contained_part2 = (cur_face_min >= -180) and (cur_face_max <= max_lon)

                if contained_part1 or contained_part2:
                    candidate_faces.append(i)

    return np.array(candidate_faces, dtype=INT_DTYPE)


@njit(cache=True)
def faces_within_lat_bounds(lats, face_bounds_lat):
    """
    Identify candidate faces that lie within a specified latitudinal interval.

    Parameters
    ----------
    lats : tuple or list of length 2
        A pair (min_lat, max_lat) specifying the query interval. All returned faces
        must be fully contained within this interval.
    face_bounds_lat : numpy.ndarray
        A 2D array of shape (n_faces, 2), where each row represents the latitude
        bounds of a face. The first element is the minimum latitude and the second
        is the maximum latitude for that face.

    Returns
    -------
    candidate_faces : numpy.ndarray
        A 1D array of integers containing the indices of the faces whose latitude
        bounds lie completely within the specified interval.
    """

    min_lat, max_lat = lats

    face_bounds_lat_min = face_bounds_lat[:, 0]
    face_bounds_lat_max = face_bounds_lat[:, 1]

    within_bounds = (face_bounds_lat_max <= max_lat) & (face_bounds_lat_min >= min_lat)
    candidate_faces = np.where(within_bounds)[0]
    return candidate_faces


@njit(cache=True, inline="always")
def _accux_gca(w0, w1, v0, v1):
    """Layer 1 — pure numerical kernel (mirrors AccuSphGeom ``accux_gca``).

    Computes the two antipodal candidate intersection points of the great-circle
    arcs w0-w1 and v0-v1.  No branching, no validity filtering.

    Returns
    -------
    pos, neg : np.ndarray, shape (3,)
        Two antipodal candidate unit vectors.
    """
    n1x_hi, n1y_hi, n1z_hi, n1x_lo, n1y_lo, n1z_lo = accucross(
        w0[0], w0[1], w0[2], w1[0], w1[1], w1[2]
    )
    n2x_hi, n2y_hi, n2z_hi, n2x_lo, n2y_lo, n2z_lo = accucross(
        v0[0], v0[1], v0[2], v1[0], v1[1], v1[2]
    )
    vx_hi, vy_hi, vz_hi, vx_lo, vy_lo, vz_lo = accucross_pair(
        n1x_hi,
        n1y_hi,
        n1z_hi,
        n1x_lo,
        n1y_lo,
        n1z_lo,
        n2x_hi,
        n2y_hi,
        n2z_hi,
        n2x_lo,
        n2y_lo,
        n2z_lo,
    )
    vx = vx_hi + vx_lo
    vy = vy_hi + vy_lo
    vz = vz_hi + vz_lo
    vn = math.sqrt(vx * vx + vy * vy + vz * vz)
    # Use np.inf safely when vn==0 (coplanar arcs): the resulting pos/neg
    # will be non-finite, so the status layer marks them invalid without branching.
    inv = 1.0 / vn if vn != 0.0 else np.inf
    pos = np.empty(3)
    pos[0] = vx * inv
    pos[1] = vy * inv
    pos[2] = vz * inv
    neg = np.empty(3)
    neg[0] = -pos[0]
    neg[1] = -pos[1]
    neg[2] = -pos[2]
    return pos, neg


@njit(cache=True)
def _try_gca_gca_intersection(w0, w1, v0, v1):
    """Layer 2 — batch/status layer (mirrors AccuSphGeom ``try_gca_gca_intersection``).

    Calls the pure numerical kernel, applies integer mask arithmetic to determine
    validity, selects the output point without if/else branching in the hot path.

    Status codes mirror AccuSphGeom:
        0  exactly one candidate is valid
        1  both candidates are valid
        2  neither candidate is valid  (includes coplanar/parallel case)
    """
    pos, neg = _accux_gca(w0, w1, v0, v1)

    pos_fin = (
        1
        if math.isfinite(pos[0]) and math.isfinite(pos[1]) and math.isfinite(pos[2])
        else 0
    )
    neg_fin = (
        1
        if math.isfinite(neg[0]) and math.isfinite(neg[1]) and math.isfinite(neg[2])
        else 0
    )
    pos_on_a = 1 if (pos_fin and on_minor_arc(pos, w0, w1)) else 0
    pos_on_b = 1 if (pos_fin and on_minor_arc(pos, v0, v1)) else 0
    neg_on_a = 1 if (neg_fin and on_minor_arc(neg, w0, w1)) else 0
    neg_on_b = 1 if (neg_fin and on_minor_arc(neg, v0, v1)) else 0

    pos_valid = pos_fin * pos_on_a * pos_on_b
    neg_valid = neg_fin * neg_on_a * neg_on_b

    pos_mask = pos_valid * (1 - neg_valid)
    neg_mask = neg_valid * (1 - pos_valid)

    point = np.empty(3)
    point[0] = pos_mask * pos[0] + neg_mask * neg[0]
    point[1] = pos_mask * pos[1] + neg_mask * neg[1]
    point[2] = pos_mask * pos[2] + neg_mask * neg[2]

    both = pos_valid * neg_valid
    none = (1 - pos_valid) * (1 - neg_valid)
    status = both + none * 2
    return point, status, pos, neg


@njit(cache=True)
def gca_gca_intersection(gca_a_xyz, gca_b_xyz):
    """Layer 3 — dispatcher / convenience API.

    Calls the batch/status layer and packages results into UXarray's existing
    array-returning API (0, 1, or 2 rows).  Coplanar/shared-endpoint handling
    lives here, outside the numerical core.
    """
    if gca_a_xyz.shape[1] != 3 or gca_b_xyz.shape[1] != 3:
        raise ValueError("The two GCAs must be in the cartesian [x, y, z] format")

    w0 = gca_a_xyz[0]
    w1 = gca_a_xyz[1]
    v0 = gca_b_xyz[0]
    v1 = gca_b_xyz[1]

    point, status, pos, neg = _try_gca_gca_intersection(w0, w1, v0, v1)

    res = np.empty((2, 3))
    count = 0
    if status == 0:
        res[0, 0] = point[0]
        res[0, 1] = point[1]
        res[0, 2] = point[2]
        count = 1
    elif status == 1:
        res[0, 0] = pos[0]
        res[0, 1] = pos[1]
        res[0, 2] = pos[2]
        res[1, 0] = neg[0]
        res[1, 1] = neg[1]
        res[1, 2] = neg[2]
        count = 2
    else:
        # status == 2: no candidate on both arcs.
        # Check for coplanar overlap (shared endpoints) outside the kernel.
        if on_minor_arc(v0, w0, w1):
            res[count, 0] = v0[0]
            res[count, 1] = v0[1]
            res[count, 2] = v0[2]
            count += 1
        if on_minor_arc(v1, w0, w1):
            res[count, 0] = v1[0]
            res[count, 1] = v1[1]
            res[count, 2] = v1[2]
            count += 1
    return res[:count]


@njit(cache=True, inline="always")
def _accux_constlat_scalar(a0, a1, a2, b0, b1, b2, const_z):
    """Layer 1 (scalar) — allocation-free numerical kernel.

    Same compensated AccuSphGeom sequence as :func:`_accux_constlat`, but takes
    the two arc endpoints as six scalars and returns the two candidate points as
    six scalars (``pos`` xy and ``neg`` xy; the z of both candidates is
    ``const_z``). Returning scalars instead of ``np.empty(3)`` arrays lets Numba
    keep everything in registers, so a batch loop over many edges does no
    per-point heap allocation. This is the preferred entry point for hot loops.

    Returns
    -------
    px, py, nx_out, ny_out : float
        ``pos = (px, py, const_z)`` and ``neg = (nx_out, ny_out, const_z)``.
        Invalid inputs propagate as non-finite coordinates.
    """
    nx_hi, ny_hi, nz_hi, nx_lo, ny_lo, nz_lo = accucross(a0, a1, a2, b0, b1, b2)
    s2_hi, s2_lo = _sum_of_squares_c((nx_hi, ny_hi), (nx_lo, ny_lo))
    denom = s2_hi + s2_lo
    s3_hi, s3_lo = _sum_of_squares_c((nx_hi, ny_hi, nz_hi), (nx_lo, ny_lo, nz_lo))
    zsq_hi, zsq_lo = two_prod(const_z, const_z)
    d_hi, d_lo = _cdp4(s3_hi, zsq_hi, s3_hi, zsq_lo, s3_lo, zsq_hi, s3_lo, zsq_lo)
    e_hi, e_lo = two_sum(s2_hi, -d_hi)
    planar_sq = e_hi + (e_lo + s2_lo - d_lo)
    s_root, s_corr = acc_sqrt_re(planar_sq)
    nx = nx_hi + nx_lo
    ny = ny_hi + ny_lo
    nz = nz_hi + nz_lo
    planar = s_root + s_corr
    xp_hi, xp_lo = _cdp2(nx * nz, const_z, -ny, planar)
    yp_hi, yp_lo = _cdp2(ny * nz, const_z, nx, planar)
    xn_hi, xn_lo = _cdp2(nx * nz, const_z, ny, planar)
    yn_hi, yn_lo = _cdp2(ny * nz, const_z, -nx, planar)
    # denom == 0 means the arc is vertical (normal has no x/y component).
    # Produce inf so the isfinite mask in the status layer rejects candidates.
    inv_denom = 1.0 / denom if denom != 0.0 else np.inf
    px = -(xp_hi + xp_lo) * inv_denom
    py = -(yp_hi + yp_lo) * inv_denom
    nxo = -(xn_hi + xn_lo) * inv_denom
    nyo = -(yn_hi + yn_lo) * inv_denom
    return px, py, nxo, nyo


@njit(cache=True, inline="always")
def _accux_constlat(x1, x2, const_z):
    """Layer 1 — pure numerical kernel (mirrors AccuSphGeom ``accux_constlat``).

    Array-returning wrapper around :func:`_accux_constlat_scalar`. Computes the
    two candidate intersection points between the great-circle arc defined by
    unit vectors *x1*, *x2* and the constant-latitude plane z = const_z. No
    branching, no validity filtering. For allocation-free hot loops call
    :func:`_accux_constlat_scalar` directly.

    Returns
    -------
    pos, neg : np.ndarray, shape (3,)
        Two antipodal candidate points.  Invalid inputs propagate as non-finite
        coordinates; the caller uses masks/status to identify validity.
    """
    px, py, nxo, nyo = _accux_constlat_scalar(
        x1[0], x1[1], x1[2], x2[0], x2[1], x2[2], const_z
    )
    pos = np.empty(3)
    pos[0] = px
    pos[1] = py
    pos[2] = const_z
    neg = np.empty(3)
    neg[0] = nxo
    neg[1] = nyo
    neg[2] = const_z
    return pos, neg


@njit(cache=True)
def _try_gca_const_lat_intersection(gca_cart, const_z):
    """Layer 2 — batch/status layer (mirrors AccuSphGeom ``try_gca_constlat_intersection``).

    Calls the pure numerical kernel, computes integer validity masks (0 or 1)
    for each candidate using finiteness and arc-membership tests, then selects
    the output point via integer arithmetic — no if/else branching in the hot path.

    Status codes mirror AccuSphGeom:
        0  exactly one candidate is valid  (normal case)
        1  both candidates are valid
        2  neither candidate is valid
    """
    x1 = gca_cart[0]
    x2 = gca_cart[1]
    pos, neg = _accux_constlat(x1, x2, const_z)

    pos_fin = int(math.isfinite(pos[0]) and math.isfinite(pos[1]))
    neg_fin = int(math.isfinite(neg[0]) and math.isfinite(neg[1]))
    pos_on = pos_fin * int(on_minor_arc(pos, x1, x2)) if pos_fin else 0
    neg_on = neg_fin * int(on_minor_arc(neg, x1, x2)) if neg_fin else 0

    pos_valid = pos_fin * pos_on
    neg_valid = neg_fin * neg_on

    pos_mask = pos_valid * (1 - neg_valid)
    neg_mask = neg_valid * (1 - pos_valid)

    point = np.empty(3)
    point[0] = pos_mask * pos[0] + neg_mask * neg[0]
    point[1] = pos_mask * pos[1] + neg_mask * neg[1]
    point[2] = pos_mask * pos[2] + neg_mask * neg[2]

    both = pos_valid * neg_valid
    none = (1 - pos_valid) * (1 - neg_valid)
    status = both + none * 2
    return point, status, pos, neg


@njit(cache=True)
def _snap_const_lat_endpoint(point, x1, x2, const_z):
    """Snap a candidate point to an arc endpoint when the endpoint lies on the latitude."""
    # 1e-14 is distance² in Cartesian between candidate and endpoint; corresponds
    # to ~1e-7 in arc length (unit sphere). Candidates within this distance are
    # snapped to the exact endpoint to avoid sub-ulp drift when the arc ends
    # exactly on the latitude circle.
    sx, sy = _snap_const_lat_endpoint_xy(
        point[0], point[1], x1[0], x1[1], x1[2], x2[0], x2[1], x2[2], const_z
    )
    out = np.empty(3)
    out[0] = sx
    out[1] = sy
    out[2] = point[2]
    return out


@njit(cache=True, inline="always")
def _snap_const_lat_endpoint_xy(px, py, a0, a1, a2, b0, b1, b2, const_z):
    """Scalar-argument form of :func:`_snap_const_lat_endpoint`.

    Returns the (possibly snapped) x, y of the candidate; z is always ``const_z``
    so it is not returned. Allocation-free for use in hot loops.
    """
    snap_sq = 1e-14
    ox = px
    oy = py
    if abs(a2 - const_z) <= ERROR_TOLERANCE:
        dx = ox - a0
        dy = oy - a1
        if dx * dx + dy * dy < snap_sq:
            ox = a0
            oy = a1
    if abs(b2 - const_z) <= ERROR_TOLERANCE:
        dx = ox - b0
        dy = oy - b1
        if dx * dx + dy * dy < snap_sq:
            ox = b0
            oy = b1
    return ox, oy


@njit(cache=True)
def gca_const_lat_intersection(gca_cart, const_z):
    """Layer 3 — dispatcher / convenience API.

    Runs the numerical kernel, validity masks, endpoint snapping, and packaging
    into UXarray's NaN-filled (2, 3) format entirely on scalars, so the only heap
    allocation is the returned array. All UXarray-specific branching lives here so
    the numerical core stays uniform. See ``_try_gca_const_lat_intersection`` for
    the array-returning form used by the layer benchmarks.
    """
    res = np.empty((2, 3))
    res.fill(np.nan)

    a0 = gca_cart[0, 0]
    a1 = gca_cart[0, 1]
    a2 = gca_cart[0, 2]
    b0 = gca_cart[1, 0]
    b1 = gca_cart[1, 1]
    b2 = gca_cart[1, 2]

    px, py, nx, ny = _accux_constlat_scalar(a0, a1, a2, b0, b1, b2, const_z)

    pos_fin = math.isfinite(px) and math.isfinite(py)
    neg_fin = math.isfinite(nx) and math.isfinite(ny)
    pos_valid = pos_fin and _on_minor_arc_xyz(px, py, const_z, a0, a1, a2, b0, b1, b2)
    neg_valid = neg_fin and _on_minor_arc_xyz(nx, ny, const_z, a0, a1, a2, b0, b1, b2)

    if pos_valid and not neg_valid:
        sx, sy = _snap_const_lat_endpoint_xy(px, py, a0, a1, a2, b0, b1, b2, const_z)
        res[0, 0] = sx
        res[0, 1] = sy
        res[0, 2] = const_z
    elif neg_valid and not pos_valid:
        sx, sy = _snap_const_lat_endpoint_xy(nx, ny, a0, a1, a2, b0, b1, b2, const_z)
        res[0, 0] = sx
        res[0, 1] = sy
        res[0, 2] = const_z
    elif pos_valid and neg_valid:
        psx, psy = _snap_const_lat_endpoint_xy(px, py, a0, a1, a2, b0, b1, b2, const_z)
        nsx, nsy = _snap_const_lat_endpoint_xy(nx, ny, a0, a1, a2, b0, b1, b2, const_z)
        dx = psx - nsx
        dy = psy - nsy
        if dx * dx + dy * dy < 1e-14:
            res[0, 0] = psx
            res[0, 1] = psy
            res[0, 2] = const_z
        else:
            res[0, 0] = psx
            res[0, 1] = psy
            res[0, 2] = const_z
            res[1, 0] = nsx
            res[1, 1] = nsy
            res[1, 2] = const_z
    return res


@njit(cache=True)
def get_number_of_intersections(arr):
    """Returns the number of intersection points for the output of the gca-const-lat intersection."""
    row1_is_nan = np.all(np.isnan(arr[0]))
    row2_is_nan = np.all(np.isnan(arr[1]))

    if row1_is_nan and row2_is_nan:
        return 0
    elif row2_is_nan:
        return 1
    else:
        return 2
