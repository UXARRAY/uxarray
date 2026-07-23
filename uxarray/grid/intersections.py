import math

import numpy as np
from numba import njit

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

# Edge/face screeners: O(n) elementwise passes used by Grid.get_edges_at_constant_*
# and get_faces_* to identify candidate edges/faces before the expensive GCA
# intersection. "no_extreme" means arc z-extrema along the great circle are not
# considered.
#
# These are deliberately plain NumPy (no @njit): they are memory-bound elementwise
# predicates where NumPy is ~2x faster than a Numba prange loop, and — unlike an
# njit kernel, which forces a full ``.values`` materialization at the call site —
# they compose with dask. When handed a dask array they reduce block-by-block via
# :func:`_flatnonzero`, so peak memory is one chunk plus the (small) index result
# rather than the whole coordinate array.


def _flatnonzero(mask):
    """Sorted indices where a 1-D boolean mask is True, numpy- or dask-backed."""
    return np.asarray(np.flatnonzero(mask))


def constant_lat_intersections_no_extreme(lat, edge_node_z):
    """Determine which edges intersect a constant line of latitude on a
    sphere, without wrapping to the opposite longitude, with extremes
    along each great circle arc not considered.

    Parameters
    ----------
    lat:
        Constant latitude value in degrees.
        May be NumPy or dask array.
    edge_node_z:
        Array of shape (n_edge, 2) containing z-coordinates of the edge nodes.
        May be NumPy or dask array.

    Returns
    -------
    intersecting_edges:
        array of indices of edges that intersect the constant latitude.
    """
    z_constant = np.sin(np.deg2rad(lat))

    d0 = edge_node_z[:, 0] - z_constant
    d1 = edge_node_z[:, 1] - z_constant

    # Edge crosses the latitude (endpoints straddle it) or lies exactly on it.
    intersecting = (d0 * d1 < 0.0) | (
        (np.abs(d0) < ERROR_TOLERANCE) & (np.abs(d1) < ERROR_TOLERANCE)
    )

    return _flatnonzero(intersecting)


def constant_lon_intersections_no_extreme(lon, edge_node_x, edge_node_y):
    """Determine which edges intersect a constant line of longitude on a
    sphere, without wrapping to the opposite longitude, with extremes
    along each great circle arc not considered.

    Parameters
    ----------
    lon:
        Constant longitude value in degrees.
    edge_node_x:
        Array of shape (n_edge, 2) containing x-coordinates of the edge nodes.
        May be NumPy or dask array.
    edge_node_y:
        Array of shape (n_edge, 2) containing y-coordinates of the edge nodes.
        May be NumPy or dask array.

    Returns
    -------
    intersecting_edges:
        array of indices of edges that intersect the constant longitude.
    """
    lon = np.deg2rad(lon)
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)

    x0 = edge_node_x[:, 0]
    x1 = edge_node_x[:, 1]
    y0 = edge_node_y[:, 0]
    y1 = edge_node_y[:, 1]

    # Signed distance to the meridian plane for each endpoint.
    dot0 = x0 * sin_lon - y0 * cos_lon
    dot1 = x1 * sin_lon - y1 * cos_lon

    # Discard edges with an endpoint on the opposite meridian (180 deg away).
    not_opposite = (x0 * cos_lon + y0 * sin_lon >= 0.0) & (
        x1 * cos_lon + y1 * sin_lon >= 0.0
    )

    # Edge crosses the longitude or lies exactly on it.
    crosses = (dot0 * dot1 < 0.0) | (
        (np.abs(dot0) < ERROR_TOLERANCE) & (np.abs(dot1) < ERROR_TOLERANCE)
    )

    return _flatnonzero(not_opposite & crosses)


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
    within_bounds = (face_bounds_lat[:, 0] <= lat) & (face_bounds_lat[:, 1] >= lat)
    return _flatnonzero(within_bounds)


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

    # Normal faces (min < max): lon inside [min, max].
    normal = face_bounds_lon_min < face_bounds_lon_max
    in_normal = normal & (lon >= face_bounds_lon_min) & (lon <= face_bounds_lon_max)
    # Antimeridian faces (min >= max): lon >= min OR lon <= max.
    in_antimeridian = (~normal) & ((lon >= face_bounds_lon_min) | (lon <= face_bounds_lon_max))

    return _flatnonzero(in_normal | in_antimeridian).astype(INT_DTYPE, copy=False)


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

    min_lon, max_lon = lons

    # For example, a query of (160, -160) would cross the antimeridian
    antimeridian = min_lon > max_lon

    # A face itself crosses the antimeridian when its stored min > max.
    face_crosses = face_bounds_lon_min > face_bounds_lon_max

    if not antimeridian:
        # Normal query interval [min_lon, max_lon]. Only non-crossing faces can
        # be strictly contained; crossing faces are excluded.
        mask = (
            (~face_crosses)
            & (face_bounds_lon_min >= min_lon)
            & (face_bounds_lon_max <= max_lon)
        )
    else:
        # Antimeridian query: effectively [min_lon, 180] U [-180, max_lon].
        crossing_contained = (
            face_crosses
            & (face_bounds_lon_min >= min_lon)
            & (face_bounds_lon_max <= max_lon)
        )
        noncrossing_contained = (~face_crosses) & (
            ((face_bounds_lon_min >= min_lon) & (face_bounds_lon_max <= 180))
            | ((face_bounds_lon_min >= -180) & (face_bounds_lon_max <= max_lon))
        )
        mask = crossing_contained | noncrossing_contained

    return _flatnonzero(mask).astype(INT_DTYPE, copy=False)


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
    return _flatnonzero(within_bounds)


@njit(cache=True, inline="always", error_model="numpy")
def _accux_gca(w0, w1, v0, v1):
    """Compute the candidate intersection points of two great-circle arcs.

    Pure numerical kernel (mirrors AccuSphGeom ``accux_gca``).

    Computes the two antipodal candidate intersection points of the great-circle
    arcs w0-w1 and v0-v1.  No branching, no validity filtering.

    Parameters
    ----------
    w0, w1 : np.ndarray, shape (3,)
        Cartesian endpoints of the first arc.
    v0, v1 : np.ndarray, shape (3,)
        Cartesian endpoints of the second arc.

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
    # Compensated norm: sum_of_squares_c over the (hi, lo) vector, then acc_sqrt_re
    # folding the low part into the root, matching AccuSphGeom accux_gca. n = root.hi.
    sum_hi, sum_lo = _sum_of_squares_c((vx_hi, vy_hi, vz_hi), (vx_lo, vy_lo, vz_lo))
    vn, _ = acc_sqrt_re(sum_hi, sum_lo)
    # vn==0 (coplanar arcs) yields inf via IEEE division under error_model="numpy",
    # so pos/neg become non-finite and the status layer masks them out. Branch-free.
    inv = 1.0 / vn
    pos = np.empty(3)
    pos[0] = vx * inv
    pos[1] = vy * inv
    pos[2] = vz * inv
    neg = np.empty(3)
    neg[0] = -pos[0]
    neg[1] = -pos[1]
    neg[2] = -pos[2]
    return pos, neg


@njit(cache=True, error_model="numpy")
def _try_gca_gca_intersection(w0, w1, v0, v1):
    """Select the valid great-circle intersection and report a status code.

    Batch/status layer (mirrors AccuSphGeom ``try_gca_gca_intersection``).

    Calls the pure numerical kernel, applies integer mask arithmetic to determine
    validity, selects the output point without if/else branching in the hot path.

    Status codes mirror AccuSphGeom:
        0  exactly one candidate is valid
        1  both candidates are valid
        2  neither candidate is valid  (includes coplanar/parallel case)
    """
    pos, neg = _accux_gca(w0, w1, v0, v1)

    pos_fin = (
        int(math.isfinite(pos[0]))
        * int(math.isfinite(pos[1]))
        * int(math.isfinite(pos[2]))
    )
    neg_fin = (
        int(math.isfinite(neg[0]))
        * int(math.isfinite(neg[1]))
        * int(math.isfinite(neg[2]))
    )
    pos_on_a = pos_fin * on_minor_arc(pos, w0, w1)
    pos_on_b = pos_fin * on_minor_arc(pos, v0, v1)
    neg_on_a = neg_fin * on_minor_arc(neg, w0, w1)
    neg_on_b = neg_fin * on_minor_arc(neg, v0, v1)

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


@njit(cache=True, error_model="numpy")
def gca_gca_intersection(gca_a_xyz, gca_b_xyz):
    """Return the intersection points of two great-circle arcs.

    Dispatcher / convenience API. Calls the batch/status layer and packages
    results into UXarray's existing array-returning API. Coplanar/shared-endpoint
    handling lives here, outside the numerical core.

    Parameters
    ----------
    gca_a_xyz : numpy.ndarray
        First great-circle arc as two Cartesian endpoints, shape ``(2, 3)``.
    gca_b_xyz : numpy.ndarray
        Second great-circle arc as two Cartesian endpoints, shape ``(2, 3)``.

    Returns
    -------
    numpy.ndarray
        Intersection points, shape ``(2, 3)``, with unused rows filled with NaN
        (0, 1, or 2 valid rows).
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


@njit(cache=True, inline="always", error_model="numpy")
def _accux_constlat_scalar(a0, a1, a2, b0, b1, b2, const_z):
    """Compute the constant-latitude intersection candidates, scalar in/out.

    Allocation-free numerical kernel. Same compensated AccuSphGeom sequence as
    :func:`_accux_constlat`, but takes the two arc endpoints as six scalars and
    returns the two candidate points as six scalars (``pos`` xy and ``neg`` xy;
    the z of both candidates is ``const_z``). Returning scalars instead of
    ``np.empty(3)`` arrays lets Numba
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
    # denom == 0 (vertical arc) yields inf via IEEE division under
    # error_model="numpy", so the isfinite mask in the status layer rejects the
    # candidates. Branch-free.
    inv_denom = 1.0 / denom
    px = -(xp_hi + xp_lo) * inv_denom
    py = -(yp_hi + yp_lo) * inv_denom
    nxo = -(xn_hi + xn_lo) * inv_denom
    nyo = -(yn_hi + yn_lo) * inv_denom
    return px, py, nxo, nyo


@njit(cache=True, inline="always")
def _accux_constlat(x1, x2, const_z):
    """Compute the two constant-latitude intersection candidates as arrays.

    Pure numerical kernel (mirrors AccuSphGeom ``accux_constlat``). An
    array-returning wrapper around :func:`_accux_constlat_scalar`. Computes the
    two candidate intersection points between the great-circle arc defined by
    unit vectors *x1*, *x2* and the constant-latitude plane z = const_z. No
    branching, no validity filtering. For allocation-free hot loops call
    :func:`_accux_constlat_scalar` directly.

    Parameters
    ----------
    x1, x2 : np.ndarray, shape (3,)
        Cartesian endpoints of the great-circle arc.
    const_z : float
        Constant-latitude plane, given as the Cartesian z value ``sin(lat)``.

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


@njit(cache=True, error_model="numpy")
def _try_gca_const_lat_intersection(gca_cart, const_z):
    """Select the valid constant-latitude intersection and report a status code.

    Batch/status layer (mirrors AccuSphGeom ``try_gca_constlat_intersection``).

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

    pos_fin = int(math.isfinite(pos[0])) * int(math.isfinite(pos[1]))
    neg_fin = int(math.isfinite(neg[0])) * int(math.isfinite(neg[1]))
    pos_on = pos_fin * on_minor_arc(pos, x1, x2)
    neg_on = neg_fin * on_minor_arc(neg, x1, x2)

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


@njit(cache=True, error_model="numpy")
def gca_const_lat_intersection(gca_cart, const_z):
    """Return the intersection points of a great-circle arc and a latitude.

    Dispatcher / convenience API. Runs the numerical kernel, validity masks,
    endpoint snapping, and packaging into UXarray's NaN-filled (2, 3) format
    entirely on scalars, so the only heap allocation is the returned array. All
    UXarray-specific branching lives here so the numerical core stays uniform.
    See ``_try_gca_const_lat_intersection`` for the array-returning form used by
    the layer benchmarks.

    Parameters
    ----------
    gca_cart : numpy.ndarray
        Great-circle arc as two Cartesian endpoints, shape ``(2, 3)``.
    const_z : float
        Constant-latitude plane, given as the Cartesian z value ``sin(lat)``.

    Returns
    -------
    numpy.ndarray
        Intersection points, shape ``(2, 3)``, with unused rows filled with NaN
        (0, 1, or 2 valid rows).
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

    pos_fin = int(math.isfinite(px)) * int(math.isfinite(py))
    neg_fin = int(math.isfinite(nx)) * int(math.isfinite(ny))
    pos_valid = pos_fin * _on_minor_arc_xyz(px, py, const_z, a0, a1, a2, b0, b1, b2)
    neg_valid = neg_fin * _on_minor_arc_xyz(nx, ny, const_z, a0, a1, a2, b0, b1, b2)

    if pos_valid ^ neg_valid:
        if pos_valid:
            sx, sy = _snap_const_lat_endpoint_xy(
                px, py, a0, a1, a2, b0, b1, b2, const_z
            )
        else:
            sx, sy = _snap_const_lat_endpoint_xy(
                nx, ny, a0, a1, a2, b0, b1, b2, const_z
            )
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
    """Return the number of intersection points in a gca-const-lat result.

    Parameters
    ----------
    arr : numpy.ndarray
        Output of :func:`gca_const_lat_intersection`, shape ``(2, 3)`` with
        unused rows filled with NaN.

    Returns
    -------
    int
        Number of non-NaN intersection points (0, 1, or 2).
    """
    row1_is_nan = np.all(np.isnan(arr[0]))
    row2_is_nan = np.all(np.isnan(arr[1]))

    if row1_is_nan and row2_is_nan:
        return 0
    elif row2_is_nan:
        return 1
    else:
        return 2
