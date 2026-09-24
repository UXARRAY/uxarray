import math

import numpy as np
from numba import njit

from uxarray.constants import ERROR_TOLERANCE, INT_DTYPE
from uxarray.errors import DimensionError
from uxarray.grid.arcs import on_minor_arc
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
from uxarray.utils.numba_math import (
    _numba_add3,
    _numba_allfinite3,
    _numba_mul3_scalar,
    _numba_neg3,
)

# Edge screeners: fast O(n) passes used by Grid.get_edges_at_constant_* to
# identify candidate edges before the expensive GCA intersection. "no_extreme"
# means arc z-extrema along the great circle are not considered.
#
# These are deliberately plain NumPy (no @njit): they are memory-bound elementwise
# predicates where NumPy is ~2x faster than a Numba prange loop,


def _flatnonzero(mask):
    """Sorted indices where a 1-D boolean mask is True"""
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
    in_antimeridian = (~normal) & (
        (lon >= face_bounds_lon_min) | (lon <= face_bounds_lon_max)
    )

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
    w0, w1 : iterables of length 3
        Cartesian endpoints of the first arc.
    v0, v1 : iterables of length 3
        Cartesian endpoints of the second arc.

    Returns
    -------
    pos, neg : tuples of length 3
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
    # so pos/neg become non-finite and the status layer masks them out.
    inv = 1.0 / vn
    pos = _numba_mul3_scalar((vx, vy, vz), inv)
    neg = _numba_neg3(pos)
    return pos, neg


@njit(cache=True, inline="always", error_model="numpy")
def _try_gca_gca_intersection(w0, w1, v0, v1):
    """Select the valid great-circle intersection and report a status code.
    Returns point, status, pos, neg.

    Batch/status layer (mirrors AccuSphGeom ``try_gca_gca_intersection``).

    Calls the pure numerical kernel, applies integer mask arithmetic to determine
    validity, selects the output point without if/else branching in the hot path.

    Parameters
    ----------
    w0, w1 : iterables of length 3
        Cartesian endpoints of the first arc.
    v0, v1 : iterables of length 3
        Cartesian endpoints of the second arc.

    Returns
    -------
    point : tuple of length 3
        The single valid intersection point, if status == 0, else meaningless.
    status : int
        Status codes mirror AccuSphGeom:
        0  exactly one candidate is valid (point)
        1  both candidates are valid (pos and neg)
        2  neither candidate is valid  (includes coplanar/parallel case)
    pos : tuple of length 3
        The positive candidate intersection point (antipodal to neg).
    neg : tuple of length 3
        The negative candidate intersection point (antipodal to pos).
    """
    pos, neg = _accux_gca(w0, w1, v0, v1)

    pos_fin = _numba_allfinite3(pos)
    neg_fin = _numba_allfinite3(neg)
    pos_on_a = pos_fin * on_minor_arc(pos, w0, w1)
    pos_on_b = pos_fin * on_minor_arc(pos, v0, v1)
    neg_on_a = neg_fin * on_minor_arc(neg, w0, w1)
    neg_on_b = neg_fin * on_minor_arc(neg, v0, v1)

    pos_valid = pos_fin * pos_on_a * pos_on_b
    neg_valid = neg_fin * neg_on_a * neg_on_b

    pos_mask = pos_valid * (1 - neg_valid)
    neg_mask = neg_valid * (1 - pos_valid)

    point = _numba_add3(
        _numba_mul3_scalar(pos, pos_mask), _numba_mul3_scalar(neg, neg_mask)
    )

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
    gca_a_xyz : iterable of 2 length-3 iterables
        First great-circle arc as two Cartesian endpoints.
        (If numpy array, has shape (2,3). If tuple, contains two length-3 tuples.)
    gca_b_xyz : iterable of 2 length-3 iterables
        Second great-circle arc as two Cartesian endpoints.
        (If numpy array, has shape (2,3). If tuple, contains two length-3 tuples.)

    Returns
    -------
    intersections : tuple of 2 length-3 tuples
        The (x,y,z) coordinates of the intersections, filling with NaNs as needed.
        E.g. if there is one intersection, returns ((x1,y1,z1) (nan,nan,nan)).

    References
    ----------
    Chen, H., Ullrich, P. A., Panetta, J., Marsico, D., Hanke, M., Jain, R.,
    Zhang, C., and Jacob, R. L. (2026). Accurate and robust geometric
    algorithms for regridding on the sphere. Geoscientific Model
    Development, 19(14), 6545-6570. https://doi.org/10.5194/gmd-19-6545-2026

    Chen, H., Ullrich, P. A., and Panetta, J. (2026). Fast and accurate
    intersections on a sphere. SIAM Journal on Scientific Computing, 48(2),
    B208-B232. https://doi.org/10.1137/25M1737614
    """
    if len(gca_a_xyz) != 2 or len(gca_b_xyz) != 2:
        raise DimensionError("Each input to gca_gca_intersection must have length 2")
    if len(gca_a_xyz[0]) != 3 or len(gca_a_xyz[1]) != 3:
        raise DimensionError(
            "gca_a points must be in cartesian format (x,y,z), but got len(gca_a[0]) != 3"
            "or len(gca_a[1]) != 3, in gca_gca_intersection(gca_a, gca_b)"
        )
    if len(gca_b_xyz[0]) != 3 or len(gca_b_xyz[1]) != 3:
        raise DimensionError(
            "gca_b points must be in cartesian format (x,y,z), but got len(gca_b[0]) != 3"
            "or len(gca_b[1]) != 3, in gca_gca_intersection(gca_a, gca_b)"
        )

    w0 = gca_a_xyz[0]
    w1 = gca_a_xyz[1]
    v0 = gca_b_xyz[0]
    v1 = gca_b_xyz[1]

    point, status, pos, neg = _try_gca_gca_intersection(w0, w1, v0, v1)

    # Always return two points; branching logic changing output shape while
    # using tuple outputs causes numba crash like "Can't unify return type".
    # (And, swapping to tiny numpy array outputs causes significant slowdown.)
    if status == 0:  # one intersection point
        result = (point, (np.nan, np.nan, np.nan))
    elif status == 1:
        result = (pos, neg)
    else:
        # status == 2: no candidate on both arcs.
        # Check for coplanar overlap (shared endpoints) outside the kernel.
        v0_on_w_arc = on_minor_arc(v0, w0, w1)
        v1_on_w_arc = on_minor_arc(v1, w0, w1)
        if v0_on_w_arc or v1_on_w_arc:
            # Ensure result will be (tuple,tuple), to avoid "Can't unify return type",
            # which occurs if there is any chance of (array,tuple) or (array,array).
            v0 = (v0[0], v0[1], v0[2])
            v1 = (v1[0], v1[1], v1[2])
            if v0_on_w_arc and v1_on_w_arc:
                result = (v0, v1)
            elif v0_on_w_arc:
                result = (v0, (np.nan, np.nan, np.nan))
            elif v1_on_w_arc:
                result = (v1, (np.nan, np.nan, np.nan))
        else:
            result = ((np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan))
    return result


@njit(cache=True, inline="always", error_model="numpy")
def _accux_constlat(a, b, const_z):
    """Compute the two constant-latitude intersection candidates as tuples

    Pure numerical kernel (mirrors AccuSphGeom ``accux_constlat``). Computes the
    two candidate intersection points between the great-circle arc defined by
    unit vectors *x1*, *x2* and the constant-latitude plane z = const_z. No
    branching, no validity filtering.

    Parameters
    ----------
    a, b : iterables of length 3
        Cartesian endpoints of the great-circle arc.
    const_z : float
        Constant-latitude plane, given as the Cartesian z value ``sin(lat)``.

    Returns
    -------
    pos, neg : tuples of length 3
        ``pos = (px, py, const_z)`` and ``neg = (nx_out, ny_out, const_z)``.
        Invalid inputs propagate as non-finite coordinates.
    """
    nx_hi, ny_hi, nz_hi, nx_lo, ny_lo, nz_lo = accucross(
        a[0], a[1], a[2], b[0], b[1], b[2]
    )
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
    return (px, py, const_z), (nxo, nyo, const_z)


@njit(cache=True, error_model="numpy")
def _try_gca_const_lat_intersection(gca_cart, const_z):
    """Select the valid constant-latitude intersection and report a status code.
    Returns point, status, pos, neg.

    Batch/status layer (mirrors AccuSphGeom ``try_gca_constlat_intersection``).

    Calls the pure numerical kernel, computes integer validity masks (0 or 1)
    for each candidate using finiteness and arc-membership tests, then selects
    the output point via integer arithmetic — no if/else branching in the hot path.

    Parameters
    ----------
    gca_cart : iterable of 2 length-3 iterables
        Cartesian endpoints of the first arc.
    const_z : float
        Constant-latitude plane, given as the Cartesian z value ``sin(lat)``.

    Returns
    -------
    point : tuple of length 3
        The single valid intersection point, if status == 0, else meaningless.
    status : int
        Status codes mirror AccuSphGeom:
        0  exactly one candidate is valid (point) (this is the normal case)
        1  both candidates are valid (pos and neg)
        2  neither candidate is valid
    pos, neg : tuple of length 3
        The candidate intersection points.
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

    point = _numba_add3(
        _numba_mul3_scalar(pos, pos_mask), _numba_mul3_scalar(neg, neg_mask)
    )

    both = pos_valid * neg_valid
    none = (1 - pos_valid) * (1 - neg_valid)
    status = both + none * 2
    return point, status, pos, neg


@njit(cache=True)
def _snap_const_lat_endpoint(point, a, b, const_z):
    """Snap a candidate point to an arc endpoint when the endpoint lies on the latitude.
    Returns (x,y,z) of the possibly-snapped point.

    point, a, b : iterables of length 3
        Cartesian coordinates of the candidate point and the two arc endpoints.
    const_z : float
        Constant-latitude plane, given as the Cartesian z value ``sin(lat)``.
    """
    # 1e-14 is distance² in Cartesian between candidate and endpoint; corresponds
    # to ~1e-7 in arc length (unit sphere). Candidates within this distance are
    # snapped to the exact endpoint to avoid sub-ulp drift when the arc ends
    # exactly on the latitude circle.
    snap_sq = 1e-14
    ox = point[0]
    oy = point[1]
    if abs(a[2] - const_z) <= ERROR_TOLERANCE:
        dx = ox - a[0]
        dy = oy - a[1]
        if dx * dx + dy * dy < snap_sq:
            ox = a[0]
            oy = a[1]
    if abs(b[2] - const_z) <= ERROR_TOLERANCE:
        dx = ox - b[0]
        dy = oy - b[1]
        if dx * dx + dy * dy < snap_sq:
            ox = b[0]
            oy = b[1]
    return (ox, oy, point[2])


@njit(cache=True, error_model="numpy")
def gca_const_lat_intersection(gca_cart, const_z):
    """Return the intersection points of a great-circle arc and a latitude.

    Dispatcher / convenience API. Runs the numerical kernel, validity masks,
    endpoint snapping, and packaging into UXarray's NaN-filled (2,3) format.
    All UXarray-specific branching lives here so the numerical core stays uniform.
    See also: ``_try_gca_const_lat_intersection``.

    Parameters
    ----------
    gca_cart : iterable of 2 length-3 iterables
        Great-circle arc as two Cartesian endpoints.
        (If numpy array, has shape (2,3). If tuple, contains two length-3 tuples.)
    const_z : float
        Constant-latitude plane, given as the Cartesian z value ``sin(lat)``.

    Returns
    -------
    intersections : tuple of 2 length-3 tuples
        The (x,y,z) coordinates of the intersections, filling with NaNs as needed.
        E.g. if there is one intersection, returns ((x1,y1,z1),(nan,nan,nan)).

    References
    ----------
    Chen, H., Ullrich, P. A., Panetta, J., Marsico, D., Hanke, M., Jain, R.,
    Zhang, C., and Jacob, R. L. (2026). Accurate and robust geometric
    algorithms for regridding on the sphere. Geoscientific Model
    Development, 19(14), 6545-6570. https://doi.org/10.5194/gmd-19-6545-2026

    Chen, H., Ullrich, P. A., and Panetta, J. (2026). Fast and accurate
    intersections on a sphere. SIAM Journal on Scientific Computing, 48(2),
    B208-B232. https://doi.org/10.1137/25M1737614
    """
    a = gca_cart[0]
    b = gca_cart[1]

    pos, neg = _accux_constlat(a, b, const_z)

    pos_fin = int(math.isfinite(pos[0])) * int(math.isfinite(pos[1]))
    neg_fin = int(math.isfinite(neg[0])) * int(math.isfinite(neg[1]))
    pos_valid = pos_fin * on_minor_arc(pos, a, b)
    neg_valid = neg_fin * on_minor_arc(neg, a, b)

    if pos_valid ^ neg_valid:
        # exactly 1 valid intersection point
        if pos_valid:
            point_snapped = _snap_const_lat_endpoint(pos, a, b, const_z)
        else:
            point_snapped = _snap_const_lat_endpoint(neg, a, b, const_z)
        result = (point_snapped, (np.nan, np.nan, np.nan))
    elif pos_valid and neg_valid:
        # probably 2 valid intersection points
        pos_snapped = _snap_const_lat_endpoint(pos, a, b, const_z)
        neg_snapped = _snap_const_lat_endpoint(neg, a, b, const_z)
        dx = pos_snapped[0] - neg_snapped[0]
        dy = pos_snapped[1] - neg_snapped[1]
        if dx * dx + dy * dy < 1e-14:
            # (actually, they are the same point! --> Only 1 valid point.)
            result = (pos_snapped, (np.nan, np.nan, np.nan))
        else:
            result = (pos_snapped, neg_snapped)
    else:
        # 0 valid intersection points
        result = ((np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan))
    return result


@njit(cache=True)
def get_number_of_intersections(arr):
    """Return the number of intersection points in a gca intersections result.

    Parameters
    ----------
    arr : iterable of 2 length-3 iterables
        Intersection points from a gca intersections function, e.g. one of:
        :func:`gca_gca_intersection` or :func:`gca_const_lat_intersection`.
        (If numpy array, has shape (2,3). If tuple, contains two length-3 tuples.)
        NaN values indicate no intersection at that point.

    Returns
    -------
    int
        Number of non-NaN intersection points (0, 1, or 2).

    References
    ----------
    Chen, H., Ullrich, P. A., Panetta, J., Marsico, D., Hanke, M., Jain, R.,
    Zhang, C., and Jacob, R. L. (2026). Accurate and robust geometric
    algorithms for regridding on the sphere. Geoscientific Model
    Development, 19(14), 6545-6570. https://doi.org/10.5194/gmd-19-6545-2026

    Chen, H., Ullrich, P. A., and Panetta, J. (2026). Fast and accurate
    intersections on a sphere. SIAM Journal on Scientific Computing, 48(2),
    B208-B232. https://doi.org/10.1137/25M1737614
    """
    row1_is_nan = np.isnan(arr[0][0]) * np.isnan(arr[0][1]) * np.isnan(arr[0][2])
    row2_is_nan = np.isnan(arr[1][0]) * np.isnan(arr[1][1]) * np.isnan(arr[1][2])

    if row1_is_nan and row2_is_nan:
        return 0
    elif row2_is_nan:
        return 1
    else:
        return 2
