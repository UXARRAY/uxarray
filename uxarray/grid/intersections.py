import math

import numpy as np
from numba import njit, prange

from uxarray.constants import ERROR_TOLERANCE, INT_DTYPE
from uxarray.grid.arcs import on_minor_arc
from uxarray.utils.computing import (
    _cdp2,
    _cdp4,
    _sum_sq_c2,
    _sum_sq_c3,
    acc_sqrt_re,
    accucross,
    accucross_pair,
    two_prod,
    two_sum,
)


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


def _gca_gca_intersection_cartesian(gca_a_xyz, gca_b_xyz):
    gca_a_xyz = np.asarray(gca_a_xyz)
    gca_b_xyz = np.asarray(gca_b_xyz)

    return gca_gca_intersection(gca_a_xyz, gca_b_xyz)


@njit(cache=True)
def gca_gca_intersection(gca_a_xyz, gca_b_xyz):
    """Find intersection point(s) of two great-circle arcs using compensated arithmetic.

    Uses ``accucross`` (compensated cross products) and ``on_minor_arc`` (compensated
    arc membership) to avoid the catastrophic cancellation that affects naive
    cross product implementations when arcs are nearly parallel.

    Parameters
    ----------
    gca_a_xyz : np.ndarray, shape (2, 3)
        Cartesian endpoints of the first great-circle arc.
    gca_b_xyz : np.ndarray, shape (2, 3)
        Cartesian endpoints of the second great-circle arc.

    Returns
    -------
    np.ndarray, shape (n, 3)
        Intersection points lying on both arcs; n is 0, 1, or 2.
    """
    if gca_a_xyz.shape[1] != 3 or gca_b_xyz.shape[1] != 3:
        raise ValueError("The two GCAs must be in the cartesian [x, y, z] format")

    w0 = gca_a_xyz[0]
    w1 = gca_a_xyz[1]
    v0 = gca_b_xyz[0]
    v1 = gca_b_xyz[1]

    # 1. Plane normals via accurate cross products — keep compensated (hi, lo).
    n1x_hi, n1y_hi, n1z_hi, n1x_lo, n1y_lo, n1z_lo = accucross(
        w0[0], w0[1], w0[2], w1[0], w1[1], w1[2]
    )
    n2x_hi, n2y_hi, n2z_hi, n2x_lo, n2y_lo, n2z_lo = accucross(
        v0[0], v0[1], v0[2], v1[0], v1[1], v1[2]
    )

    res = np.empty((2, 3))
    count = 0

    # Degenerate check: collapsed (zero-length) input arc.
    n1x = n1x_hi + n1x_lo
    n1y = n1y_hi + n1y_lo
    n1z = n1z_hi + n1z_lo
    n2x = n2x_hi + n2x_lo
    n2y = n2y_hi + n2y_lo
    n2z = n2z_hi + n2z_lo
    if (
        n1x * n1x + n1y * n1y + n1z * n1z == 0.0
        or n2x * n2x + n2y * n2y + n2z * n2z == 0.0
    ):
        return res[:count]

    # 2. Intersection direction: compensated cross of the two plane normals.
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

    if vn == 0.0 or not (
        math.isfinite(vx)
        and math.isfinite(vy)
        and math.isfinite(vz)
        and math.isfinite(vn)
    ):
        # Parallel (coplanar) arcs: check whether endpoints of one lie on the other.
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

    # 3. Two antipodal candidate intersection points; keep those on both arcs.
    inv = 1.0 / vn
    pos = np.empty(3)
    pos[0] = vx * inv
    pos[1] = vy * inv
    pos[2] = vz * inv
    neg = np.empty(3)
    neg[0] = -pos[0]
    neg[1] = -pos[1]
    neg[2] = -pos[2]

    if on_minor_arc(pos, w0, w1) and on_minor_arc(pos, v0, v1):
        res[count, 0] = pos[0]
        res[count, 1] = pos[1]
        res[count, 2] = pos[2]
        count += 1

    if on_minor_arc(neg, w0, w1) and on_minor_arc(neg, v0, v1):
        res[count, 0] = neg[0]
        res[count, 1] = neg[1]
        res[count, 2] = neg[2]
        count += 1

    return res[:count]


@njit(cache=True)
def gca_const_lat_intersection(gca_cart, const_z):
    """Find intersection point(s) of a great-circle arc and a constant-latitude line.

    Implements the ``accux_constlat`` algorithm from AccuSphGeom
    (gca_constlat_intersection.hpp) using compensated arithmetic throughout
    to achieve near-machine-precision accuracy even for arcs nearly tangent
    to the latitude circle.

    Computes the plane normal via ``accucross``, forms the discriminant using
    compensated sum-of-squares and ``acc_sqrt_re``, solves for the two candidate
    intersection points with compensated dot products, and retains only those
    that are finite and lie on the minor arc.

    Parameters
    ----------
    gca_cart : np.ndarray, shape (2, 3)
        Cartesian coordinates of the two endpoints of the great-circle arc.
    const_z : float
        The constant z-coordinate (= sin(latitude)) of the latitude line.

    Returns
    -------
    np.ndarray, shape (2, 3)
        Intersection point(s). Missing entries are NaN-filled rows. The first
        valid intersection is in row 0; a second (rare) intersection in row 1.
    """
    res = np.empty((2, 3))
    res.fill(np.nan)

    x1 = gca_cart[0]
    x2 = gca_cart[1]

    # 1. Plane normal via compensated cross product (keeps hi, lo residuals).
    nx_hi, ny_hi, nz_hi, nx_lo, ny_lo, nz_lo = accucross(
        x1[0], x1[1], x1[2], x2[0], x2[1], x2[2]
    )

    # 2. s2 = nx²+ny²  (compensated, on hi/lo pairs — matches sum_of_squares_c<2>).
    s2_hi, s2_lo = _sum_sq_c2(nx_hi, nx_lo, ny_hi, ny_lo)
    denom = s2_hi + s2_lo
    if denom == 0.0:
        return res

    # 3. s3 = |n|² = nx²+ny²+nz²  (compensated — matches sum_of_squares_c<3>).
    s3_hi, s3_lo = _sum_sq_c3(nx_hi, nx_lo, ny_hi, ny_lo, nz_hi, nz_lo)

    # 4. zsq = z₀² exactly (two_prod replaces two_prod_fma; same exact result).
    zsq_hi, zsq_lo = two_prod(const_z, const_z)

    # 5. d = s3 · zsq  via 4-term compensated dot product matching C++:
    #    compensated_dot_product({s3_hi, s3_hi, s3_lo, s3_lo},
    #                            {zsq_hi, zsq_lo, zsq_hi, zsq_lo})
    d_hi, d_lo = _cdp4(
        s3_hi,
        zsq_hi,
        s3_hi,
        zsq_lo,
        s3_lo,
        zsq_hi,
        s3_lo,
        zsq_lo,
    )
    # Note: Numba doesn't allow negative sign in function args, so negate d_hi explicitly.
    neg_d_hi = -d_hi

    # 6. planar_sq = s2 − d  (compensated two_sum on the high parts + low correction).
    e_hi, e_lo = two_sum(s2_hi, neg_d_hi)
    planar_sq = e_hi + (e_lo + s2_lo - d_lo)

    if planar_sq < 0.0:
        return res

    # 7. Accurate square root of discriminant.
    s_root, s_corr = acc_sqrt_re(planar_sq)

    # Collapse compensated values to scalars for the final formula.
    nx = nx_hi + nx_lo
    ny = ny_hi + ny_lo
    nz = nz_hi + nz_lo
    planar = s_root + s_corr

    # 8. Numerators via 2-term compensated dot products (matches C++ accux_constlat).
    #    x_pos = -(nx*nz*z₀  + (−ny)*planar) / denom
    #    y_pos = -(ny*nz*z₀  +   nx *planar) / denom
    #    x_neg = -(nx*nz*z₀  +   ny *planar) / denom
    #    y_neg = -(ny*nz*z₀  + (−nx)*planar) / denom
    xp_hi, xp_lo = _cdp2(nx * nz, const_z, -ny, planar)
    yp_hi, yp_lo = _cdp2(ny * nz, const_z, nx, planar)
    xn_hi, xn_lo = _cdp2(nx * nz, const_z, ny, planar)
    yn_hi, yn_lo = _cdp2(ny * nz, const_z, -nx, planar)

    inv_denom = 1.0 / denom
    p1 = np.empty(3)
    p1[0] = -(xp_hi + xp_lo) * inv_denom
    p1[1] = -(yp_hi + yp_lo) * inv_denom
    p1[2] = const_z

    p2 = np.empty(3)
    p2[0] = -(xn_hi + xn_lo) * inv_denom
    p2[1] = -(yn_hi + yn_lo) * inv_denom
    p2[2] = const_z

    # 9a. Snap computed (x, y) to any arc endpoint that lies exactly on the latitude.
    #     Adjacent edges sharing such an endpoint would otherwise return slightly
    #     different coordinates; snapping gives them the same exact value so that
    #     deduplication in the caller works correctly.  Matches Hongyu's suggestion
    #     of mask-selection to snap after computing rather than branching out early.
    _snap_sq = 1e-14  # distance² ≈ (1e-7)² — well above algorithm error (~1e-15)
    for xe in (x1, x2):
        if abs(xe[2] - const_z) <= ERROR_TOLERANCE:
            dx = p1[0] - xe[0]
            dy = p1[1] - xe[1]
            if dx * dx + dy * dy < _snap_sq:
                p1[0] = xe[0]
                p1[1] = xe[1]
            dx = p2[0] - xe[0]
            dy = p2[1] - xe[1]
            if dx * dx + dy * dy < _snap_sq:
                p2[0] = xe[0]
                p2[1] = xe[1]

    # 9b. Retain each candidate that is finite and lies on the minor arc.
    p1_ok = math.isfinite(p1[0]) and math.isfinite(p1[1]) and on_minor_arc(p1, x1, x2)
    p2_ok = math.isfinite(p2[0]) and math.isfinite(p2[1]) and on_minor_arc(p2, x1, x2)

    # When both candidates are valid but nearly identical (tangent/endpoint case),
    # treat as a single intersection — same as the C++ scalar gca_constlat_intersection
    # which returns only one point when status==0 (exactly one candidate lies on the arc).
    if p1_ok and p2_ok:
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        if dx * dx + dy * dy < _snap_sq:
            p2_ok = False

    if p1_ok and p2_ok:
        res[0, 0] = p1[0]
        res[0, 1] = p1[1]
        res[0, 2] = p1[2]
        res[1, 0] = p2[0]
        res[1, 1] = p2[1]
        res[1, 2] = p2[2]
    elif p1_ok:
        res[0, 0] = p1[0]
        res[0, 1] = p1[1]
        res[0, 2] = p1[2]
    elif p2_ok:
        res[0, 0] = p2[0]
        res[0, 1] = p2[1]
        res[0, 2] = p2[2]

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
