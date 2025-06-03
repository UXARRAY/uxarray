import numpy as np
from numba import njit, prange

from uxarray.constants import ERROR_TOLERANCE, INT_DTYPE, MACHINE_EPSILON
from uxarray.grid.arcs import (
    extreme_gca_z,
    in_between,
    point_within_gca,
)
from uxarray.grid.utils import (
    _angle_of_2_vectors,
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
    if gca_a_xyz.shape[1] != 3 or gca_b_xyz.shape[1] != 3:
        raise ValueError("The two GCAs must be in the cartesian [x, y, z] format")

    # Extract points
    w0_xyz = gca_a_xyz[0]
    w1_xyz = gca_a_xyz[1]
    v0_xyz = gca_b_xyz[0]
    v1_xyz = gca_b_xyz[1]

    angle_w0w1 = _angle_of_2_vectors(w0_xyz, w1_xyz)
    angle_v0v1 = _angle_of_2_vectors(v0_xyz, v1_xyz)

    if angle_w0w1 > np.pi:
        w0_xyz, w1_xyz = w1_xyz, w0_xyz

    if angle_v0v1 > np.pi:
        v0_xyz, v1_xyz = v1_xyz, v0_xyz

    w0w1_norm = np.cross(w0_xyz, w1_xyz)
    v0v1_norm = np.cross(v0_xyz, v1_xyz)
    cross_norms = np.cross(w0w1_norm, v0v1_norm)

    # Initialize result array and counter
    res = np.empty((2, 3))
    count = 0

    # Check if the two GCAs are parallel
    if np.allclose(cross_norms, 0.0, atol=MACHINE_EPSILON):
        if point_within_gca(v0_xyz, w0_xyz, w1_xyz):
            res[count, :] = v0_xyz
            count += 1

        if point_within_gca(v1_xyz, w0_xyz, w1_xyz):
            res[count, :] = v1_xyz
            count += 1

        return res[:count, :]

    # Normalize the cross_norms
    cross_norms = cross_norms / np.linalg.norm(cross_norms)
    x1_xyz = cross_norms
    x2_xyz = -x1_xyz

    # Check intersection points
    if point_within_gca(x1_xyz, w0_xyz, w1_xyz) and point_within_gca(
        x1_xyz, v0_xyz, v1_xyz
    ):
        res[count, :] = x1_xyz
        count += 1

    if point_within_gca(x2_xyz, w0_xyz, w1_xyz) and point_within_gca(
        x2_xyz, v0_xyz, v1_xyz
    ):
        res[count, :] = x2_xyz
        count += 1

    return res[:count, :]


@njit(cache=True)
def gca_const_lat_intersection(gca_cart, const_z):
    """Calculate the intersection point(s) of a Great Circle Arc (GCA) and a
    constant latitude line in a Cartesian coordinate system.

    Parameters
    ----------
    gca_cart : [2, 3] np.ndarray Cartesian coordinates of the two end points GCA.
    const_z : float
        The constant latitude represented in cartesian of the latitude line.

    Returns
    -------
    np.ndarray
        Cartesian coordinates of the intersection point(s) the shape is [2, 3]. If no intersections are found,
        all values a `nan`. If one intersection is found, the first column represent the intersection point, and
        if two intersections are found, each column represents a point.

    """
    res = np.empty((2, 3))
    res.fill(np.nan)

    x1, x2 = gca_cart

    # Check if the constant latitude has the same latitude as the GCA endpoints
    x1_at_const_z = np.isclose(
        x1[2], const_z, rtol=ERROR_TOLERANCE, atol=ERROR_TOLERANCE
    )
    x2_at_const_z = np.isclose(
        x2[2], const_z, rtol=ERROR_TOLERANCE, atol=ERROR_TOLERANCE
    )

    if x1_at_const_z and x2_at_const_z:
        res[0] = x1
        res[1] = x2
        return res
    elif x1_at_const_z:
        res[0] = x1
        return res
    elif x2_at_const_z:
        res[0] = x2
        return res

    # If the constant latitude is not the same as the GCA endpoints, calculate the intersection point
    z_min = extreme_gca_z(gca_cart, extreme_type="min")
    z_max = extreme_gca_z(gca_cart, extreme_type="max")

    # Check if the constant latitude is within the GCA range
    if not in_between(z_min, const_z, z_max):
        return res

    n = np.cross(x1, x2)

    nx, ny, nz = n

    s_tilde = np.sqrt(nx**2 + ny**2 - (nx**2 + ny**2 + nz**2) * const_z**2)
    p1_x = -(1.0 / (nx**2 + ny**2)) * (const_z * nx * nz + s_tilde * ny)
    p2_x = -(1.0 / (nx**2 + ny**2)) * (const_z * nx * nz - s_tilde * ny)
    p1_y = -(1.0 / (nx**2 + ny**2)) * (const_z * ny * nz - s_tilde * nx)
    p2_y = -(1.0 / (nx**2 + ny**2)) * (const_z * ny * nz + s_tilde * nx)

    p1 = np.array([p1_x, p1_y, const_z])
    p2 = np.array([p2_x, p2_y, const_z])

    p1_intersects_gca = point_within_gca(p1, gca_cart[0], gca_cart[1])
    p2_intersects_gca = point_within_gca(p2, gca_cart[0], gca_cart[1])

    if p1_intersects_gca and p2_intersects_gca:
        res[0] = p1
        res[1] = p2
    elif p1_intersects_gca:
        res[0] = p1
    elif p2_intersects_gca:
        res[0] = p2

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
