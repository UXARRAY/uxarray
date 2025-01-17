import numpy as np
from uxarray.constants import MACHINE_EPSILON, ERROR_TOLERANCE, INT_DTYPE
from uxarray.grid.utils import (
    _angle_of_2_vectors,
)
from uxarray.grid.arcs import (
    in_between,
    extreme_gca_z,
    point_within_gca,
)
from uxarray.utils.computing import allclose, cross, norm, isclose


from numba import njit, prange


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
def constant_lat_intersections_face_bounds(lat, face_bounds_lat):
    """Identifies the candidate faces on a grid that intersect with a given
    constant latitude.

    This function checks whether the specified latitude, `lat`, in degrees lies within
    the latitude bounds of grid faces, defined by `face_min_lat_rad` and `face_max_lat_rad`,
    which are given in radians. The function returns the indices of the faces where the
    latitude is within these bounds.

    Parameters
    ----------
    lat : float
        The latitude in degrees for which to find intersecting faces.
    TODO:

    Returns
    -------
    candidate_faces : numpy.ndarray
        A 1D array containing the indices of the faces that intersect with the given latitude.
    """

    face_bounds_lat_min = face_bounds_lat[:, 0]
    face_bounds_lat_max = face_bounds_lat[:, 1]

    within_bounds = (face_bounds_lat_min <= lat) & (face_bounds_lat_max >= lat)
    candidate_faces = np.where(within_bounds)[0]
    return candidate_faces


@njit(cache=True)
def constant_lon_intersections_face_bounds(lon, face_bounds_lon):
    """Identifies the candidate faces on a grid that intersect with a given
    constant longitude.

    This function checks whether the specified longitude, `lon`, in degrees lies within
    the longitude bounds of grid faces, defined by `face_min_lon_rad` and `face_max_lon_rad`,
    which are given in radians. The function returns the indices of the faces where the
    longitude is within these bounds.

    Parameters
    ----------
    lon : float
        The longitude in degrees for which to find intersecting faces.
    TODO:

    Returns
    -------
    candidate_faces : numpy.ndarray
        A 1D array containing the indices of the faces that intersect with the given longitude.
    """

    face_bounds_lon_min = face_bounds_lon[:, 0]
    face_bounds_lon_max = face_bounds_lon[:, 1]
    n_face = face_bounds_lon.shape[0]

    candidate_faces = []
    for i in range(n_face):
        cur_face_bounds_lon_min = face_bounds_lon_min[i]
        cur_face_bounds_lon_max = face_bounds_lon_max[i]

        if cur_face_bounds_lon_min < cur_face_bounds_lon_max:
            if (lon >= cur_face_bounds_lon_min) & (lon <= cur_face_bounds_lon_max):
                candidate_faces.append(i)
        else:
            # antimeridian case
            if (lon >= cur_face_bounds_lon_min) | (lon <= cur_face_bounds_lon_max):
                candidate_faces.append(i)

    return np.array(candidate_faces, dtype=INT_DTYPE)


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

    w0w1_norm = cross(w0_xyz, w1_xyz)
    v0v1_norm = cross(v0_xyz, v1_xyz)
    cross_norms = cross(w0w1_norm, v0v1_norm)

    # Initialize result array and counter
    res = np.empty((2, 3))
    count = 0

    # Check if the two GCAs are parallel
    if allclose(cross_norms, 0.0, atol=MACHINE_EPSILON):
        if point_within_gca(v0_xyz, w0_xyz, w1_xyz):
            res[count, :] = v0_xyz
            count += 1

        if point_within_gca(v1_xyz, w0_xyz, w1_xyz):
            res[count, :] = v1_xyz
            count += 1

        return res[:count, :]

    # Normalize the cross_norms
    cross_norms = cross_norms / norm(cross_norms)
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
    res = np.empty((2, 3))
    res.fill(np.nan)

    x1, x2 = gca_cart

    # Check if the constant latitude has the same latitude as the GCA endpoints
    # We are using the relative tolerance and ERROR_TOLERANCE since the constZ is calculated from np.sin, which
    # may have some floating-point error.
    x1_at_const_z = isclose(x1[2], const_z, rtol=ERROR_TOLERANCE, atol=ERROR_TOLERANCE)
    x2_at_const_z = isclose(x2[2], const_z, rtol=ERROR_TOLERANCE, atol=ERROR_TOLERANCE)

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
    lat_min = np.arcsin(z_min)
    lat_max = np.arcsin(z_max)

    const_lat_rad = np.arcsin(const_z)

    # TODO:
    # Check if the constant latitude is within the GCA range
    # Because the constant latitude is calculated from np.sin, which may have some floating-point error,
    if not in_between(lat_min, const_lat_rad, lat_max):
        return res

    n = cross(x1, x2)

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
    row1_is_nan = np.all(np.isnan(arr[0]))
    row2_is_nan = np.all(np.isnan(arr[1]))

    if row1_is_nan and row2_is_nan:
        return 0
    elif row2_is_nan:
        return 1
    else:
        return 2
