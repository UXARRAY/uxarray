import numpy as np
from uxarray.constants import MACHINE_EPSILON, ERROR_TOLERANCE, INT_DTYPE
from uxarray.grid.utils import _newton_raphson_solver_for_gca_constLat
from uxarray.grid.arcs import (
    point_within_gca,
    in_between,
    _extreme_gca_latitude_cartesian,
    _point_within_gca_cartesian,
)
from uxarray.grid.coordinates import _xyz_to_lonlat_rad_scalar
import platform
import warnings
from uxarray.utils.computing import cross_fma, allclose, cross, norm


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

    gca_a_lonlat = np.array(
        [
            _xyz_to_lonlat_rad_scalar(
                gca_a_xyz[0, 0], gca_a_xyz[0, 1], gca_a_xyz[0, 2]
            ),
            _xyz_to_lonlat_rad_scalar(
                gca_a_xyz[1, 0], gca_a_xyz[1, 1], gca_a_xyz[1, 2]
            ),
        ]
    )
    gca_b_lonlat = np.array(
        [
            _xyz_to_lonlat_rad_scalar(
                gca_b_xyz[0, 0], gca_b_xyz[0, 1], gca_b_xyz[0, 2]
            ),
            _xyz_to_lonlat_rad_scalar(
                gca_b_xyz[1, 0], gca_b_xyz[1, 1], gca_b_xyz[1, 2]
            ),
        ]
    )
    return gca_gca_intersection(gca_a_xyz, gca_a_lonlat, gca_b_xyz, gca_b_lonlat)


@njit(cache=True)
def gca_gca_intersection(gca_a_xyz, gca_a_lonlat, gca_b_xyz, gca_b_lonlat):
    if gca_a_xyz.shape[1] != 3 or gca_b_xyz.shape[1] != 3:
        raise ValueError("The two GCAs must be in the cartesian [x, y, z] format")

    # Extract points
    w0_xyz = gca_a_xyz[0]
    w1_xyz = gca_a_xyz[1]
    v0_xyz = gca_b_xyz[0]
    v1_xyz = gca_b_xyz[1]

    w0_lonlat = gca_a_lonlat[0]
    w1_lonlat = gca_a_lonlat[1]
    v0_lonlat = gca_b_lonlat[0]
    v1_lonlat = gca_b_lonlat[1]

    # Compute normals and orthogonal bases
    w0w1_norm = cross(w0_xyz, w1_xyz)
    v0v1_norm = cross(v0_xyz, v1_xyz)
    cross_norms = cross(w0w1_norm, v0v1_norm)

    # Initialize result array and counter
    res = np.empty((2, 3))
    count = 0

    # Check if the two GCAs are parallel
    if allclose(cross_norms, 0.0, atol=MACHINE_EPSILON):
        if point_within_gca(v0_xyz, v0_lonlat, w0_xyz, w0_lonlat, w1_xyz, w1_lonlat):
            res[count, :] = v0_xyz
            count += 1

        if point_within_gca(v1_xyz, v1_lonlat, w0_xyz, w0_lonlat, w1_xyz, w1_lonlat):
            res[count, :] = v1_xyz
            count += 1

        return res[:count, :]

    # Normalize the cross_norms
    cross_norms = cross_norms / norm(cross_norms)
    x1_xyz = cross_norms
    x2_xyz = -x1_xyz

    # Get lon/lat arrays
    x1_lonlat = _xyz_to_lonlat_rad_scalar(x1_xyz[0], x1_xyz[1], x1_xyz[2])
    x2_lonlat = _xyz_to_lonlat_rad_scalar(x2_xyz[0], x2_xyz[1], x2_xyz[2])

    # Check intersection points
    if point_within_gca(
        x1_xyz, x1_lonlat, w0_xyz, w0_lonlat, w1_xyz, w1_lonlat
    ) and point_within_gca(x1_xyz, x1_lonlat, v0_xyz, v0_lonlat, v1_xyz, v1_lonlat):
        res[count, :] = x1_xyz
        count += 1

    if point_within_gca(
        x2_xyz, x2_lonlat, w0_xyz, w0_lonlat, w1_xyz, w1_lonlat
    ) and point_within_gca(x2_xyz, x2_lonlat, v0_xyz, v0_lonlat, v1_xyz, v1_lonlat):
        res[count, :] = x2_xyz
        count += 1

    return res[:count, :]


def gca_const_lat_intersection(
    gca_cart, constZ, fma_disabled=True, verbose=False, is_directed=False
):
    """Calculate the intersection point(s) of a Great Circle Arc (GCA) and a
    constant latitude line in a Cartesian coordinate system.

    To reduce relative errors, the Fused Multiply-Add (FMA) operation is utilized.
    A warning is raised if the given coordinates are not in the cartesian coordinates, or
    they cannot be accurately handled using floating-point arithmetic.

    Parameters
    ----------
    gca_cart : [2, 3] np.ndarray Cartesian coordinates of the two end points GCA.
    constZ : float
        The constant latitude represented in cartesian of the latitude line.
    fma_disabled : bool, optional (default=True)
        If True, the FMA operation is disabled. And a naive `np.cross` is used instead.
    verbose : bool, optional (default=False)
        If True, the function prints out the intermediate results.
    is_directed : bool, optional (default=False)
        If True, the GCA is considered to be directed, which means it can only from v0-->v1. If False, the GCA is undirected,
        and we will always assume the small circle (The one less than 180 degree) side is the GCA.

    Returns
    -------
    np.ndarray
        Cartesian coordinates of the intersection point(s) the shape is [n_intersext_pts, 3].

    Raises
    ------
    ValueError
        If the input GCA is not in the cartesian [x, y, z] format.

    Warning
    -------
        If running on the Windows system with fma_disabled=False since the C/C++ implementation of FMA in MS Windows
        is fundamentally broken. (bug report: https://bugs.python.org/msg312480)

        If the intersection point cannot be converged using the Newton-Raphson method, the initial guess intersection
        point is used instead, proceed with caution.
    """
    x1, x2 = gca_cart

    # Check if the constant latitude has the same latitude as the GCA endpoints
    # We are using the relative tolerance and ERROR_TOLERANCE since the constZ is calculated from np.sin, which
    # may have some floating-point error.
    res = None
    if np.isclose(x1[2], constZ, rtol=ERROR_TOLERANCE, atol=ERROR_TOLERANCE):
        res = np.array([x1]) if res is None else np.vstack((res, x1))

    if np.isclose(x2[2], constZ, rtol=ERROR_TOLERANCE, atol=ERROR_TOLERANCE):
        res = np.array([x2]) if res is None else np.vstack((res, x2))

    if res is not None:
        return res

    # If the constant latitude is not the same as the GCA endpoints, calculate the intersection point
    # TODO:
    lat_min = _extreme_gca_latitude_cartesian(gca_cart, extreme_type="min")
    lat_max = _extreme_gca_latitude_cartesian(gca_cart, extreme_type="max")

    constLat_rad = np.arcsin(constZ)

    # Check if the constant latitude is within the GCA range
    # Because the constant latitude is calculated from np.sin, which may have some floating-point error,
    if not in_between(lat_min, constLat_rad, lat_max):
        pass
        return np.array([])

    if fma_disabled:
        n = cross(x1, x2)

    else:
        # Raise a warning for Windows users
        if platform.system() == "Windows":
            warnings.warn(
                "The C/C++ implementation of FMA in MS Windows is reportedly broken. Use with care. (bug report: "
                "https://bugs.python.org/msg312480)"
                "The single rounding cannot be guaranteed, hence the relative error bound of 3u cannot be guaranteed."
            )
        n = cross_fma(x1, x2)

    nx, ny, nz = n

    s_tilde = np.sqrt(nx**2 + ny**2 - (nx**2 + ny**2 + nz**2) * constZ**2)
    p1_x = -(1.0 / (nx**2 + ny**2)) * (constZ * nx * nz + s_tilde * ny)
    p2_x = -(1.0 / (nx**2 + ny**2)) * (constZ * nx * nz - s_tilde * ny)
    p1_y = -(1.0 / (nx**2 + ny**2)) * (constZ * ny * nz - s_tilde * nx)
    p2_y = -(1.0 / (nx**2 + ny**2)) * (constZ * ny * nz + s_tilde * nx)

    p1 = np.array([p1_x, p1_y, constZ])
    p2 = np.array([p2_x, p2_y, constZ])

    res = None

    # Now test which intersection point is within the GCA range
    if _point_within_gca_cartesian(p1, gca_cart, is_directed=is_directed):
        try:
            converged_pt = _newton_raphson_solver_for_gca_constLat(
                p1, gca_cart, verbose=verbose
            )

            if converged_pt is None:
                # The point is not be able to be converged using the jacobi method, raise a warning and continue with p2
                warnings.warn(
                    "The intersection point cannot be converged using the Newton-Raphson method. "
                    "The initial guess intersection point is used instead, procced with caution."
                )
                res = np.array([p1]) if res is None else np.vstack((res, p1))
            else:
                res = (
                    np.array([converged_pt])
                    if res is None
                    else np.vstack((res, converged_pt))
                )
        except RuntimeError:
            raise RuntimeError(f"Error encountered with initial guess: {p1}")

    if _point_within_gca_cartesian(p2, gca_cart, is_directed=is_directed):
        try:
            converged_pt = _newton_raphson_solver_for_gca_constLat(
                p2, gca_cart, verbose=verbose
            )
            if converged_pt is None:
                # The point is not be able to be converged using the jacobi method, raise a warning and continue with p2
                warnings.warn(
                    "The intersection point cannot be converged using the Newton-Raphson method. "
                    "The initial guess intersection point is used instead, procced with caution."
                )
                res = np.array([p2]) if res is None else np.vstack((res, p2))
            else:
                res = (
                    np.array([converged_pt])
                    if res is None
                    else np.vstack((res, converged_pt))
                )
        except RuntimeError:
            raise RuntimeError(f"Error encountered with initial guess: {p2}")

    return res if res is not None else np.array([])
