import numpy as np
from uxarray.constants import MACHINE_EPSILON, ERROR_TOLERANCE
from uxarray.grid.utils import _newton_raphson_solver_for_gca_constLat
from uxarray.grid.arcs import point_within_gca, extreme_gca_latitude, in_between
import platform
import warnings
from uxarray.utils.computing import cross_fma, allclose, dot, cross, norm


def gca_gca_intersection(gca1_cart, gca2_cart, fma_disabled=True):
    """Calculate the intersection point(s) of two Great Circle Arcs (GCAs) in a
    Cartesian coordinate system.

    To reduce relative errors, the Fused Multiply-Add (FMA) operation is utilized.
    A warning is raised if the given coordinates are not in the cartesian coordinates, or
    they cannot be accurately handled using floating-point arithmetic.

    Parameters
    ----------
    gca1_cart : [n, 3] np.ndarray where n is the number of intersection points
        Cartesian coordinates of the first GCA.
    gca2_cart : [n, 3] np.ndarray where n is the number of intersection points
        Cartesian coordinates of the second GCA.
    fma_disabled : bool, optional (default=True)
        If True, the FMA operation is disabled. And a naive `np.cross` is used instead.

    Returns
    -------
    np.ndarray
        Cartesian coordinates of the intersection point(s). Returns an empty array if no intersections,
        a 2D array with one row if one intersection, and a 2D array with two rows if two intersections.

    Raises
    ------
    ValueError
        If the input GCAs are not in the cartesian [x, y, z] format.
    """

    # Support lists as an input
    gca1_cart = np.asarray(gca1_cart)
    gca2_cart = np.asarray(gca2_cart)
    # Check if the two GCAs are in the cartesian format (size of three)
    if gca1_cart.shape[1] != 3 or gca2_cart.shape[1] != 3:
        raise ValueError("The two GCAs must be in the cartesian [x, y, z] format")

    w0, w1 = gca1_cart
    v0, v1 = gca2_cart

    # Compute normals and orthogonal bases using FMA
    if fma_disabled:
        w0w1_norm = cross(w0, w1)
        v0v1_norm = cross(v0, v1)
        cross_norms = cross(w0w1_norm, v0v1_norm)
    else:
        w0w1_norm = cross_fma(w0, w1)
        v0v1_norm = cross_fma(v0, v1)
        cross_norms = cross_fma(w0w1_norm, v0v1_norm)

        # Raise a warning for windows users
        if platform.system() == "Windows":
            warnings.warn(
                "The C/C++ implementation of FMA in MS Windows is reportedly broken. Use with care. (bug report: "
                "https://bugs.python.org/msg312480)"
                "The single rounding cannot be guaranteed, hence the relative error bound of 3u cannot be guaranteed."
            )

    # Check perpendicularity conditions and floating-point arithmetic limitations
    if not allclose(dot(w0w1_norm, w0), 0.0, atol=MACHINE_EPSILON) or not allclose(
        dot(w0w1_norm, w1), 0.0, atol=MACHINE_EPSILON
    ):
        warnings.warn(
            "The current input data cannot be computed accurately using floating-point arithmetic. Use with caution."
        )

    if not allclose(dot(v0v1_norm, v0), 0.0, 0.0, atol=MACHINE_EPSILON) or not allclose(
        dot(v0v1_norm, v1), 0.0, atol=MACHINE_EPSILON
    ):
        warnings.warn(
            "The current input data cannot be computed accurately using floating-point arithmetic.  Use with caution. "
        )

    if not allclose(
        dot(cross_norms, v0v1_norm), 0.0, atol=MACHINE_EPSILON
    ) or not allclose(dot(cross_norms, w0w1_norm), 0.0, atol=MACHINE_EPSILON):
        warnings.warn(
            "The current input data cannot be computed accurately using floating-point arithmetic. Use with caution. "
        )

    # If the cross_norms is zero, the two GCAs are parallel
    if allclose(cross_norms, 0.0, atol=MACHINE_EPSILON):
        res = []
        # Check if the two GCAs are overlapping
        if point_within_gca(v0, [w0, w1]):
            # The two GCAs are overlapping, return both end points
            res.append(v0)

        if point_within_gca(v1, [w0, w1]):
            res.append(v1)
        return np.array(res)

    # Normalize the cross_norms
    cross_norms = cross_norms / norm(cross_norms)
    x1 = cross_norms
    x2 = -x1

    res = []

    # Determine which intersection point is within the GCAs range
    if point_within_gca(x1, [w0, w1]) and point_within_gca(x1, [v0, v1]):
        res.append(x1)

    if point_within_gca(x2, [w0, w1]) and point_within_gca(x2, [v0, v1]):
        res.append(x2)

    return np.array(res)


def gca_constLat_intersection(
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
    lat_min = extreme_gca_latitude(gca_cart, extreme_type="min")
    lat_max = extreme_gca_latitude(gca_cart, extreme_type="max")

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
    if point_within_gca(p1, gca_cart, is_directed=is_directed):
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

    if point_within_gca(p2, gca_cart, is_directed=is_directed):
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
