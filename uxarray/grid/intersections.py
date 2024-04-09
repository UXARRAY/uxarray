import numpy as np
from uxarray.constants import ERROR_TOLERANCE
from uxarray.grid.utils import _newton_raphson_solver_for_gca_constLat
from uxarray.grid.arcs import point_within_gca
import platform
import warnings
from uxarray.utils.computing import cross_fma


def gca_gca_intersection(gca1_cart, gca2_cart, fma_disabled=False):
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
    fma_disabled : bool, optional (default=False)
        If True, the FMA operation is disabled. And a naive `np.cross` is used instead.

    Returns
    -------
    np.ndarray
        Cartesian coordinates of the intersection point(s).

    Raises
    ------
    ValueError
        If the input GCAs are not in the cartesian [x, y, z] format.



    Warning
    -------
        If the current input data cannot be computed accurately using floating-point arithmetic. Use with care

        If running on the Windows system with fma_disabled=False since the C/C++ implementation of FMA in MS Windows
        is fundamentally broken. (bug report: https://bugs.python.org/msg312480)
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
        w0w1_norm = np.cross(w0, w1)
        v0v1_norm = np.cross(v0, v1)
        cross_norms = np.cross(w0w1_norm, v0v1_norm)
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
    if not np.allclose(
        np.dot(w0w1_norm, w0), 0, atol=ERROR_TOLERANCE
    ) or not np.allclose(np.dot(w0w1_norm, w1), 0, atol=ERROR_TOLERANCE):
        warnings.warn(
            "The current input data cannot be computed accurately using floating-point arithmetic. Use with caution."
        )

    if not np.allclose(
        np.dot(v0v1_norm, v0), 0, atol=ERROR_TOLERANCE
    ) or not np.allclose(np.dot(v0v1_norm, v1), 0, atol=ERROR_TOLERANCE):
        warnings.warn(
            "The current input data cannot be computed accurately using floating-point arithmetic.  Use with caution. "
        )

    if not np.allclose(
        np.dot(cross_norms, v0v1_norm), 0, atol=ERROR_TOLERANCE
    ) or not np.allclose(np.dot(cross_norms, w0w1_norm), 0, atol=ERROR_TOLERANCE):
        warnings.warn(
            "The current input data cannot be computed accurately using floating-point arithmetic. Use with caution. "
        )

    # If the cross_norms is zero, the two GCAs are parallel
    if np.allclose(cross_norms, 0, atol=ERROR_TOLERANCE):
        return np.array([])

    # Normalize the cross_norms
    cross_norms = cross_norms / np.linalg.norm(cross_norms)
    x1 = cross_norms
    x2 = -x1

    res = np.array([])

    # Determine which intersection point is within the GCAs range
    if point_within_gca(x1, [w0, w1]) and point_within_gca(x1, [v0, v1]):
        res = np.append(res, x1)

    elif point_within_gca(x2, [w0, w1]) and point_within_gca(x2, [v0, v1]):
        res = np.append(res, x2)

    return res


def gca_constLat_intersection(
    gca_cart, constZ, fma_disabled=False, verbose=False, is_directed=False
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
    fma_disabled : bool, optional (default=False)
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
    """
    x1, x2 = gca_cart

    if fma_disabled:
        n = np.cross(x1, x2)

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

    s_tilde = np.sqrt(nx**2 + ny**2 - np.linalg.norm(n) ** 2 * constZ**2)
    p1_x = -(1.0 / (nx**2 + ny**2)) * (constZ * nx * nz + s_tilde * ny)
    p2_x = -(1.0 / (nx**2 + ny**2)) * (constZ * nx * nz - s_tilde * ny)
    p1_y = -(1.0 / (nx**2 + ny**2)) * (constZ * ny * nz - s_tilde * nx)
    p2_y = -(1.0 / (nx**2 + ny**2)) * (constZ * ny * nz + s_tilde * nx)

    p1 = np.array([p1_x, p1_y, constZ])
    p2 = np.array([p2_x, p2_y, constZ])

    res = None

    # Now test which intersection point is within the GCA range
    if point_within_gca(p1, gca_cart, is_directed=is_directed):
        converged_pt = _newton_raphson_solver_for_gca_constLat(
            p1, gca_cart, verbose=verbose
        )
        res = (
            np.array([converged_pt]) if res is None else np.vstack((res, converged_pt))
        )

    if point_within_gca(p2, gca_cart, is_directed=is_directed):
        converged_pt = _newton_raphson_solver_for_gca_constLat(
            p2, gca_cart, verbose=verbose
        )
        res = (
            np.array([converged_pt]) if res is None else np.vstack((res, converged_pt))
        )

    return res if res is not None else np.array([])
