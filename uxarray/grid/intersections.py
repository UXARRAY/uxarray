import numpy as np
from uxarray.constants import ERROR_TOLERANCE
from uxarray.grid.utils import cross_fma
from uxarray.grid.lines import point_within_gca
import platform
import warnings


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
        raise ValueError(
            "The two GCAs must be in the cartesian [x, y, z] format")

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
    if not np.allclose(np.dot(w0w1_norm, w0), 0,
                       atol=ERROR_TOLERANCE) or not np.allclose(
                           np.dot(w0w1_norm, w1), 0, atol=ERROR_TOLERANCE):
        warnings.warn(
            "The current input data cannot be computed accurately using floating-point arithmetic. Use with care "
        )

    if not np.allclose(np.dot(v0v1_norm, v0), 0,
                       atol=ERROR_TOLERANCE) or not np.allclose(
                           np.dot(v0v1_norm, v1), 0, atol=ERROR_TOLERANCE):
        warnings.warn(
            "The current input data cannot be computed accurately using floating-point arithmetic.  Use with care "
        )

    if not np.allclose(
            np.dot(cross_norms,
                   v0v1_norm), 0, atol=ERROR_TOLERANCE) or not np.allclose(
                       np.dot(cross_norms, w0w1_norm), 0, atol=ERROR_TOLERANCE):
        warnings.warn(
            "The current input data cannot be computed accurately using floating-point arithmetic. Use with care "
        )

    # If the cross_norms is zero, the two GCAs are parallel
    if np.allclose(cross_norms, 0, atol=ERROR_TOLERANCE):
        return np.array([])

    x1 = cross_norms
    x2 = -x1

    res = np.array([])

    # Determine which intersection point is within the GCAs range
    if point_within_gca(x1, [w0, w1]) and point_within_gca(x1, [v0, v1]):
        res = np.append(res, x1)

    elif point_within_gca(x2, [w0, w1]) and point_within_gca(x2, [v0, v1]):
        res = np.append(res, x2)

    return res
