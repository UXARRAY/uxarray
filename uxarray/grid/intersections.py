import numpy as np
from uxarray.constants import INT_DTYPE, INT_FILL_VALUE, ERROR_TOLERANCE
from uxarray.grid.utils import gram_schmidt
from uxarray.grid.coordinates import node_xyz_to_lonlat_rad
from uxarray.grid.lines import point_within_GCA


def get_GCA_GCA_intersection(gca1_cart, gca2_cart):
    """Get the intersection point(s) of two Great Circle Arcs.

    Parameters
    ----------
    gca1_cart : np.ndarray
        Cartesian coordinates of the first GCA
    gca2_cart : np.ndarray
        Cartesian coordinates of the second GCA

    Returns
    -------
    np.ndarray
        Cartesian coordinates of the intersection point(s)
    """
    # Check if the two GCRs are in the cartesian format (size of three)
    if gca1_cart.shape[1] != 3 or gca2_cart.shape[1] != 3:
        raise ValueError(
            "The two GCRs must be in the cartesian[x, y, z] format")

    w0, w1 = gca1_cart
    v0, v1 = gca2_cart

    # Compute normals and orthogonal bases
    w0w1_norm = np.cross(w0, w1)
    v0v1_norm = np.cross(v0, v1)

    w0w1norm_orthogonal = gram_schmidt([w0w1_norm.copy(),
                                        w0.copy(),
                                        w1.copy()])[0]
    v0v1norm_orthogonal = gram_schmidt([v0v1_norm.copy(),
                                        v0.copy(),
                                        v1.copy()])[0]
    cross_norms = np.cross(w0w1_norm, v0v1_norm)
    cross_norms_orthogonal = gram_schmidt(
        [cross_norms.copy(),
         w0w1_norm.copy(),
         v0v1_norm.copy()])[0]

    # Check perpendicularity conditions
    if not np.allclose(
            np.dot(w0w1norm_orthogonal, w0), 0,
            atol=ERROR_TOLERANCE) or not np.allclose(
                np.dot(w0w1norm_orthogonal, w1), 0, atol=ERROR_TOLERANCE):
        raise ValueError(
            "The current input data cannot be computed using the floating point arithmetic. Please "
            "turn on the multi-precision mode and re-run.")

    if not np.allclose(
            np.dot(v0v1norm_orthogonal, v0), 0,
            atol=ERROR_TOLERANCE) or not np.allclose(
                np.dot(v0v1norm_orthogonal, v1), 0, atol=ERROR_TOLERANCE):
        raise ValueError(
            "The current input data cannot be computed using the floating point arithmetic. Please "
            "turn on the multi-precision mode and re-run.")

    if not np.allclose(np.dot(cross_norms_orthogonal, v0v1norm_orthogonal),
                       0,
                       atol=ERROR_TOLERANCE) or not np.allclose(
                           np.dot(cross_norms_orthogonal, w0w1norm_orthogonal),
                           0,
                           atol=ERROR_TOLERANCE):
        raise ValueError(
            "The current input data cannot be computed using the floating point arithmetic. Please "
            "turn on the multi-precision mode and re-run.")

    # Compute intersection points
    if np.allclose(cross_norms, 0, atol=ERROR_TOLERANCE):
        return np.array([0, 0, 0])

    x1 = cross_norms
    x2 = -x1

    if point_within_GCA(x1, [w0, w1]) and point_within_GCA(x1, [v0, v1]):
        return x1
    elif point_within_GCA(x2, [w0, w1]) and point_within_GCA(x2, [v0, v1]):
        return x2
    elif np.all(x1 == 0):
        return np.array([0, 0, 0])  # two vectors are parallel to each other
    else:
        return np.array([-1, -1, -1])  # Intersection out of the interval
