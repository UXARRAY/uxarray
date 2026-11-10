"""
Purpose: numba math helpers/primitives

Creating lots of tiny numpy arrays (or lists) costs a lot in numba;
    it's much more efficient to use individual components
    or write tuples, when looping across points.
    E.g., instead of v1=np.array((x1,y1,z1)); v2=np.array((x2,y2,z2)); np.dot(v1,v2),
        using v1=(x1,y1,z1); v2=(x2,y2,z2); _numba_dot3(v1,v2) is much faster,
        because it doesn't need to allocate lists/arrays.
    See issue #1648 for more details.
"""

import math

import numpy as np
from numba import njit

from uxarray.utils.computing import diff_of_products

# ------- basic arithmetic with vectors ------- #


@njit(cache=True)
def _numba_add3(u, v):
    """component-wise addition of 3-vectors; returns (u[0] + v[0], u[1] + v[1], u[2] + v[2])"""
    return u[0] + v[0], u[1] + v[1], u[2] + v[2]


@njit(cache=True)
def _numba_add3_scalar(u, scalar):
    """component-wise addition of 3-vector and scalar; returns (u[0] + scalar, u[1] + scalar, u[2] + scalar)"""
    return u[0] + scalar, u[1] + scalar, u[2] + scalar


@njit(cache=True)
def _numba_sub3(u, v):
    """component-wise subtraction of 3-vectors; returns (u[0] - v[0], u[1] - v[1], u[2] - v[2])"""
    return u[0] - v[0], u[1] - v[1], u[2] - v[2]


# _numba_sub3_scalar not provided; just use _numba_add3_scalar with negative scalar


@njit(cache=True)
def _numba_neg3(u):
    """component-wise negation of 3-vector; returns (-u[0], -u[1], -u[2])"""
    return -u[0], -u[1], -u[2]


@njit(cache=True)
def _numba_mul3(u, v):
    """component-wise multiplication of 3-vectors; returns (u[0] * v[0], u[1] * v[1], u[2] * v[2])"""
    return u[0] * v[0], u[1] * v[1], u[2] * v[2]


@njit(cache=True)
def _numba_mul3_scalar(u, scalar):
    """component-wise multiplication of 3-vector and scalar; returns (u[0] * scalar, u[1] * scalar, u[2] * scalar)"""
    return u[0] * scalar, u[1] * scalar, u[2] * scalar


@njit(cache=True)
def _numba_div3(u, v):
    """component-wise division of 3-vectors; returns (u[0] / v[0], u[1] / v[1], u[2] / v[2])"""
    return u[0] / v[0], u[1] / v[1], u[2] / v[2]


@njit(cache=True)
def _numba_div3_scalar(u, scalar):
    """component-wise division of 3-vector by scalar; returns (u[0] / scalar, u[1] / scalar, u[2] / scalar)"""
    return u[0] / scalar, u[1] / scalar, u[2] / scalar


@njit(cache=True)
def _numba_sqrt3(u):
    """component-wise square root of 3-vector; returns (sqrt(u[0]), sqrt(u[1]), sqrt(u[2]))"""
    return np.sqrt(u[0]), np.sqrt(u[1]), np.sqrt(u[2])


# ------- vector arithmetic ------- #


@njit(cache=True)
def _numba_norm3(u):
    """Euclidean norm of a 3-vector; returns sqrt(u[0]**2 + u[1]**2 + u[2]**2)"""
    return (u[0] ** 2 + u[1] ** 2 + u[2] ** 2) ** 0.5
    # for some reason, using **0.5 instead of np.sqrt makes numba more likely to
    # properly respect float64 inputs instead of using float32 precision.


@njit(cache=True)
def _numba_dot3(u, v):
    """dot product of two 3-vectors; returns u[0]*v[0] + u[1]*v[1] + u[2]*v[2]"""
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


@njit(cache=True)
def _numba_cross3(u, v):
    """cross product of two 3-vectors.

    Parameters
    ----------
    u : iterable of length 3
        The first input vector.
    v : iterable of length 3
        The second input vector.

    Examples
    --------
    >>> _numba_cross3((1, 2, 30), (4, 0, 6))
    (12, 114, -8)  # (2*6 - 30*0, 30*4 - 1*6, 1*0 - 2*4)
    """
    cx = u[1] * v[2] - u[2] * v[1]
    cy = u[2] * v[0] - u[0] * v[2]
    cz = u[0] * v[1] - u[1] * v[0]
    return (cx, cy, cz)


@njit(cache=True, inline="always")
def _numba_diff_of_products3(u, a, v, b):
    """component-wise u * a - v * b for 3-vectors u, v and scalars a, b, without cancellation.

    Each component is evaluated with Kahan's 2x2 determinant
    (:func:`uxarray.utils.computing.diff_of_products`), so it is accurate even
    when ``u * a`` and ``v * b`` nearly cancel.

    Returns
    -------
    hi, lo : tuple of 3 floats
        High and low parts of each component; ``hi[i] + lo[i]`` is the
        accurate value. ``hi`` alone equals the naive ``u * a - v * b``, so the
        two must be added, e.g. with ``_numba_add3(hi, lo)``.
    """
    h0, l0 = diff_of_products(u[0], a, v[0], b)
    h1, l1 = diff_of_products(u[1], a, v[1], b)
    h2, l2 = diff_of_products(u[2], a, v[2], b)
    return (h0, h1, h2), (l0, l1, l2)


# ------- convenience functions / helpers ------- #


@njit(cache=True)
def _numba_allfinite3(u):
    """Return (as 1 or 0) whether all components of a 3-vector are finite (not NaN or Inf)."""
    return (
        int(math.isfinite(u[0])) * int(math.isfinite(u[1])) * int(math.isfinite(u[2]))
    )
    # (use `*` instead of `and` to avoid branching logic)


@njit(cache=True)
def _numba_allclose3(u, v, rtol=1e-05, atol=1e-08):
    """Return (as 1 or 0) whether all components of two 3-vectors are close to each other."""
    return (
        int(np.isclose(u[0], v[0], rtol=rtol, atol=atol))
        * int(np.isclose(u[1], v[1], rtol=rtol, atol=atol))
        * int(np.isclose(u[2], v[2], rtol=rtol, atol=atol))
    )
