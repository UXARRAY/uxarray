import numpy as np
from numba import njit


@njit(cache=True)
def clip(a, a_min, a_max):
    return np.clip(a, a_min, a_max)


@njit(cache=True)
def arcsin(x):
    return np.arcsin(x)


@njit(cache=True)
def all(a):
    """Numba decorated implementation of ``np.all()``

    See Also
    --------
    numpy.all
    """

    return np.all(a)


@njit(cache=True)
def isclose(a, b, rtol=1e-05, atol=1e-08):
    """Numba decorated implementation of ``np.isclose()``

    See Also
    --------
    numpy.isclose
    """

    return np.isclose(a, b, rtol=rtol, atol=atol)


@njit(cache=True)
def allclose(a, b, rtol=1e-05, atol=1e-08):
    """Numba decorated implementation of ``np.allclose()``

    See Also
    --------
    numpy.allclose
    """
    return np.allclose(a, b, rtol=rtol, atol=atol)


@njit(cache=True)
def cross(a, b):
    """Numba decorated implementation of ``np.cross()``

    See Also
    --------
    numpy.cross
    """
    return np.cross(a, b)


@njit(cache=True)
def dot(a, b):
    """Numba decorated implementation of ``np.dot()``

    See Also
    --------
    numpy.dot
    """
    return np.dot(a, b)


@njit(cache=True)
def norm(x):
    """Numba decorated implementation of ``np.linalg.norm()``

    See Also
    --------
    numpy.linalg.norm
    """
    return np.linalg.norm(x)
