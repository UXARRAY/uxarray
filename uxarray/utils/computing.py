import numpy as np
import sys


def _fmms(a, b, c, d):
    """
    Calculate the difference of products using the FMA (fused multiply-add) operation: (a * b) - (c * d).

    This operation leverages the fused multiply-add operation when available on the system and rounds the result only once.
    The relative error of this operation is bounded by 1.5 ulps when no overflow and underflow occur.

    Parameters
    ----------
    a (float): The first value of the first product.
    b (float): The second value of the first product.
    c (float): The first value of the second product.
    d (float): The second value of the second product.

    Returns
    -------
    float: The difference of the two products.

    Example
    -------
    >>> _fmms(3.0,2.0,1.0,1.0)
    5.0

    Reference
    ---------
    Claude-Pierre Jeannerod, Nicolas Louvet, and Jean-Michel Muller, Further
    analysis of Kahan’s algorithm for the accurate computation of 2 x 2 determinants,
    Mathematics of Computation, vol. 82, no. 284, pp. 2245-2264, 2013.
    [Read more](https://ens-lyon.hal.science/ensl-00649347) (DOI: 10.1090/S0025-5718-2013-02679-8)
    """
    import pyfma
    cd = c * d
    err = pyfma.fma(-c, d, cd)
    dop = pyfma.fma(a, b, -cd)
    return dop + err


def cross_fma(v1, v2):
    """Calculate the cross product of two 3D vectors utilizing the fused
    multiply-add operation.

    Parameters
    ----------
    v1 (np.array): The first vector of size 3.
    v2 (np.array): The second vector of size 3.

    Returns
    -------
    np.array: The cross product vector of size 3.

    Example
    -------
    >>> v1 = np.array([1.0, 2.0, 3.0])
    >>> v2 = np.array([4.0, 5.0, 6.0])
    >>> cross_fma(v1, v2)
    array([-3.0, 6.0, -3.0])
    """
    x = _fmms(v1[1], v2[2], v1[2], v2[1])
    y = _fmms(v1[2], v2[0], v1[0], v2[2])
    z = _fmms(v1[0], v2[1], v1[1], v2[0])
    return np.array([x, y, z])


def dot_fma(v1, v2):
    """Calculate the dot product of two vectors using the FMA (fused multiply-
    add) operation.

    This implementation leverages the FMA operation to provide a more accurate result. Currently the ComptDot product
    algorithm is used, which provides a relative error of approvimately u + n^2u^2cond(v1 dot v2), where u is 0.5 ulps,
    n is the length of the vectors, and cond(v1 dot v2) is the condition number of the naive dot product of v1 and v2.
    This operatin takes approvimately 3 + 10 * n flops, where n is the length of the vectors.

    Parameters
    ----------
    v1 : list of float
        The first vector.
    v2 : list of float
        The second vector. Must be the same length as v1.

    Returns
    -------
    float
        The dot product of the two vectors.

    Raises
    ------
    ValueError
        If the input vectors `v1` and `v2` are not of the same length.

    Examples
    --------
    >>> dot_fma([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    32.0

    References
    ----------
    S. Graillat, Ph. Langlois, and N. Louvet. "Accurate dot products with FMA." Presented at RNC 7, 2007, Nancy, France. DALI-LP2A Laboratory, University of Perpignan, France.
    [Poster](https://www-pequan.lip6.fr/~graillat/papers/posterRNC7.pdf)
    """
    if len(v1) != len(v2):
        raise ValueError("Input vectors must be of the same length")

    s, c = _two_prod_fma(v1[0], v2[0])
    for i in range(1, len(v1)):
        p, pi = _two_prod_fma(v1[i], v2[i])
        s, signma = _two_sum(s, p)
        c = c + pi + signma

    return s + c


def _two_prod_fma(a, b):
    """
    Error-free transformation of the product of two floating-point numbers using FMA, such that a * b = x + y exactly.

    Parameters
    ----------
    a, b : float
        The floating-point numbers to be multiplied.

    Returns
    -------
    tuple of float
        The product and the error term.

    Examples
    --------
    >>> _two_prod_fma(1.0, 2.0)
    (2.0, 0.0)
    """
    import pyfma
    x = a * b
    y = pyfma.fma(a, b, -x)
    return x, y


def _err_fmac(a, b, c):
    """
    Error-free transformation for the FMA operation. such that x = FMA(a,b,c) and a * b + c = x + y + z exactly.
    Thhis function is only available in round to the nearest mode and takes approximately 17 flops

    Parameters
    ----------
    a, b, c : float
        The operands for the FMA operation.

    Returns
    -------
    tuple of float
        The result of the FMA operation and two error terms.

    References
    ----------
    Graillat, Stef & Langlois, Philippe & Louvet, Nicolas. (2006). Improving the compensated Horner scheme with
    a Fused Multiply and Add. 2. 1323-1327. 10.1145/1141277.1141585.
    """
    if sys.float_info.rounds == 1:
        import pyfma
        x = pyfma.fma(a, b, c)
        u1, u2 = _fast_two_mult(a, b)
        alpha1, alpha2 = _two_sum(c, u2)
        beta1, beta2 = _two_sum(u1, alpha1)
        gamma = (beta1 - x) + beta2
        y, z = _fast_two_sum(gamma, alpha2)
        return x, y, z
    else:
        raise ValueError(
            "3FMA operation is only available in round to the nearest mode. and the current mode is "
            + str(sys.float_info.rounds))


def _two_sum(a, b):
    """
    Error-free transformation of the sum of two floating-point numbers such that a + b = x + y exactly

    Parameters
    ----------
    a, b : float
        The floating-point numbers to be added.

    Returns
    -------
    tuple of float
        The sum and the error term.

    Examples
    --------
    >>> _two_sum(1.0, 2.0)
    (3.0, 0.0)
    """
    x = a + b
    z = x - a
    y = (a - (x - z)) + (b - z)
    return x, y


def _fast_two_mult(a, b):
    """
    Error-free transformation of the product of two floating-point numbers such that a * b = x + y exactly.

    This function is faster than the _two_prod_fma function.
    """
    x = a * b
    y = a * b - x
    return x, y


def _fast_two_sum(a, b):
    """Compute a fast error-free transformation of the sum of two floating-
    point numbers.

    This function is a faster alternative to `_two_sum` for computing the sum
    of two floating-point numbers `a` and `b`, such that a + b = x + y exactly.
    Note: |a| must be no less than |b|.

    Parameters
    ----------
    a, b : float
        The floating-point numbers to be added. It is required that |a| >= |b|.

    Returns
    -------
    tuple of float
        The rounded sum of `a` and `b`, and the error term. The error term represents the difference between the exact sum and the rounded sum.

    Raises
    ------
    ValueError
        If |a| < |b|.

    Examples
    --------
    >>> _fast_two_sum(2.0, 1.0)
    (3.0, 0.0)

    >>> _fast_two_sum(1.0, 2.0)
    Traceback (most recent call last):
        ...
    ValueError: |a| must be greater than or equal to |b|.
    """
    if abs(a) >= abs(b):
        x = a + b
        b_tile = x - a
        y = b - b_tile
        return x, y

    else:
        raise ValueError("|a| must be greater than or equal to |b|.")
