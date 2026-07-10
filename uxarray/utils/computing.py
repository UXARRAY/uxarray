import sys

import numpy as np


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
    >>> _fmms(3.0, 2.0, 1.0, 1.0)
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
    S. Graillat, Ph. Langlois, and N. Louvet. "Accurate dot products with FMA." Presented at RNC 7, 2007, Nancy, France.
    DALI-LP2A Laboratory, University of Perpignan, France.
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
    """Error-free transformation of the product of two floating-point numbers
    using FMA, such that a * b = x + y exactly.

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

    Reference
    ---------
    Stef Graillat. Accurate Floating Point Product and Exponentiation.
    IEEE Transactions on Computers, 58(7), 994–1000, 2009.10.1109/TC.2008.215.
    """
    import pyfma

    x = a * b
    y = pyfma.fma(a, b, -x)
    return x, y


def _err_fmac(a, b, c):
    """Error-free transformation for the FMA operation. such that x =
    FMA(a,b,c) and a * b + c = x + y + z exactly. Thhis function is only
    available in round to the nearest mode and takes approximately 17 flops.

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

    Ogita, Takeshi & Rump, Siegfried & Oishi, Shin’ichi. (2005). Accurate Sum and Dot Product.
    SIAM J. Scientific Computing. 26. 1955-1988. 10.1137/030601818.
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
            + str(sys.float_info.rounds)
        )


def _two_sum(a, b):
    """Error-free transformation of the sum of two floating-point numbers such
    that a + b = x + y exactly.

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

    Reference
    ---------
    D. Knuth. 1998. The Art of Computer Programming (3rd ed.). Vol. 2. Addison-Wesley, Reading, MA.
    """
    x = a + b
    z = x - a
    y = (a - (x - z)) + (b - z)
    return x, y


def _fast_two_mult(a, b):
    """Error-free transformation of the product of two floating-point numbers
    such that a * b = x + y exactly.

    This function is faster than the _two_prod_fma function.

    Parameters
    ----------
    a, b : float
        The floating-point numbers to be multiplied.

    Returns
    -------
    tuple of float
        The product and the error term.

    References
    ----------
    Vincent Lefèvre, Nicolas Louvet, Jean-Michel Muller, Joris Picot, and Laurence Rideau. 2023.
    Accurate Calculation of Euclidean Norms Using Double-word Arithmetic.
    ACM Trans. Math. Softw. 49, 1, Article 1 (March 2023), 34 pages. https://doi.org/10.1145/3568672
    """
    import pyfma

    x = a * b
    y = pyfma.fma(a, b, -x)
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

    Reference
    ---------
    T. J. Dekker. A Floating-Point Technique for Extending the Available Precision.
    Numerische Mathematik, 18(3), 224–242,1971. 10.1007/BF01397083.
    Available at: https://doi.org/10.1007/BF01397083.
    """
    if abs(a) >= abs(b):
        x = a + b
        b_tile = x - a
        y = b - b_tile
        return x, y

    else:
        raise ValueError("|a| must be greater than or equal to |b|.")


def _comp_prod_fma(vec):
    """Compute the compensated product using Fused Multiply-Add (FMA).

    This function computes the product of elements in a vector using a
    compensated algorithm with Fused Multiply-Add to reduce numerical errors.

    Parameters
    ----------
    vec : list of float
        The vector whose elements are to be multiplied.

    Returns
    -------
    float
        The compensated product of the elements in the vector.

    Examples
    --------
    >>> _comp_prod_fma([1.1, 2.2, 3.3])
    7.986000000000001

    Reference
    ---------
    Takeshi Ogita, Siegfried M. Rump, and Shin'ichi Oishi. 2005. Accurate Sum and Dot Product.
    SIAM J. Sci. Comput. 26, 6 (2005), 1955–1988. https://doi.org/10.1137/030601818
    """
    import pyfma

    p1 = vec[0]
    e1 = 0.0
    for i in range(1, len(vec)):
        p_i, pi = _two_prod_fma(p1, vec[i])
        ei = pyfma.fma(e1, vec[i], pi)
        p1 = p_i
        e1 = ei
    res = p1 + e1
    return res


def _sum_of_squares_re(vec):
    """Compute the sum of squares of a vector using a compensated algorithm.

    This function calculates the sum of squares of the elements in a vector,
    employing a compensation technique to reduce numerical errors.

    Parameters
    ----------
    vec : list of float
        The vector whose elements' squares are to be summed.

    Returns
    -------
    float
        The compensated sum of the squares of the elements in the vector.

    Examples
    --------
    >>> _sum_of_squares_re([1.0, 2.0, 3.0])
    14.0

    Reference
    ---------
    Stef Graillat, Christoph Lauter, PING Tak Peter Tang,
    Naoya Yamanaka, and Shin’ichi Oishi. Efficient Calculations of Faith-
    fully Rounded L2-Norms of n-Vectors. ACM Transactions on Mathemat-
    ical Software, 41(4), Article 24, 2015. 10.1145/2699469. Available at:
    https://doi.org/10.1145/2699469.
    """
    P, p = _two_square(vec)
    S, s = _two_sum(P[0], P[1])
    for i in range(2, len(vec)):
        H, h = _two_sum(S, P[i])
        S, s = _two_sum(H, s + h)
    sump = sum(p)
    H, h = _two_sum(S, sump)
    S, s = _fast_two_sum(H, s + h)
    return S + s


def _vec_sum(p):
    """Compute the sum of a vector using a compensated summation algorithm.

    This function calculates the sum of the elements in a vector using a
    compensated summation algorithm to reduce numerical errors.

    Parameters
    ----------
    p : list of float
        The vector whose elements are to be summed.

    Returns
    -------
    float
        The compensated sum of the elements in the vector.

    Examples
    --------
    >>> _vec_sum([1.0, 2.0, 3.0])
    6.0

    Reference
    ---------
    Takeshi Ogita, Siegfried M. Rump, and Shin'ichi Oishi. 2005. Accurate Sum and Dot Product.
    SIAM J. Sci. Comput. 26, 6 (2005), 1955–1988. https://doi.org/10.1137/030601818
    """
    pi_1 = p[0]
    sigma_i1 = 0

    for i in range(1, len(p)):
        pi, qi = _two_sum(pi_1, p[i])
        sigma_i = sigma_i1 + qi
        pi_1 = pi
        sigma_i1 = sigma_i

    res = pi_1 + sigma_i1
    return res


def _norm_faithful(x):
    """Compute the faithful norm of a vector.

    This function calculates the faithful norm (L2 norm) of a vector,
    which is a more numerically stable version of the Euclidean norm.

    Parameters
    ----------
    x : list of float
        The vector whose norm is to be computed.

    Returns
    -------
    float
        The faithful norm of the vector.

    Examples
    --------
    >>> _norm_faithful([1.0, 2.0, 3.0])
    3.7416573867739413
    """
    return _norm_l(x)


def _norm_l(x):
    """Compute the L2 norm (Euclidean norm) of a vector using a compensated
    algorithm.

    This function calculates the L2 norm of a vector, employing a compensation
    technique to reduce numerical errors during the computation. It involves
    computing the sum of squares of the vector elements in a numerically stable way.

    Parameters
    ----------
    x : list of float
        The vector whose L2 norm is to be computed.

    Returns
    -------
    float
        The compensated L2 norm of the vector.

    Examples
    --------
    >>> _norm_l([1.0, 2.0, 3.0])
    3.7416573867739413

    Reference
    ---------
    Vincent Lef`evre, Nicolas Louvet, Jean-Michel Muller,
    Joris Picot, and Laurence Rideau. Accurate Calculation of Euclidean
    Norms Using Double-Word Arithmetic. ACM Transactions on Mathemat-
    ical Software, 49(1), 1–34, March 2023. 10.1145/3568672
    """
    P, p = _two_square(x)
    S, s = _two_sum(P[0], P[1])
    for i in range(2, len(x)):
        H, h = _two_sum(S, P[i])
        S, s = _two_sum(H, s + h)
    sump = sum(p)
    H, h = _two_sum(S, sump)
    S, s = _fast_two_sum(H, s + h)
    res = _acc_sqrt(S, s)
    return res


def _norm_g(x):
    """Compute the compensated Euclidean norm of a vector.

    This function calculates the Euclidean norm (L2 norm) of a vector,
    using a compensated algorithm to reduce numerical errors.

    Parameters
    ----------
    x : list of float
        The vector whose norm is to be computed.

    Returns
    -------
    float
        The compensated Euclidean norm of the vector.

    Examples
    --------
    >>> _norm_g([1.0, 2.0, 3.0])
    3.7416573867739413

    Reference
    ---------
    Stef Graillat, Christoph Lauter, PING Tak Peter Tang,
    Naoya Yamanaka, and Shin’ichi Oishi. Efficient Calculations of Faith-
    fully Rounded L2-Norms of n-Vectors. ACM Transactions on Mathemat-
    ical Software, 41(4), Article 24, 2015. 10.1145/2699469. Available at:
    https://doi.org/10.1145/2699469.
    """
    S = 0
    s = 0
    for x_i in x:
        P, p = _two_prod_fma(x_i, x_i)
        H, h = _two_sum(S, P)
        c = s + p
        d = h + c
        S, s = _fast_two_sum(H, d)
    res = _acc_sqrt(S, s)
    return res


def _two_square(Aa):
    """Compute the square of a number with a compensation for the round-off
    error.

    This function calculates the square of a given number and compensates
    for the round-off error that occurs during the squaring.

    Parameters
    ----------
    Aa : float
        The number to be squared.

    Returns
    -------
    tuple of float
        The square of the number and the compensated round-off error.

    Examples
    --------
    >>> _two_square(2.0)
    (4.0, 0.0)

    Reference
    ---------
    Siegfried Rump. Fast and accurate computation of the Euclidean norm of a vector. J
    apan Journal of Industrial and Applied Mathematics, 40, 2023. 10.1007/s13160-023-00593-8
    """
    P = Aa * Aa
    A, a = _split(Aa)
    p = a * a - ((P - A * A) - 2 * a * A)
    return P, p


def _acc_sqrt(T, t):
    """Compute the accurate square root of a number with a compensation for
    round-off error.

    This function calculates the square root of a number, taking into account
    a compensation term for the round-off error.

    Parameters
    ----------
    T : float
        The number whose square root is to be computed.
    t : float
        The compensation term for round-off error.

    Returns
    -------
    float
        The accurate square root of the number.

    Examples
    --------
    >>> _acc_sqrt(9.0, 0.0)
    3.0

    References
    ----------
    Vincent Lef`evre, Nicolas Louvet, Jean-Michel Muller,
    Joris Picot, and Laurence Rideau. Accurate Calculation of Euclidean
    Norms Using Double-Word Arithmetic. ACM Transactions on Mathematical Software, 49(1), 1–34, March 2023. 10.1145/3568672

    Marko Lange and Siegfried Rump. Faithfully Rounded
    Floating-point Computations. ACM Transactions on Mathematical Soft-
    ware, 46, 1-20, 2020. 10.1145/3290955
    """
    P = np.sqrt(T)
    H, h = _two_square(P)
    r = (T - H) - h
    r = t + r
    p = r / (2 * P)
    res = P + p
    return res


def _split(a):
    """Split a floating-point number into two parts: The rounded floating point
    presentation and its error. This can be utlized to substitute the FMA
    operation on the software level.

    Parameters
    ----------
    a : float
        The number to be split.

    Returns
    -------
    tuple of float
        The high and low precision parts of the number.

    Examples
    --------
    >>> _split(12345.6789)
    (12345.67578125, 0.00311875)

    Reference
    ---------
     T. J. Dekker. A Floating-Point Technique for Extending the Available Precision.
     Numerische Mathematik, 18(3), 224–242,
    1971. 10.1007/BF01397083. Available at: https://doi.org/10.1007/
    BF01397083.
    27
    """
    y = (2**27 + 1) * a
    x = y - (y - a)
    y = a - x
    return x, y
