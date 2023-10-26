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
    s,c = _two_prod_fma(v1[0], v2[0])
    for i in range(1, len(v1)):
        s, alpha, beta = _three_fma(v1[i], v2[i], s)
        c = c + (alpha + beta)
    return s + c

def _two_prod_fma(a, b):
    import pyfma
    x = a * b
    y = pyfma.fma(a, b, -x)
    return [x, y]

def _three_fma(a,b,c):
    # Reference:
    # Graillat, Stef & Langlois, Philippe & Louvet, Nicolas. (2006). Improving the compensated Horner scheme with a Fused Multiply and Add. 2. 1323-1327. 10.1145/1141277.1141585.
    import pyfma
    x = pyfma.fma(a, b, c)
    u1, u2 = _two_prod_fma(a, b)
    alph1, z = _two_sum(b, u2)
    beta1, beta2 = _two_sum(u1, alph1)
    y = (beta1 - x) + beta2
    return [x, y, z]

def _two_sum(a, b):
    x = a + b
    z = x - a
    y = (a - (x - z)) + (b + z)
    return [x, y]
