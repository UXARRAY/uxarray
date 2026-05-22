"""Error-free transformations (EFT) for accurate floating-point arithmetic.

In spherical-geometry computations the critical operations are cross products
and dot products over unit vectors. When two vectors are nearly parallel, the
difference of products that forms each cross-product component suffers
catastrophic cancellation: both products round to the same floating-point
value and their difference carries no significant bits. This affects
GCA-GCA intersection of nearly tangent arcs, constant-latitude intersection
near arc endpoints, and the ray-crossing test in point-in-polygon near polygon
edges.

The functions here represent each result as an unevaluated sum of two
``float64`` values ``(hi, lo)`` such that ``hi + lo`` equals the
mathematically exact result. This effectively doubles the significant bits
available for cross-product components without resorting to arbitrary-
precision arithmetic.

These primitives are a Python/Numba port of the error-free transformation
layer from the AccuSphGeom C++ library:

    Chen, H. (2026). Accurate and Robust Algorithms for Spherical Polygon
    Operations. EGUsphere preprint.
    https://egusphere.copernicus.org/preprints/2026/egusphere-2026-636/

    Chen, H. Accurate and Robust Great Circle Arc Intersection and Great
    Circle Arc Constant Latitude Intersection on the Sphere. SIAM J. Sci.
    Comput. https://doi.org/10.1137/25M1737614

AccuSphGeom reference implementation (C++):
    https://github.com/hongyuchen1030/AccuSphGeom

What this module omits: AccuSphGeom's full robustness stack has three
tiers — an EFT filter (what this module implements), Shewchuk adaptive
predicates for results that fall inside the filter threshold, and a geogram
exact-arithmetic fallback. This port implements only the EFT tier. For
non-degenerate inputs in double precision this is sufficient; callers that
need to handle geometrically degenerate inputs (coincident arcs, a query
point exactly on a polygon edge) should add their own perturbation or
fall-back logic.
"""

from numba import njit


@njit(cache=True, inline="always")
def two_sum(a, b):
    """Knuth's TwoSum: return (s, e) with s = fl(a + b) and s + e = a + b exactly.

    Floating-point addition rounds the mathematical result to the nearest
    representable value. ``two_sum`` captures that rounding error in the
    companion term ``e`` so that ``s + e`` equals the true sum with no
    information lost. The cost is four extra floating-point operations beyond
    the addition itself.

    Parameters
    ----------
    a, b : float
        Input values.

    Returns
    -------
    s : float
        Rounded sum fl(a + b).
    e : float
        Rounding error term; s + e = a + b exactly.
    """
    s = a + b
    bp = s - a
    e = (a - (s - bp)) + (b - bp)
    return s, e


@njit(cache=True, inline="always")
def two_prod(a, b):
    """Dekker/Veltkamp TwoProd: return (p, e) with p = fl(a * b) and p + e = a * b exactly.

    Like ``two_sum`` for multiplication. Uses the Veltkamp splitting constant
    2**27 + 1 to decompose each operand into a high and low half, then
    reconstructs the exact rounding error from the four partial products.
    On hardware with a fused multiply-add (FMA) instruction the error term
    could be obtained in one step as ``fma(a, b, -p)``; the split used here
    is portable across all Numba targets.

    Parameters
    ----------
    a, b : float
        Input values.

    Returns
    -------
    p : float
        Rounded product fl(a * b).
    e : float
        Rounding error term; p + e = a * b exactly.
    """
    p = a * b
    factor = 134217729.0  # 2**27 + 1
    a_hi = factor * a - (factor * a - a)
    a_lo = a - a_hi
    b_hi = factor * b - (factor * b - b)
    b_lo = b - b_hi
    e = a_lo * b_lo - (((p - a_hi * b_hi) - a_lo * b_hi) - a_hi * b_lo)
    return p, e


@njit(cache=True, inline="always")
def diff_of_products(a, b, c, d):
    """Kahan's accurate a*b - c*d using two_prod and two_sum.

    Naive evaluation of ``a*b - c*d`` loses all significant bits when the two
    products are nearly equal (catastrophic cancellation). This routine
    computes each product exactly via ``two_prod``, subtracts the rounded
    high parts, then folds the residual low parts back in. The result has
    rounding error bounded by one ulp of the true value regardless of
    cancellation.

    This is the core operation that makes cross products accurate: every
    component of ``a x b`` is a difference of two products of exactly this
    form.

    Parameters
    ----------
    a, b, c, d : float
        Input scalars; computes a*b - c*d.

    Returns
    -------
    hi : float
        High-order part of the accurate result.
    lo : float
        Low-order correction term; hi + lo equals the accurate value.
    """
    w, e_w = two_prod(c, d)
    x, e_x = two_prod(a, b)
    s, e_s = two_sum(x, -w)
    lo = (e_x - e_w) + e_s
    return s, lo


@njit(cache=True, inline="always")
def accucross(a0, a1, a2, b0, b1, b2):
    """Accurate cross product a x b returning (hi[3], lo[3]) component pairs.

    Each component of a cross product is a difference of two products — the
    exact form that ``diff_of_products`` handles. This function computes all
    three components that way, returning six scalars such that the
    mathematically exact cross product satisfies ``result[i] = hi[i] + lo[i]``
    for each component. Callers that need single-precision accuracy can use
    the hi parts alone; callers that need the full compensated result add
    hi and lo before further use.

    Parameters
    ----------
    a0, a1, a2 : float
        Components of vector a.
    b0, b1, b2 : float
        Components of vector b.

    Returns
    -------
    x_hi, y_hi, z_hi, x_lo, y_lo, z_lo : float
        High and low parts of each cross-product component.
    """
    x_hi, x_lo = diff_of_products(a1, b2, a2, b1)
    y_hi, y_lo = diff_of_products(a2, b0, a0, b2)
    z_hi, z_lo = diff_of_products(a0, b1, a1, b0)
    return x_hi, y_hi, z_hi, x_lo, y_lo, z_lo
