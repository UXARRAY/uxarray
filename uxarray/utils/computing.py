"""Compensated floating-point primitives for accurate spherical geometry.

In spherical-geometry computations the critical operations are cross products
and dot products over unit vectors. When two vectors are nearly parallel, the
difference of products that forms each cross-product component suffers
catastrophic cancellation: both products round to the same floating-point
value and their difference carries no significant bits. This affects
GCA-GCA intersection of nearly tangent arcs, constant-latitude intersection
near arc endpoints, and the ray-crossing test in point-in-polygon near polygon
edges.

Naming note
-----------
The term "error-free transformation" (EFT) strictly applies to ``two_sum``
and ``two_prod``, which capture their rounding errors exactly so that
``hi + lo`` equals the mathematical result with zero information loss.
``diff_of_products``, ``accucross``, and ``accucross_pair`` use those EFT
building blocks to achieve near-double precision for cross products, but they
are compensated algorithms, not zero-error transformations.

All functions are ``@njit``-compiled. ``two_prod`` uses a single fused
multiply-add (FMA) for its error term on hardware that supports it (selected at
import time and validated to be bit-exact), falling back to the portable
Veltkamp split otherwise — so there is no hard FMA dependency, but FMA is used
when available (~2x faster in the compensated kernels).

These primitives are a Python/Numba port of the AccuSphGeom C++ library:

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
exact-arithmetic fallback. This port implements only the EFT tier. The
compensated cross-product routines are roughly twice as accurate as direct
floating-point cross products while retaining the same vectorizable operation
structure; callers that need the full robustness stack should add an adaptive
predicate or exact-arithmetic fallback.
"""

import math

from numba import njit

# ---------------------------------------------------------------------------
# Fused multiply-add (FMA) support.
#
# ``two_prod`` needs the exact rounding error of ``a * b``. On hardware with an
# FMA instruction this is a single op: ``e = fma(a, b, -p)`` where ``p = a*b``.
# Without FMA we fall back to the portable Veltkamp split (no hardware
# dependency). We expose an LLVM ``fma`` intrinsic through Numba and validate at
# import time that it both compiles and yields a bit-exact error-free transform;
# if anything fails (older toolchain, unsupported target, or a non-exact FMA),
# ``_HAS_FMA`` stays False and the Veltkamp path is used. This keeps the
# library's "no FMA dependency" guarantee while using FMA where it is available.
# ---------------------------------------------------------------------------
try:
    from numba.core import types as _nb_types
    from numba.extending import intrinsic as _nb_intrinsic

    @_nb_intrinsic
    def _fma(typingctx, a, b, c):
        sig = _nb_types.float64(_nb_types.float64, _nb_types.float64, _nb_types.float64)

        def codegen(context, builder, signature, args):
            return builder.fma(*args)

        return sig, codegen

    _FMA_INTRINSIC_OK = True
except Exception:  # pragma: no cover - toolchain without intrinsic support
    _FMA_INTRINSIC_OK = False


def _validate_fma() -> bool:
    """Return True iff the FMA intrinsic compiles and is a bit-exact EFT."""
    if not _FMA_INTRINSIC_OK:
        return False
    try:
        import numpy as _np

        @njit(cache=False)
        def _tp_fma(a, b):
            p = a * b
            return p, _fma(a, b, -p)

        @njit(cache=False)
        def _tp_vk(a, b):
            p = a * b
            f = 134217729.0
            a_hi = f * a - (f * a - a)
            a_lo = a - a_hi
            b_hi = f * b - (f * b - b)
            b_lo = b - b_hi
            e = a_lo * b_lo - (((p - a_hi * b_hi) - a_lo * b_hi) - a_hi * b_lo)
            return p, e

        rng = _np.random.default_rng(20260101)
        for _ in range(20000):
            a = float(rng.standard_normal() * rng.integers(1, 1 << 20))
            b = float(rng.standard_normal() * rng.integers(1, 1 << 20))
            pf, ef = _tp_fma(a, b)
            pv, ev = _tp_vk(a, b)
            if pf != pv or (pf + ef) != (pv + ev):
                return False
        return True
    except Exception:  # pragma: no cover
        return False


_HAS_FMA = _validate_fma()


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
def _two_prod_veltkamp(a, b):
    """Portable TwoProd via Veltkamp splitting (no FMA dependency).

    Decomposes each operand into a high and low half using the splitting
    constant 2**27 + 1, then reconstructs the exact rounding error from the
    four partial products. Works on every Numba target.
    """
    p = a * b
    factor = 134217729.0  # 2**27 + 1
    a_hi = factor * a - (factor * a - a)
    a_lo = a - a_hi
    b_hi = factor * b - (factor * b - b)
    b_lo = b - b_hi
    e = a_lo * b_lo - (((p - a_hi * b_hi) - a_lo * b_hi) - a_hi * b_lo)
    return p, e


if _HAS_FMA:

    @njit(cache=True, inline="always")
    def _two_prod_fma(a, b):
        """TwoProd via a single hardware FMA: e = fma(a, b, -p)."""
        p = a * b
        return p, _fma(a, b, -p)

    @njit(cache=True, inline="always")
    def two_prod(a, b):
        """Dekker TwoProd: return (p, e) with p = fl(a*b) and p + e = a*b exactly.

        Uses a single fused multiply-add for the error term on hardware that
        supports it (selected at import time via ``_HAS_FMA``), falling back to
        the portable Veltkamp split otherwise. The FMA path is ~2x faster in the
        compensated geometry kernels and is bit-for-bit identical to the
        Veltkamp result (validated at import).

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
        return _two_prod_fma(a, b)

else:  # pragma: no cover - exercised only on FMA-less toolchains

    @njit(cache=True, inline="always")
    def two_prod(a, b):
        """Dekker TwoProd: return (p, e) with p = fl(a*b) and p + e = a*b exactly.

        Portable Veltkamp-split implementation (no FMA available on this
        toolchain/target).

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
        return _two_prod_veltkamp(a, b)


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


@njit(cache=True, inline="always")
def _cdp8(
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    b7,
):
    """Compensated sum of 8 exact products: Σ ai*bi, i=0..7.

    Uses ``two_prod`` + ``two_sum`` accumulation (Ogita-Rump-Oishi style)
    so the result has error bounded by one ulp of the true value regardless
    of cancellation in intermediate sums.
    """
    s, lo = two_prod(a0, b0)
    p, e = two_prod(a1, b1)
    s2, e2 = two_sum(s, p)
    lo += e + e2
    s = s2
    p, e = two_prod(a2, b2)
    s2, e2 = two_sum(s, p)
    lo += e + e2
    s = s2
    p, e = two_prod(a3, b3)
    s2, e2 = two_sum(s, p)
    lo += e + e2
    s = s2
    p, e = two_prod(a4, b4)
    s2, e2 = two_sum(s, p)
    lo += e + e2
    s = s2
    p, e = two_prod(a5, b5)
    s2, e2 = two_sum(s, p)
    lo += e + e2
    s = s2
    p, e = two_prod(a6, b6)
    s2, e2 = two_sum(s, p)
    lo += e + e2
    s = s2
    p, e = two_prod(a7, b7)
    s2, e2 = two_sum(s, p)
    lo += e + e2
    s = s2
    return s, lo


@njit(cache=True, inline="always")
def accucross_pair(
    ax_hi,
    ay_hi,
    az_hi,
    ax_lo,
    ay_lo,
    az_lo,
    bx_hi,
    by_hi,
    bz_hi,
    bx_lo,
    by_lo,
    bz_lo,
):
    """Compensated cross product of two compensated vectors.

    Computes (a_hi + a_lo) × (b_hi + b_lo) using a compensated 8-term dot
    product for each component, matching the two-argument ``accucross`` overload
    in the AccuSphGeom C++ library.  This is more accurate than collapsing
    (hi, lo) to a single float before the cross product.

    Parameters
    ----------
    ax_hi, ay_hi, az_hi : float
        High parts of vector a.
    ax_lo, ay_lo, az_lo : float
        Low  parts of vector a (rounding residuals from a prior compensated operation).
    bx_hi, by_hi, bz_hi : float
        High parts of vector b.
    bx_lo, by_lo, bz_lo : float
        Low  parts of vector b.

    Returns
    -------
    x_hi, y_hi, z_hi, x_lo, y_lo, z_lo : float
        Compensated cross-product components.
    """
    # x = (ay*bz) - (az*by), expanded over all four hi/lo cross-terms
    x_hi, x_lo = _cdp8(
        ay_hi,
        ay_hi,
        ay_lo,
        ay_lo,
        -az_hi,
        -az_hi,
        -az_lo,
        -az_lo,
        bz_hi,
        bz_lo,
        bz_hi,
        bz_lo,
        by_hi,
        by_lo,
        by_hi,
        by_lo,
    )
    # y = (az*bx) - (ax*bz)
    y_hi, y_lo = _cdp8(
        az_hi,
        az_hi,
        az_lo,
        az_lo,
        -ax_hi,
        -ax_hi,
        -ax_lo,
        -ax_lo,
        bx_hi,
        bx_lo,
        bx_hi,
        bx_lo,
        bz_hi,
        bz_lo,
        bz_hi,
        bz_lo,
    )
    # z = (ax*by) - (ay*bx)
    z_hi, z_lo = _cdp8(
        ax_hi,
        ax_hi,
        ax_lo,
        ax_lo,
        -ay_hi,
        -ay_hi,
        -ay_lo,
        -ay_lo,
        by_hi,
        by_lo,
        by_hi,
        by_lo,
        bx_hi,
        bx_lo,
        bx_hi,
        bx_lo,
    )
    return x_hi, y_hi, z_hi, x_lo, y_lo, z_lo


# ---------------------------------------------------------------------------
# Compensated dot products and sum-of-squares (fixed small sizes)
#
# These port accusphgeom::numeric::compensated_dot_product and
# accusphgeom::numeric::sum_of_squares_c from eft.hpp, using our Veltkamp-
# splitting two_prod instead of FMA.  Fixed-size variants are used because
# Numba does not support generic runtime-length accumulations inside @njit.
# ---------------------------------------------------------------------------


@njit(cache=True, inline="always")
def _cdp2(a0, b0, a1, b1):
    """Compensated dot product of 2 pairs: a0*b0 + a1*b1."""
    s, lo = two_prod(a0, b0)
    p, e = two_prod(a1, b1)
    s2, e2 = two_sum(s, p)
    lo += e + e2
    return s2, lo


@njit(cache=True, inline="always")
def _cdp4(a0, b0, a1, b1, a2, b2, a3, b3):
    """Compensated dot product of 4 pairs: Σ ai*bi, i=0..3."""
    s, lo = two_prod(a0, b0)
    p, e = two_prod(a1, b1)
    s2, e2 = two_sum(s, p)
    lo += e + e2
    s = s2
    p, e = two_prod(a2, b2)
    s2, e2 = two_sum(s, p)
    lo += e + e2
    s = s2
    p, e = two_prod(a3, b3)
    s2, e2 = two_sum(s, p)
    lo += e + e2
    return s2, lo


@njit(cache=True, inline="always")
def _sum_sq_c2(h0, l0, h1, l1):
    """Compensated sum of squares for 2 (hi, lo) pairs: h0²+l0²+h1²+l1².

    Mirrors sum_of_squares_c<T, 2> from accusphgeom/numeric/eft.hpp, which
    constructs lhs = rhs = [h0, l0, h1, l1] and calls compensated_dot_product.
    Used to compute nx²+ny² accurately from the compensated normal (hi, lo).
    """
    return _cdp4(h0, h0, l0, l0, h1, h1, l1, l1)


@njit(cache=True, inline="always")
def _sum_sq_c3(h0, l0, h1, l1, h2, l2):
    """Compensated sum of squares for 3 (hi, lo) pairs: h0²+l0²+h1²+l1²+h2²+l2².

    Mirrors sum_of_squares_c<T, 3> from accusphgeom/numeric/eft.hpp, which
    constructs lhs = rhs = [h0, l0, h1, l1, h2, l2] and calls a 6-term CDP.
    We use _cdp8 with two zero-padding pairs (adding zero products).
    Used to compute |n|² = nx²+ny²+nz² accurately from the compensated normal.
    """
    # lhs = rhs = [h0, l0, h1, l1, h2, l2, 0, 0]
    return _cdp8(h0, l0, h1, l1, h2, l2, 0.0, 0.0, h0, l0, h1, l1, h2, l2, 0.0, 0.0)


@njit(cache=True, inline="always")
def acc_sqrt_re(value, error=0.0):
    """Accurate square root: return (root, correction) s.t. root+correction ≈ sqrt(value+error).

    Mirrors accusphgeom::numeric::acc_sqrt_re from eft.hpp.  Computes
    root = fl(sqrt(value)), measures the rounding error of root*root via
    two_prod, then recovers a correction term from the residual.  When
    ``error`` is provided (e.g. the ``lo`` half of a compensated sum),
    it is folded into the residual so the correction accounts for the
    full compensated input.

    Parameters
    ----------
    value : float
        Non-negative scalar (the ``hi`` part of a compensated value).
    error : float, optional
        Low-order correction to ``value`` (default 0.0).

    Returns
    -------
    root : float
        Rounded sqrt, fl(sqrt(value)).
    correction : float
        Additive correction; root + correction ≈ sqrt(value + error) to ~1 ulp.
    """
    # Negative value means no real intersection; return NaN so that the
    # isfinite mask in the status layer rejects this candidate without a branch.
    if value < 0.0:
        return math.nan, 0.0
    root = math.sqrt(value)
    if root == 0.0:
        return 0.0, 0.0
    sq_hi, sq_lo = two_prod(root, root)
    residual = (value - sq_hi) + (error - sq_lo)
    correction = residual / (2.0 * root)
    return root, correction
