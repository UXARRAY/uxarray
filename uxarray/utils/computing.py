"""Compensated floating-point primitives for accurate spherical geometry.

Cross and dot products over nearly-parallel unit vectors suffer catastrophic
cancellation in double precision, which degrades GCA-GCA and constant-latitude
intersections and the point-in-polygon ray test. These ``@njit`` primitives
recover near-double precision using error-free transformations (``two_sum``,
``two_prod``) and compensated algorithms built on them. ``two_prod`` uses a
hardware FMA when one is available (validated bit-exact at import time) and
falls back to the Veltkamp split otherwise, so there is no hard FMA dependency.

Python/Numba port of the AccuSphGeom C++ library (EFT tier only; the adaptive
Shewchuk predicate and exact-arithmetic fallback tiers are not ported):

    Chen, H. (2026). Accurate and Robust Algorithms for Spherical Polygon
    Operations. EGUsphere preprint egusphere-2026-636.
    Chen, H. Great Circle Arc Intersection and Constant Latitude Intersection
    on the Sphere. SIAM J. Sci. Comput. https://doi.org/10.1137/25M1737614
    Reference implementation: https://github.com/hongyuchen1030/AccuSphGeom
"""

import math

from numba import njit

# Fused multiply-add (FMA) support. ``two_prod`` needs the exact rounding error
# of ``a * b``; with hardware FMA this is ``e = fma(a, b, -p)``, otherwise we
# fall back to the portable Veltkamp split. The LLVM ``fma`` intrinsic is
# validated at import time; if it is unavailable or non-exact, ``_HAS_FMA``
# stays False and the Veltkamp path is used.
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


if _FMA_INTRINSIC_OK:

    @njit(cache=True, inline="always")
    def _two_prod_fma(a, b):
        p = a * b
        return p, _fma(a, b, -p)


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


def _validate_fma(n_samples=2000) -> bool:
    """Return True iff the FMA intrinsic compiles and is a bit-exact EFT."""
    if not _FMA_INTRINSIC_OK:
        return False
    try:
        import numpy as _np

        rng = _np.random.default_rng(20260101)
        for _ in range(n_samples):
            a = float(rng.standard_normal() * rng.integers(1, 1 << 20))
            b = float(rng.standard_normal() * rng.integers(1, 1 << 20))
            pf, ef = _two_prod_fma(a, b)
            pv, ev = _two_prod_veltkamp(a, b)
            # Compare the residuals *directly*. Do not compare ``pf + ef``
            # against ``pv + ev``: both sums round straight back to the product
            # (|e| <= ulp(p)/2 by construction), so that predicate collapses to
            # ``pf != pv`` -- a tautology, since both are fl(a*b). The exact
            # residual is unique and representable, so a correct FMA and the
            # Veltkamp split must agree bit-for-bit.
            if pf != pv or ef != ev:
                return False
        return True
    except Exception:  # pragma: no cover
        return False


_HAS_FMA = _validate_fma()


if _HAS_FMA:

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
    lo = e_x + (e_s - e_w)
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


# Compensated dot products and sum-of-squares, fixed small sizes (Numba does not
# support generic runtime-length accumulations inside @njit).


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
def _fast_two_sum(a, b):
    """Fast TwoSum: (x, y) with x = fl(a + b), x + y = a + b exactly.

    Requires ``|a| >= |b|`` (or ``a == 0``) for the error term to be exact —
    the callers here satisfy this because ``a`` is a running non-negative sum.
    """
    x = a + b
    return x, (a - x) + b


@njit(cache=True, inline="always")
def _sum_non_neg(a_hi, a_lo, b_hi, b_lo):
    """Add two non-negative compensated values (mirrors accusphgeom ``sum_non_neg``)."""
    hh, h = two_sum(a_hi, b_hi)
    d = h + (a_lo + b_lo)
    return _fast_two_sum(hh, d)


@njit(cache=True, inline="always")
def _sum_of_squares_c(hi, lo):
    """Compensated squared norm ``Σ (hi[i] + lo[i])²`` of a compensated vector.

    Faithful port of ``sum_of_squares_c<T, N>`` from
    accusphgeom/numeric/eft.hpp: a compensated ``Σ hi²`` plus the cross-term
    correction ``2·Σ hi·li``.

    Parameters
    ----------
    hi, lo : tuple of float
        Equal-length tuples of the high and low parts of each vector component.
        Numba specializes this per tuple length at compile time and keeps the
        tuples register-resident (no allocation) — the direct analog of the C++
        template. Used for both nx²+ny² (denominator) and nx²+ny²+nz² (|n|²).

    Returns
    -------
    tuple of float
        The compensated squared norm as a ``(hi, lo)`` pair.
    """
    n = len(hi)
    s_hi = 0.0
    s_lo = 0.0
    for i in range(n):  # compensated Σ hi²
        ph, pl = two_prod(hi[i], hi[i])
        s_hi, s_lo = _sum_non_neg(s_hi, s_lo, ph, pl)
    r_hi, r_lo = two_prod(hi[0], lo[0])  # accurate Σ hi·li
    for i in range(1, n):
        p, e = two_prod(hi[i], lo[i])
        r_hi, e2 = two_sum(r_hi, p)
        r_lo += e + e2
    return _fast_two_sum(s_hi, (2.0 * (r_hi + r_lo)) + s_lo)


@njit(cache=True, inline="always", error_model="numpy")
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
    # Branch-free, matching AccuSphGeom acc_sqrt_re exactly. Negative value
    # yields nan via math.sqrt and root==0 yields nan via the 0/0 correction,
    # both under error_model="numpy"; the isfinite mask in the status layer
    # rejects such candidates.
    root = math.sqrt(value)
    sq_hi, sq_lo = two_prod(root, root)
    # Residual accumulation order matches AccuSphGeom acc_sqrt_re exactly:
    # (value - square.hi) - square.lo + error.
    residual = (value - sq_hi) - sq_lo + error
    correction = residual / (2.0 * root)
    return root, correction
